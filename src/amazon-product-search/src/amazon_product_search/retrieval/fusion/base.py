import logging
from typing import Any

from amazon_product_search.retrieval.core.protocols import ResultFuser
from amazon_product_search.retrieval.core.types import FusionConfig, RetrievalResponse
from amazon_product_search.retrieval.response import Result
from amazon_product_search.retrieval.score_normalizer import min_max_scale

logger = logging.getLogger(__name__)


class FlexibleResultFuser(ResultFuser):
    """Flexible fusion system that can combine any number of retrieval responses."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    def fuse(self, responses: list[RetrievalResponse], weights: dict[str, float] | None = None) -> RetrievalResponse:
        """Fuse multiple retrieval responses using configured strategy."""
        if not responses:
            return RetrievalResponse(results=[], total_hits=0, engine_name="fused")

        if len(responses) == 1:
            return responses[0]

        logger.debug(f"Fusing {len(responses)} responses using method '{self.config.method}'")

        # Apply score normalization if needed
        normalized_responses = self._normalize_scores(responses)

        # Apply weights
        weighted_responses = self._apply_weights(normalized_responses, weights)

        # Combine responses
        if self.config.method == "weighted_sum":
            fused_response = self._weighted_sum_fusion(weighted_responses)
        elif self.config.method == "rrf":
            fused_response = self._rrf_fusion(weighted_responses)
        elif self.config.method == "borda_count":
            fused_response = self._borda_count_fusion(weighted_responses)
        elif self.config.method == "max":
            fused_response = self._max_fusion(weighted_responses)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.method}")

        fused_response.engine_name = "fused"
        fused_response.metadata["fusion_method"] = self.config.method
        fused_response.metadata["num_responses"] = len(responses)
        fused_response.metadata["source_engines"] = [r.engine_name for r in responses]

        return fused_response

    def _normalize_scores(self, responses: list[RetrievalResponse]) -> list[RetrievalResponse]:
        """Apply score normalization to responses."""
        if self.config.normalization == "none":
            return responses

        normalized_responses = []
        for response in responses:
            if not response.results:
                normalized_responses.append(response)
                continue

            scores = [result.score for result in response.results]

            if self.config.normalization == "min_max":
                normalized_scores = min_max_scale(scores, min_val=0)
            elif self.config.normalization == "z_score":
                normalized_scores = self._z_score_normalize(scores)
            elif self.config.normalization == "rank_based":
                normalized_scores = self._rank_based_normalize(len(scores))
            else:
                normalized_scores = scores

            # Create new response with normalized scores
            normalized_results = [
                Result(
                    product=result.product,
                    score=norm_score,
                    explanation=result.explanation
                )
                for result, norm_score in zip(response.results, normalized_scores, strict=True)
            ]

            normalized_response = RetrievalResponse(
                results=normalized_results,
                total_hits=response.total_hits,
                engine_name=response.engine_name,
                processing_time_ms=response.processing_time_ms,
                metadata=response.metadata.copy()
            )
            normalized_responses.append(normalized_response)

        return normalized_responses

    def _apply_weights(
        self, responses: list[RetrievalResponse], weights: dict[str, float] | None
    ) -> list[RetrievalResponse]:
        """Apply weights to responses."""
        if not weights:
            # Use config weights or equal weights
            weights = self.config.weights if self.config.weights else {r.engine_name: 1.0 for r in responses}

        weighted_responses = []
        for response in responses:
            weight = weights.get(response.engine_name, 1.0)
            if weight != 1.0 and response.results:
                # Apply weight to scores
                weighted_results = [
                    Result(
                        product=result.product,
                        score=result.score * weight,
                        explanation=result.explanation
                    )
                    for result in response.results
                ]

                weighted_response = RetrievalResponse(
                    results=weighted_results,
                    total_hits=response.total_hits,
                    engine_name=response.engine_name,
                    processing_time_ms=response.processing_time_ms,
                    metadata=response.metadata.copy()
                )
                weighted_responses.append(weighted_response)
            else:
                weighted_responses.append(response)

        return weighted_responses

    def _weighted_sum_fusion(self, responses: list[RetrievalResponse]) -> RetrievalResponse:
        """Combine responses using weighted sum of scores."""
        product_scores: dict[str, float] = {}
        product_data: dict[str, dict[str, Any]] = {}
        total_hits = 0

        for response in responses:
            total_hits = max(total_hits, response.total_hits)
            for result in response.results:
                product_id = result.product["product_id"]
                product_data[product_id] = result.product
                product_scores[product_id] = product_scores.get(product_id, 0) + result.score

        # Sort by combined score
        sorted_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

        # Create results
        results = [
            Result(
                product=product_data[product_id],
                score=score,
                explanation={"fusion_method": "weighted_sum"}
            )
            for product_id, score in sorted_items
        ]

        return RetrievalResponse(
            results=results,
            total_hits=total_hits
        )

    def _rrf_fusion(self, responses: list[RetrievalResponse]) -> RetrievalResponse:
        """Combine responses using Reciprocal Rank Fusion."""
        product_scores: dict[str, float] = {}
        product_data: dict[str, dict[str, Any]] = {}
        total_hits = 0
        k = self.config.ranking_constant

        for response in responses:
            total_hits = max(total_hits, response.total_hits)
            for rank, result in enumerate(response.results):
                product_id = result.product["product_id"]
                product_data[product_id] = result.product
                rrf_score = 1.0 / (k + rank + 1)
                product_scores[product_id] = product_scores.get(product_id, 0) + rrf_score

        # Sort by RRF score
        sorted_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            Result(
                product=product_data[product_id],
                score=score,
                explanation={"fusion_method": "rrf", "k": k}
            )
            for product_id, score in sorted_items
        ]

        return RetrievalResponse(
            results=results,
            total_hits=total_hits
        )

    def _borda_count_fusion(self, responses: list[RetrievalResponse]) -> RetrievalResponse:
        """Combine responses using Borda count method."""
        product_scores: dict[str, int] = {}
        product_data: dict[str, dict[str, Any]] = {}
        total_hits = 0

        for response in responses:
            total_hits = max(total_hits, response.total_hits)
            max_rank = len(response.results)
            for rank, result in enumerate(response.results):
                product_id = result.product["product_id"]
                product_data[product_id] = result.product
                borda_score = max_rank - rank
                product_scores[product_id] = product_scores.get(product_id, 0) + borda_score

        # Sort by Borda score
        sorted_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            Result(
                product=product_data[product_id],
                score=float(score),
                explanation={"fusion_method": "borda_count"}
            )
            for product_id, score in sorted_items
        ]

        return RetrievalResponse(
            results=results,
            total_hits=total_hits
        )

    def _max_fusion(self, responses: list[RetrievalResponse]) -> RetrievalResponse:
        """Combine responses using maximum score."""
        product_scores: dict[str, float] = {}
        product_data: dict[str, dict[str, Any]] = {}
        total_hits = 0

        for response in responses:
            total_hits = max(total_hits, response.total_hits)
            for result in response.results:
                product_id = result.product["product_id"]
                product_data[product_id] = result.product
                current_max = product_scores.get(product_id, 0)
                product_scores[product_id] = max(current_max, result.score)

        # Sort by max score
        sorted_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            Result(
                product=product_data[product_id],
                score=score,
                explanation={"fusion_method": "max"}
            )
            for product_id, score in sorted_items
        ]

        return RetrievalResponse(
            results=results,
            total_hits=total_hits
        )

    def _z_score_normalize(self, scores: list[float]) -> list[float]:
        """Normalize scores using z-score."""
        if len(scores) <= 1:
            return scores

        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return [0.0] * len(scores)

        return [(score - mean_score) / std_dev for score in scores]

    def _rank_based_normalize(self, num_results: int) -> list[float]:
        """Normalize using rank-based scoring."""
        return [1.0 / (i + 1) for i in range(num_results)]
