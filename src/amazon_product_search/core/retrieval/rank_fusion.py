from dataclasses import dataclass
from typing import Any, Literal

from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.retrieval.score_normalizer import min_max_scale
from amazon_product_search.core.retrieval.weighting_strategy import DynamicWeighting, FixedWeighting


@dataclass
class RankFusion:
    fuser: Literal["search_engine", "own"] = "search_engine"
    fusion_strategy: Literal["fuse", "append"] = "fuse"
    # When fusion_strategy is "fuse", the following options are available.
    enable_score_normalization: bool = False
    rrf: bool | int = False
    weighting_strategy: Literal["fixed", "dynamic"] = "fixed"


def _normalize_scores(response: Response) -> Response:
    """Normalize scores in a response.

    Args:
        response (Response): A response from retrieval.

    Returns:
        Response: A response with normalized scores.
    """
    scores = [result.score for result in response.results]
    normalized_scores = min_max_scale(scores, min_val=0)
    results = [
        Result(product=result.product, score=normalized_score)
        for result, normalized_score in zip(response.results, normalized_scores, strict=True)
    ]
    return Response(results=results, total_hits=response.total_hits)


def _rrf_scores(response: Response, k: int = 60) -> Response:
    """Adjust scores in a response using RRF (Reciprocal Rank Fusion).

    Args:
        response (Response): A response from retrieval.
        k (int, optional): The ranking constant. Defaults to 60.

    Returns:
        Response: A response with adjusted scores.
    """
    adjusted_scores = []
    for i in range(len(response.results)):
        adjusted_scores.append(1 / (k + i + 1))
    results = [
        Result(product=result.product, score=adjusted_score)
        for result, adjusted_score in zip(response.results, adjusted_scores, strict=True)
    ]
    return Response(results=results, total_hits=response.total_hits)


def _append_results(original_response: Response, alternative_response: Response, size: int) -> Response:
    """Return a response with results from alternative_response appended to original_response.

    Args:
        original_response (Response): The original response.
        alternative_response (Response): An alternative response to append if necessary.
        size (int): The number of results to return.

    Returns:
        Response: The response with results from alternative_response appended to original_response.
    """
    results = original_response.results
    total_hits = original_response.total_hits
    if len(results) < size:
        results += alternative_response.results[: size - len(results)]
        total_hits = len(results)
    return Response(results=results, total_hits=total_hits)


def _merge_responses_by_score(sparse_response: Response, dense_response: Response, size: int) -> Response:
    """Merge two responses by score.

    Args:
        sparse_response (Response): A response from sparse retrieval.
        dense_response (Response): A response from dense retrieval.

    Returns:
        Response: A merged response.
    """
    id_to_product: dict[str, dict[str, Any]] = {}
    sparse_results: dict[str, float] = {}
    dense_results: dict[str, float] = {}

    for result in sparse_response.results:
        product_id = result.product["product_id"]
        id_to_product[product_id] = result.product
        sparse_results[product_id] = result.score
    for result in dense_response.results:
        product_id = result.product["product_id"]
        id_to_product[product_id] = result.product
        dense_results[product_id] = result.score

    results: list[Result] = []
    for product_id in sparse_results.keys() | dense_results.keys():
        score = sparse_results.get(product_id, 0) + dense_results.get(product_id, 0)
        result = Result(product=id_to_product[product_id], score=score)
        results.append(result)
    total_hits = max(sparse_response.total_hits, dense_response.total_hits, len(results))

    results = sorted(results, key=lambda result: (result.score, result.product["product_id"]), reverse=True)[:size]
    return Response(results=results, total_hits=total_hits)


def fuse(
    query: str,
    sparse_response: Response,
    dense_response: Response,
    sparse_boost: float,
    dense_boost: float,
    rank_fusion: RankFusion,
    size: int,
) -> Response:
    if rank_fusion.fusion_strategy == "append":
        return _append_results(sparse_response, dense_response, size)

    if rank_fusion.enable_score_normalization:
        sparse_response = _normalize_scores(sparse_response)
        dense_response = _normalize_scores(dense_response)
    elif rank_fusion.rrf:
        if isinstance(rank_fusion.rrf, bool):
            sparse_response = _rrf_scores(sparse_response)
            dense_response = _rrf_scores(dense_response)
        else:
            sparse_response = _rrf_scores(sparse_response, k=rank_fusion.rrf)
            dense_response = _rrf_scores(dense_response, k=rank_fusion.rrf)

    if rank_fusion.weighting_strategy:
        weighting_strategy = (
            FixedWeighting({"sparse": sparse_boost, "dense": dense_boost})
            if rank_fusion.weighting_strategy == "fixed"
            else DynamicWeighting()
        )
        for result in sparse_response.results:
            result.score *= weighting_strategy.apply("sparse", query)
        for result in dense_response.results:
            result.score *= weighting_strategy.apply("dense", query)

    return _merge_responses_by_score(sparse_response, dense_response, size)
