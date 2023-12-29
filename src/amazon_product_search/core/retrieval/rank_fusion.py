from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal

from amazon_product_search.core.es.response import Response, Result
from amazon_product_search.core.retrieval.score_normalizer import min_max_scale
from amazon_product_search.core.retrieval.weighting_strategy import DynamicWeighting, FixedWeighting

_ScoreTransformationMethod = Literal["min_max", "rrf", "borda"]

ScoreTransformationMethod = _ScoreTransformationMethod | list[_ScoreTransformationMethod] | None  # type: ignore

FusionStrategy = Literal["sum", "max", "append"]


def score_transformation_method_to_str(score_transformation_method: ScoreTransformationMethod) -> str:
    if isinstance(score_transformation_method, str):
        return {
            "min_max": "MM",
            "rrf": "RRF",
            "borda": "Borda",
            "append": "Append",
        }[score_transformation_method]
    if isinstance(score_transformation_method, list):
        for_sparse, for_dense = tuple(score_transformation_method)
        if for_sparse == "min_max" and for_dense is None:
            return "MM-LEX"
    return str(score_transformation_method)


@dataclass
class RankFusion:
    fuser: Literal["search_engine", "own"] = "search_engine"
    # When `fuser == "own"`, the following options are available.
    fusion_strategy: FusionStrategy = "sum"
    # When `fusion_method != "append"`, the following options are available.
    score_transformation_method: ScoreTransformationMethod = "min_max"
    weighting_strategy: Literal["fixed", "dynamic"] = "fixed"
    ranking_constant: int = 60


def _min_max_scores(response: Response) -> Response:
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


def _rrf_scores(response: Response) -> Response:
    return _rrf_scores_with_k(response)


def _rrf_scores_with_k(response: Response, k: int = 60) -> Response:
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


def _borda_counts(response: Response, n: int) -> Response:
    new_scores = []
    for i in range(len(response.results)):
        new_scores.append(n - i)
    results = [
        Result(product=result.product, score=adjusted_score)
        for result, adjusted_score in zip(response.results, new_scores, strict=True)
    ]
    return Response(results=results, total_hits=response.total_hits)


def _apply_score_transformation(
    sparse_response: Response, dense_response: Response, rank_fusion: RankFusion, size: int
) -> tuple[Response, Response]:
    score_transformation_method = rank_fusion.score_transformation_method
    if score_transformation_method is None:
        return sparse_response, dense_response

    score_transformation_method_dict: dict[str | None, Callable[[Response], Response]] = {
        "min_max": _min_max_scores,
        "rrf": partial(_rrf_scores_with_k, k=rank_fusion.ranking_constant),
        "borda": partial(_borda_counts, n=size),
        None: lambda response: response,
    }

    if isinstance(score_transformation_method, str):
        sparse_response = score_transformation_method_dict[score_transformation_method](sparse_response)
        dense_response = score_transformation_method_dict[score_transformation_method](dense_response)
        return sparse_response, dense_response

    if isinstance(score_transformation_method, list):
        for_sparse, for_dense = tuple(score_transformation_method)
        sparse_response = score_transformation_method_dict[for_sparse](sparse_response)
        dense_response = score_transformation_method_dict[for_dense](dense_response)
        return sparse_response, dense_response

    raise ValueError(f"Invalid score_transformation_method: {score_transformation_method}")


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


def _merge_responses_by_score(
    sparse_response: Response, dense_response: Response, fusion_strategy: FusionStrategy, size: int
) -> Response:
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
        sparse_score, dense_score = sparse_results.get(product_id, 0), dense_results.get(product_id, 0)
        score = max(sparse_score, dense_score) if fusion_strategy == "max" else sparse_score + dense_score
        explanation = {
            "sparse_score": sparse_score,
            "dense_score": dense_score,
        }
        result = Result(product=id_to_product[product_id], score=score, explanation=explanation)
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

    sparse_response, dense_response = _apply_score_transformation(sparse_response, dense_response, rank_fusion, size)

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

    return _merge_responses_by_score(sparse_response, dense_response, rank_fusion.fusion_strategy, size)
