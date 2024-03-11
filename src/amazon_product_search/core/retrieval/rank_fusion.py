from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal

from amazon_product_search.core.retrieval.response import Response, Result
from amazon_product_search.core.retrieval.score_normalizer import min_max_scale
from amazon_product_search.core.retrieval.weighting_strategy import FixedWeighting

_ScoreTransformationMethod = Literal["min_max", "rrf", "borda"]

ScoreTransformationMethod = _ScoreTransformationMethod | list[_ScoreTransformationMethod] | None  # type: ignore

CombinationMethod = Literal["sum", "max", "append"]


@dataclass
class RankFusion:
    combination_method: CombinationMethod = "sum"
    # When `combination_method != "append"`, the following options are available.
    score_transformation_method: ScoreTransformationMethod = "min_max"
    weighting_strategy: Literal["fixed"] = "fixed"
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


def _borda_counts(response: Response, n: int) -> Response:
    new_scores = []
    for i in range(len(response.results)):
        new_scores.append(n - i)
    results = [
        Result(product=result.product, score=adjusted_score)
        for result, adjusted_score in zip(response.results, new_scores, strict=True)
    ]
    return Response(results=results, total_hits=response.total_hits)


def _transform_scores(
    lexical_response: Response, semantic_response: Response, rank_fusion: RankFusion, size: int
) -> tuple[Response, Response]:
    score_transformation_method = rank_fusion.score_transformation_method
    if score_transformation_method is None:
        return lexical_response, semantic_response

    score_transformation_method_dict: dict[str | None, Callable[[Response], Response]] = {
        "min_max": _min_max_scores,
        "rrf": partial(_rrf_scores, k=rank_fusion.ranking_constant),
        "borda": partial(_borda_counts, n=size),
        None: lambda response: response,
    }

    if isinstance(score_transformation_method, str):
        lexical_response = score_transformation_method_dict[score_transformation_method](lexical_response)
        semantic_response = score_transformation_method_dict[score_transformation_method](semantic_response)
        return lexical_response, semantic_response

    if isinstance(score_transformation_method, list):
        for_lexical, for_semantic = tuple(score_transformation_method)
        lexical_response = score_transformation_method_dict[for_lexical](lexical_response)
        semantic_response = score_transformation_method_dict[for_semantic](semantic_response)
        return lexical_response, semantic_response

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


def _combine_responses(
    lexical_response: Response, semantic_response: Response, combination_method: CombinationMethod, size: int
) -> Response:
    """Merge two responses by score.

    Args:
        lexical_response (Response): A response from lexical search.
        semantic_response (Response): A response from semantic search.
        combination_method (CombinationMethod): The method to combine results.
        size (int): The number of results to return.

    Returns:
        Response: A merged response.
    """
    id_to_product: dict[str, dict[str, Any]] = {}
    lexical_results: dict[str, float] = {}
    semantic_results: dict[str, float] = {}

    for result in lexical_response.results:
        product_id = result.product["product_id"]
        id_to_product[product_id] = result.product
        lexical_results[product_id] = result.score
    for result in semantic_response.results:
        product_id = result.product["product_id"]
        id_to_product[product_id] = result.product
        semantic_results[product_id] = result.score

    results: list[Result] = []
    for product_id in lexical_results.keys() | semantic_results.keys():
        lexical_score, semantic_score = lexical_results.get(product_id, 0), semantic_results.get(product_id, 0)
        score = max(lexical_score, semantic_score) if combination_method == "max" else lexical_score + semantic_score
        explanation = {
            "lexical_score": lexical_score,
            "semantic_score": semantic_score,
        }
        result = Result(product=id_to_product[product_id], score=score, explanation=explanation)
        results.append(result)
    total_hits = max(lexical_response.total_hits, semantic_response.total_hits, len(results))

    results = sorted(results, key=lambda result: (result.score, result.product["product_id"]), reverse=True)[:size]
    return Response(results=results, total_hits=total_hits)


def fuse(
    query: str,
    lexical_response: Response,
    semantic_response: Response,
    lexical_boost: float,
    semantic_boost: float,
    rank_fusion: RankFusion,
    size: int,
) -> Response:
    if rank_fusion.combination_method == "append":
        return _append_results(lexical_response, semantic_response, size)

    lexical_response, semantic_response = _transform_scores(lexical_response, semantic_response, rank_fusion, size)

    if rank_fusion.weighting_strategy:
        weighting_strategy = FixedWeighting({"lexical": lexical_boost, "semantic": semantic_boost})
        for result in lexical_response.results:
            result.score *= weighting_strategy.apply("lexical", query)
        for result in semantic_response.results:
            result.score *= weighting_strategy.apply("semantic", query)

    return _combine_responses(lexical_response, semantic_response, rank_fusion.combination_method, size)
