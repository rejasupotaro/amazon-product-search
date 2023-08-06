def min_max_scale(x: list[float], min_val: float | None = None, max_val: float | None = None) -> list[float]:
    if not x:
        return []

    if min_val is None:
        min_val = min(x)
    if max_val is None:
        max_val = max(x)

    if min_val == max_val:
        return [0.5] * len(x)

    return [(val - min_val) / (max_val - min_val) for val in x]
