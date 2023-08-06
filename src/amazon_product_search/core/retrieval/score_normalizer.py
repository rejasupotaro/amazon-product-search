import numpy as np


def min_max_scale(x: np.ndarray, min_val: float | None, max_val: float | None) -> np.ndarray:
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)
