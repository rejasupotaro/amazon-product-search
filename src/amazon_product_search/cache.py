from functools import wraps
from typing import Any, Callable


def instance_lru_cache(maxsize: int) -> Callable:
    cache: dict[tuple[Any, ...], Any] = {}
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any) -> Any:
            if args in cache:
                return cache[args]
            result = func(*args)
            if maxsize is not None and len(cache) >= maxsize:
                cache.popitem()
            cache[args] = result
            return result
        return wrapper
    return decorator
