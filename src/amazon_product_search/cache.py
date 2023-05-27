import weakref
from functools import lru_cache, wraps
from typing import Any, Callable


def weak_lru_cache(maxsize: int = 128, typed: bool = False) -> Callable:
    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs) -> Any:
            return func(_self(), *args, **kwargs)

        @wraps(func)
        def inner(self, *args, **kwargs) -> Any:
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper
