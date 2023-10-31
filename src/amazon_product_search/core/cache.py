import weakref
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Any, Callable, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        return self.cache.get(key)

    def set(self, key: K, value: V) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value


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
