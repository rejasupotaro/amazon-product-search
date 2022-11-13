from typing import Generic, TypeVar

T = TypeVar("T")


class WeakReference(Generic[T]):
    def __init__(self, ref: T):
        self.ref = ref
