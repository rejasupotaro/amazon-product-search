from amazon_product_search.cache import weak_lru_cache


class Counter:
    def __init__(self):
        self.n = 0

    @weak_lru_cache(maxsize=1)
    def f(self, i: int):
        """n will be incremented only when i is not in cache."""
        self.n += 1


def test_instance_cache():
    counter = Counter()
    for i, expected in [
        (1, 1),  # 1 is not cached, counter should be incremented.
        (1, 1),  # 1 is cached, counter should remain the same.
        (2, 2),  # 2 is not cached, counter should be incremented.
        (2, 2),  # 2 is cached, counter should remain the same.
        (1, 3),  # 1 is not cached (evicted), counter should be incremented.
    ]:
        counter.f(i)
        assert counter.n == expected
