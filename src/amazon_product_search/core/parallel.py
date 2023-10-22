from asyncio import Semaphore
from typing import Coroutine


def limit_concurrency(
    coroutines: list[Coroutine],
    max_concurrency: int,
) -> list[Coroutine]:
    semaphore = Semaphore(max_concurrency)

    async def with_concurrency_limit(coroutine):
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(c) for c in coroutines]
