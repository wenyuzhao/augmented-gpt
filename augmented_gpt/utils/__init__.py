import asyncio
from typing import TypeVar, Awaitable


T = TypeVar("T")


def block_on(co: Awaitable[T]) -> T:
    loop = asyncio.get_running_loop()
    return loop.run_until_complete(co)


from . import tts, stt

__all__ = ["block_on", "tts", "stt"]
