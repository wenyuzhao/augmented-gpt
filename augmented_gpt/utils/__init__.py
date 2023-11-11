import asyncio
from typing import Any, TypeVar, Generator


T = TypeVar("T")


def block_on(co: Generator[Any, None, T]) -> T:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(co)


from . import tts, stt

__all__ = ["block_on", "tts", "stt"]
