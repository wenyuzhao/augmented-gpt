from augmented_gpt import AugmentedGPT, Message, param, function
from augmented_gpt.plugins import *
from typing import Optional
import pytest


@function
async def get_current_weather(
    location: str = param("The city and state, e.g. San Francisco, CA"),
    unit: Optional[str] = param(enum=["celsius", "fahrenheit"], default="fahrenheit"),
):
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


@pytest.mark.asyncio
async def test_function_call():
    gpt = AugmentedGPT(functions=[get_current_weather])
    response = gpt.chat_completion(
        [
            Message(role="user", content="What is the weather like in boston?"),
        ]
    )
    async for msg in response:
        print(msg)
