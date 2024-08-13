from augmented_gpt import AugmentedGPT, Message, Role, param, tool
from typing import Optional
import pytest
import asyncio
import dotenv

dotenv.load_dotenv()


@tool
async def get_current_weather(
    location: str = param("The city and state, e.g. San Francisco, CA"),
    unit: Optional[str] = param(enum=["celsius", "fahrenheit"], default="fahrenheit"),
):
    """Get the current weather in a given location"""
    await asyncio.sleep(0.1)
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


@pytest.mark.asyncio
async def test_function_call():
    gpt = AugmentedGPT(model="openai/gpt-4o-mini", tools=[get_current_weather])
    response = gpt.chat_completion(
        [Message(role=Role.USER, content="What is the weather like in boston?")]
    )
    all_assistant_content = ""
    async for msg in response.messages():
        if msg.role == Role.ASSISTANT:
            assert msg.content is None or isinstance(msg.content, str)
            all_assistant_content += msg.content or ""
        print(msg)
    assert "72" in all_assistant_content
