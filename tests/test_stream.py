from augmented_gpt import AugmentedGPT, Message, Role, param, tool
from augmented_gpt.plugins import *
from typing import Optional
import pytest
import dotenv

dotenv.load_dotenv()


@tool
def get_current_weather(
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
    gpt = AugmentedGPT(model="gpt-3.5-turbo", tools=[get_current_weather])
    response = gpt.chat_completion(
        [
            Message(role=Role.USER, content="What is the weather like in boston?"),
        ],
        stream=True,
    )
    all_assistant_content = ""
    async for stream in response:
        content = ""
        async for delta in stream:
            assert delta is None or isinstance(delta, str)
            content += delta
            print(" - ", delta)
        msg = await stream.wait_for_completion()
        if msg.role == Role.ASSISTANT:
            all_assistant_content += content
        print(msg)
    assert "72" in all_assistant_content
