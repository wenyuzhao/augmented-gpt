from augmented_gpt import AugmentedGPT, Message, param, function
from augmented_gpt.plugins import *
from typing import Optional


@function
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


def test_function_call():
    gpt = AugmentedGPT(functions=[get_current_weather])
    response = gpt.chat_completion(
        [
            Message(role="user", content="What is the weather like in boston?"),
        ],
        stream=True,
    )
    all_assistant_content = ""
    for msg in response:
        content = ""
        for delta in msg:
            content += delta
            print(" - ", delta)
        if msg.message().role == "assistant":
            all_assistant_content += content
        print(msg.message())
    assert "72" in all_assistant_content
