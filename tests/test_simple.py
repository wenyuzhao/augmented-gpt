from augmented_gpt import AugmentedGPT, Message, param, function
from augmented_gpt.plugins import *
from typing import *
import json


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
        ]
    )
    for msg in response:
        print(msg)


class FakeWeatherPlugin(Plugin):
    @function
    def get_current_weather(
        self,
        location: str = param("The city and state, e.g. San Francisco, CA"),
        unit: Optional[str] = param(
            enum=["celsius", "fahrenheit"], default="fahrenheit"
        ),
    ):
        """Get the current weather in a given location"""
        return {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }


def test_weather_and_memory_plugin():
    file = "_test-data.csv"
    with open(file, "w+") as f:
        f.write("")
    gpt = AugmentedGPT(plugins=[FakeWeatherPlugin(), MemoryPlugin(file)])
    response = gpt.chat_completion(
        [
            Message(role="user", content="What is the weather like in boston?"),
        ]
    )
    for msg in response:
        print(msg)
    gpt = AugmentedGPT(plugins=[MemoryPlugin(file)])
    response = gpt.chat_completion(
        [
            Message(
                role="user", content="When did I ask you for the weather in boston?"
            ),
        ]
    )
    for msg in response:
        print(msg)
