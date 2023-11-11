from augmented_gpt import AugmentedGPT, Message, Role, param, function
from augmented_gpt.plugins import *
from typing import Optional


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
    gpt = AugmentedGPT(
        model="gpt-3.5-turbo",
        plugins=[FakeWeatherPlugin(), MemoryPlugin(data_file=file)],
    )
    response = gpt.chat_completion(
        [
            Message(role=Role.USER, content="What is the weather like in boston?"),
        ]
    )
    for msg in response:
        print(msg)
    gpt = AugmentedGPT(model="gpt-3.5-turbo", plugins=[MemoryPlugin(file)])
    response = gpt.chat_completion(
        [
            Message(
                role=Role.USER,
                content="What was your response when I asked you about the weather in Boston?",
            ),
        ]
    )
    all_assistant_content = ""
    for msg in response:
        if msg.role == Role.ASSISTANT:
            all_assistant_content += msg.content or ""
        print(msg)
    assert "72" in all_assistant_content
