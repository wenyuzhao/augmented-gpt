from agentia import AugmentedGPT, Message, Role, param, tool
from agentia.plugins import Plugin
from typing import Optional
import pytest
import dotenv

dotenv.load_dotenv()


class FakeWeatherPlugin(Plugin):
    @tool
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


@pytest.mark.asyncio
async def test_weather_and_memory_plugin():
    file = "_test-data.csv"
    with open(file, "w+") as f:
        f.write("")
    gpt = AugmentedGPT(model="openai/gpt-4o-mini", tools=[FakeWeatherPlugin()])
    response = gpt.chat_completion(
        [Message(role=Role.USER, content="What is the weather like in boston?")]
    )
    all_assistant_content: str = ""
    async for msg in response.messages():
        if msg.role == Role.ASSISTANT:
            assert msg.content is None or isinstance(msg.content, str)
            all_assistant_content += msg.content or ""
        print(msg)
    assert "72" in all_assistant_content
