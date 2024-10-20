from agentia import Agent, UserMessage, tool, utils
from agentia.message import AssistantMessage, ContentPartImage, ContentPartText
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_vision():
    gpt = Agent(model="gpt-4-vision-preview")
    response = gpt.chat_completion(
        [
            UserMessage(
                content=[
                    ContentPartText("What is this animal?"),
                    ContentPartImage(
                        "https://icons.iconarchive.com/icons/iconarchive/cute-animal/256/Cute-Cat-icon.png"
                    ),
                ],
            ),
        ]
    )
    all_assistant_content = ""
    async for msg in response:
        assert msg.content is None or isinstance(msg.content, str)
        all_assistant_content += msg.content or ""
        print(msg)
    assert "cat" in all_assistant_content.lower()


@pytest.mark.asyncio
async def test_utils_image_vision():
    response = await utils.image.vision(
        "https://icons.iconarchive.com/icons/iconarchive/cute-animal/256/Cute-Cat-icon.png",
        "What is this animal?",
    )
    assert "cat" in response.lower()
