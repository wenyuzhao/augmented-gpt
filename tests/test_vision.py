from augmented_gpt import AugmentedGPT, Message, Role
from augmented_gpt.message import ContentPartImage, ContentPartText
from augmented_gpt.plugins import *
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_vision():
    gpt = AugmentedGPT(model="gpt-4-vision-preview")
    response = gpt.chat_completion(
        [
            Message(
                role=Role.USER,
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
        if msg.role == Role.ASSISTANT:
            assert msg.content is None or isinstance(msg.content, str)
            all_assistant_content += msg.content or ""
        print(msg)
    assert "cat" in all_assistant_content.lower()
