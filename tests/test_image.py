from augmented_gpt import utils
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_image_generation():
    url = await utils.image.generate("A pet cat", size="256x256")
    print(url)
    response = await utils.image.vision(url, "What is this animal?")
    assert "cat" in response.lower()
