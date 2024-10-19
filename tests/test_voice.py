from agentia import utils
import tempfile
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_function_call():
    fp = tempfile.NamedTemporaryFile(suffix=".mp3")
    fp.close()
    await utils.voice.tts("Hello World!", fp.name)
    text = await utils.voice.stt(fp.name)
    assert "world" in text.lower()
