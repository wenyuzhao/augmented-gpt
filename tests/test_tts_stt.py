from augmented_gpt import AugmentedGPT
from augmented_gpt import utils
import tempfile
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_function_call():
    gpt = AugmentedGPT()
    fp = tempfile.NamedTemporaryFile(suffix=".mp3")
    fp.close()
    tts = utils.tts.TextToSpeech(gpt.api_key)
    await tts.speak("Hello World!", fp.name)
    stt = utils.stt.SpeechToText(gpt.api_key)
    text = await stt.transcribe(fp.name)
    assert "world" in text.lower()
