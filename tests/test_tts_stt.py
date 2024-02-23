from augmented_gpt import utils
import tempfile
import pytest
import dotenv
import os

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_function_call():
    fp = tempfile.NamedTemporaryFile(suffix=".mp3")
    fp.close()
    tts = utils.tts.TextToSpeech(os.environ["OPENAI_API_KEY"])
    await tts.speak("Hello World!", fp.name)
    stt = utils.stt.SpeechToText(os.environ["OPENAI_API_KEY"])
    text = await stt.transcribe(fp.name)
    assert "world" in text.lower()
