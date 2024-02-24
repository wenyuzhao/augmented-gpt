from augmented_gpt import utils
import tempfile
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_function_call():
    fp = tempfile.NamedTemporaryFile(suffix=".mp3")
    fp.close()
    tts = utils.tts.TextToSpeech()
    await tts.speak("Hello World!", fp.name)
    stt = utils.stt.SpeechToText()
    text = await stt.transcribe(fp.name)
    assert "world" in text.lower()
