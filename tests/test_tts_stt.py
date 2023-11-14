from augmented_gpt import AugmentedGPT
from augmented_gpt import utils
import tempfile


def test_function_call():
    gpt = AugmentedGPT()
    fp = tempfile.NamedTemporaryFile(suffix=".mp3")
    fp.close()
    tts = utils.tts.TextToSpeech(gpt.api_key)
    tts.speak_sync("Hello World!", fp.name)
    stt = utils.stt.SpeechToText(gpt.api_key)
    text = stt.transcribe_sync(fp.name)
    assert "world" in text.lower()
