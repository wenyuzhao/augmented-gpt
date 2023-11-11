
from typing import Literal

import openai
from pathlib import Path

Voices = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

class SpeechToText:
    def __init__(self, api_key: str, model:Literal["whisper-1"]= "whisper-1"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, path: str | Path) -> str:
        with open(path, "rb") as audio_file:
            transcript = self.client.audio.translations.create(model="whisper-1", file=audio_file, response_format="text")
            return transcript