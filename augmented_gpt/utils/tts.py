from typing import Literal, Optional

import openai
from pathlib import Path

Voices = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class TextToSpeech:
    def __init__(
        self,
        api_key: str,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        voice: Voices = "alloy",
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model

    def speak(
        self,
        text: str,
        output: Optional[str | Path] = None,
        voice: Optional[Voices] = None,
    ):
        voice = voice or self.voice
        response = self.client.audio.speech.create(
            model=self.model, voice=voice, input=text
        )
        response.stream_to_file(output)
