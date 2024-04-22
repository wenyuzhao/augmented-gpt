from typing import Literal, Optional

import openai
from pathlib import Path
import os

Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class TextToSpeech:
    def __init__(
        self,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        voice: Voice = "alloy",
        api_key: str | None = None,
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.voice: Voice = voice
        self.model = model

    async def speak(
        self,
        text: str,
        output: str | Path,
        voice: Optional[Voice] = None,
    ):
        _voice: Voice = voice or self.voice
        response = self.client.audio.speech.with_streaming_response.create(
            model=self.model, voice=_voice, input=text
        )
        async with response as r:
            await r.stream_to_file(output)


__all__ = ["TextToSpeech", "Voice"]
