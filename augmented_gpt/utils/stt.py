from typing import Literal

import openai
from pathlib import Path
import os


class SpeechToText:
    def __init__(
        self, model: Literal["whisper-1"] = "whisper-1", api_key: str | None = None
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.model = model

    async def transcribe(self, path: str | Path) -> str:
        with open(path, "rb") as audio_file:
            transcript = await self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            return transcript.text


__all__ = ["SpeechToText"]
