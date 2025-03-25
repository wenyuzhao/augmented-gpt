import os
from agentia.history import History
from . import ModelOptions
from ..tools import ToolRegistry
from .openai import OpenAIBackend


class OpenRouterBackend(OpenAIBackend):
    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(model, tools, options, history, api_key, base_url)
        if providers := os.environ.get("OPENROUTER_PROVIDERS"):
            self.extra_body["provider"] = {
                "order": [x.strip() for x in providers.strip().split(",")]
            }
        self.has_reasoning = True
        self.extra_body["transforms"] = ["middle-out"]
