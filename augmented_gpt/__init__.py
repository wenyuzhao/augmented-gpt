from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(format="[%(levelname)s:%(name)s] %(message)s")

from .decorators import param, function
from .message import (
    MessageStream,
    Message,
    Role,
    ToolCall,
    FunctionCall,
    ContentPart,
    ContentPartImage,
    ContentPartText,
)
from .plugins import Plugin
from . import plugins
from .augmented_gpt import AugmentedGPT, GPTOptions, ChatCompletion, ServerError
from . import utils

__all__ = [
    "utils",
    "param",
    "function",
    "Message",
    "MessageStream",
    "Role",
    "Plugin",
    "plugins",
    "AugmentedGPT",
    "ChatCompletion",
    "GPTOptions",
    "ServerError",
    "ToolCall",
    "FunctionCall",
    "ContentPart",
    "ContentPartImage",
    "ContentPartText",
]
