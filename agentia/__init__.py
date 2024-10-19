from __future__ import annotations

import logging

LOGGER = logging.getLogger("augmented-gpt")
MSG_LOGGER = LOGGER.getChild("message")

from .decorators import tool
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
from .agent import (
    Agent,
    ChatCompletion,
    ServerError,
    ChatCompletionEvent,
    UserConsentEvent,
    ToolCallEvent,
)
from .llm import ModelOptions
from . import utils
from .tools import ToolInfo, ToolRegistry, Tools


__all__ = [
    "utils",
    "tool",
    "Message",
    "MessageStream",
    "Role",
    "Plugin",
    "plugins",
    "Agent",
    "ChatCompletion",
    "ChatCompletionEvent",
    "UserConsentEvent",
    "ToolCallEvent",
    "ModelOptions",
    "ServerError",
    "ToolCall",
    "FunctionCall",
    "ContentPart",
    "ContentPartImage",
    "ContentPartText",
    "ToolInfo",
    "ToolRegistry",
    "Tools",
]
