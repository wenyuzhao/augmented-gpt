from __future__ import annotations

import logging

logging.basicConfig(format="[%(levelname)s:%(name)s] %(message)s")

from .decorators import param, function
from .message import MessageStream, Message, Role
from .plugins import Plugin
from . import plugins
from .augmented_gpt import AugmentedGPT, GPTOptions, ChatCompletion

__all__ = [
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
]
