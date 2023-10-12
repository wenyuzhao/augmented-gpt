from __future__ import annotations

import logging

logging.basicConfig(format="[%(levelname)s:%(name)s] %(message)s")

from .decorators import param, function
from .message import MessageStream, Message
from .plugins import Plugin
from . import plugins
from .augmented_gpt import AugmentedGPT, GPTOptions

__all__ = [
    "param",
    "function",
    "Message",
    "MessageStream",
    "Plugin",
    "plugins",
    "AugmentedGPT",
    "GPTOptions",
]
