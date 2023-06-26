import logging

logging.basicConfig(format="[%(levelname)s:%(name)s] %(message)s")

from .decorators import *
from .message import *
from .plugins import Plugin
from . import plugins
from .augmented_gpt import AugmentedGPT

__all__ = [
    "param",
    "function",
    "Message",
    "MessageStream",
    "Plugin",
    "plugins",
    "AugmentedGPT",
]
