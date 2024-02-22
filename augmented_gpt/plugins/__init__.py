import openai

from ..decorators import *
from ..message import Message
import datetime


class Plugin:
    client: openai.AsyncOpenAI

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def register(self, client: openai.AsyncOpenAI):
        self.client = client

    def on_new_chat_message(self, msg: Message) -> Any:
        ...


class TimestampPlugin(Plugin):
    @function
    def get_current_timestamp(self):
        """Get the current tempstamp in ISO format"""
        return datetime.datetime.now().isoformat()


from .memory import *
from .calc import *


def all_plugins():
    return [TimestampPlugin(), MemoryPlugin(), CalculatorPlugin()]
