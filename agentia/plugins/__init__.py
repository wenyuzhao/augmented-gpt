from ..decorators import *
from ..message import Message
import datetime
from typing import TYPE_CHECKING
from agentia import tool

if TYPE_CHECKING:
    from ..augmented_gpt import AugmentedGPT


class Plugin:
    def __init__(self, name: Optional[str] = None, config: Any = None):
        self.name = name or self.__class__.__name__
        self.config = config
        self.client: "AugmentedGPT"

    def register(self, client: "AugmentedGPT"):
        self.client = client

    def on_new_chat_message(self, msg: Message) -> Any: ...


class TimestampPlugin(Plugin):
    @tool
    def get_current_timestamp(self):
        """Get the current tempstamp in ISO format"""
        return datetime.datetime.now().isoformat()


from .calc import *


def all_plugins():
    return [TimestampPlugin(), CalculatorPlugin()]
