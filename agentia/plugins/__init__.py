from ..decorators import tool
from ..message import Message
import datetime
from typing import TYPE_CHECKING, Any
from agentia import tool

if TYPE_CHECKING:
    from ..agent import Agent


class Plugin:
    def __init__(self, name: str | None = None, config: Any = None):
        self.name = name or self.__class__.__name__
        self.config = config
        self.agent: "Agent"

    def register(self, agent: "Agent"):
        self.agent = agent

    def on_new_chat_message(self, msg: Message) -> Any: ...


class ClockPlugin(Plugin):
    @tool
    def get_current_time(self):
        """Get the current time in ISO format"""
        return datetime.datetime.now().isoformat()


from .calc import *
from .code import *


ALL_PLUGINS = {
    "clock": ClockPlugin,
    "calc": CalculatorPlugin,
    "code": CodePlugin,
}
