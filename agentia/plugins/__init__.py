from ..decorators import tool
from ..message import Message
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from ..agent import Agent


class Plugin:
    NAME: str | None

    def __init__(self, config: Any = None):
        if hasattr(self, "NAME") and self.NAME:
            self.name = self.NAME.strip()
        else:
            self.name = self.__class__.__name__
            if self.name.endswith("Plugin"):
                self.name = self.name[:-6]
        self.cache_key = f"plugins.{self.name}".lower()
        self.config = config
        self.agent: "Agent"

    def _register(self, agent: "Agent"):
        self.agent = agent
        self.log = self.agent.log.getChild(self.name)

    async def init(self):
        pass

    def on_new_chat_message(self, msg: Message) -> Any: ...


from .calc import *
from .clock import *
from .code import *
from .memory import *
from .mstodo import *
from .search import *
from .dalle import *
from .vision import *
from .web import *


ALL_PLUGINS: dict[str, Type[Plugin]] = {
    "clock": ClockPlugin,
    "calc": CalculatorPlugin,
    "code": CodePlugin,
    "mstodo": MSToDoPlugin,
    "memory": MemoryPlugin,
    "search": SearchPlugin,
    "dalle": DallEPlugin,
    "vision": VisionPlugin,
    "web": WebPlugin,
}


def register_plugin(name: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
    def wrapper(cls: Type[Plugin]):
        ALL_PLUGINS[name] = cls
        return cls

    return wrapper
