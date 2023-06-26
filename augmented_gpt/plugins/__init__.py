import inspect
from ..decorators import *
from ..message import Message
import datetime
from logging import Logger


class Plugin:
    logger: Logger

    def register(self, gpt: "AugmentedGPT"):
        for n, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not hasattr(method, "gpt_function_call_info"):
                continue
            func_info = getattr(method, "gpt_function_call_info")
            clsname = self.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            if not func_info["name"].startswith(clsname + "-"):
                func_info["name"] = clsname + "-" + func_info["name"]
            gpt.add_function(method)
        self.logger = gpt.logger

    def _log_call(self, name: str, *args, **kwargs):
        msg = f"➡️ {name}: " + ", ".join(args) + ", ".join((f"{k}={v}" for k, v in kwargs.items()))
        self.logger.debug(msg)

    def on_new_chat_message(self, msg: Message):
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
