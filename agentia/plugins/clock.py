import datetime
from ..decorators import *
from . import Plugin


class ClockPlugin(Plugin):
    @tool
    def get_current_time(self):
        """Get the current time in ISO format"""
        return datetime.datetime.now().isoformat()
