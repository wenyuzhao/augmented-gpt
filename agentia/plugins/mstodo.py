from ..decorators import *
from . import Plugin
from typing import Annotated, override
import pymstodo


class MSToDoPlugin(Plugin):
    def __test_token(self, client_id: str, client_secret: str, token: Any):
        try:
            client = pymstodo.ToDoConnection(
                client_id=client_id, client_secret=client_secret, token=token
            )
            _lists = client.get_lists()
            return True
        except Exception as e:
            return False

    @override
    async def init(self):
        self.agent.log.info("MSToDoPlugin initialized")
        client_id = self.config.get("client_id")
        client_secret = self.config.get("client_secret")

        with self.agent.open_cache() as cache:
            token = None
            key = self.cache_key + ".token"
            if key in cache:
                if self.__test_token(client_id, client_secret, cache[key]):
                    token = cache[key]
            if token is None:
                auth_url = pymstodo.ToDoConnection.get_auth_url(client_id)
                redirect_resp = input(
                    f"Go here and authorize:\n{auth_url}\n\nPaste the full redirect URL below:\n"
                )
                token = pymstodo.ToDoConnection.get_token(
                    client_id, client_secret, redirect_resp
                )
                cache[key] = token
        self.client = pymstodo.ToDoConnection(
            client_id=client_id, client_secret=client_secret, token=token
        )

    @tool
    def list_tasks(self):
        """List all the to-do tasks"""
        lists = self.client.get_lists()
        task_list = lists[0]
        tasks = self.client.get_tasks(task_list.list_id)
        return [str(x) for x in tasks]
