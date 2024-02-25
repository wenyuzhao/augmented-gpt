from augmented_gpt.augmented_gpt import ToolRegistry
from aiohttp import web
import asyncio
from urllib.parse import urlparse


class __GPTsActionServer:
    def __init__(
        self, tools: ToolRegistry, host: str, port: int = 6001, base_url: str = "/"
    ):
        self.host = host
        self.base_url = base_url
        self.port = port
        self.tools = tools

    async def actions_schema(self, request: web.Request):
        return web.json_response(self.tools.to_gpts_json(self.host + self.base_url))

    async def handle_action(self, request: web.Request):
        print("handle_action", request.query, request.match_info)
        tool_name = request.match_info.get("name", None)
        if tool_name is None:
            return web.json_response({"error": "Invalid tool name"}, status=400)
        args = {k: v for k, v in request.rel_url.query.items()}
        res = await self.tools.call_function_raw(name=tool_name, args=args, tool_id="")
        return web.json_response(res)

    async def handle(self, request: web.Request):
        print("Handling request", request.query, request.match_info)
        return web.json_response({"error": "oops"})

    async def run(self) -> None:
        print(f"Starting GPTs frontend on port {self.port}")
        app = web.Application()
        app.router.add_get(f"{self.base_url}", self.actions_schema)
        app.router.add_get(self.base_url + "/actions/{name}", self.handle_action)
        handler = app.make_handler()
        loop = asyncio.get_event_loop()
        await loop.create_server(handler, "0.0.0.0", self.port)


async def start_gpts_action_server(tools: ToolRegistry, url: str, port: int):
    while url.endswith("/"):
        url = url[:-1]
    o = urlparse(url)
    host = o.scheme + "://" + o.netloc
    base_url = o.path
    server = __GPTsActionServer(tools, host=host, base_url=base_url, port=port)
    await server.run()
