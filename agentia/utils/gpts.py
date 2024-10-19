from pathlib import Path
from agentia import LOGGER
from agentia.tools import ToolRegistry
from aiohttp import web
from urllib.parse import urlparse
import uuid, jinja2
import os.path
import json


class GPTsActionServer:
    def __init__(
        self,
        tools: ToolRegistry,
        url: str,
        port: int,
        access_code: str | None,
        token_storage: Path | str | None,
        app: web.Application | None = None,
    ):
        while url.endswith("/"):
            url = url[:-1]
        o = urlparse(url)
        self.host = o.scheme + "://" + o.netloc
        self.base_url = o.path
        self.port = port
        self.tools = tools
        self.access_code = access_code
        self.token_storage = token_storage
        self.valid_tokens: set[str] = set()
        if token_storage is not None and os.path.exists(token_storage):
            with open(token_storage, "r") as f:
                self.valid_tokens = set(json.load(f))
        self.app = app or web.Application()
        self.__register_routes()

    async def actions_schema(self, request: web.Request):
        return web.json_response(self.tools.to_gpts_json(self.host + self.base_url))

    async def handle_action(self, request: web.Request):
        err = await self.verify(request)
        if err is not None:
            return err
        tool_name = request.match_info.get("name", None)
        LOGGER.debug(f"GPTs Action: {tool_name}")
        if tool_name is None:
            LOGGER.error(f"GPTs Action not found: {tool_name}")
            return web.json_response({"error": "Invalid tool name"}, status=400)
        args = {k: v for k, v in request.rel_url.query.items()}
        res = await self.tools.call_function_raw(name=tool_name, args=args, tool_id="")
        return web.json_response(res)

    async def handle_auth(self, request: web.Request):
        redirect_uri: str = request.query["redirect_uri"]
        state: str = request.query["state"]
        LOGGER.debug(f"OAuth: Auth redirect_uri={redirect_uri} state={state}")
        verify_access_code_url = f"{self.base_url}/oauth/verify_access_code?redirect_uri={redirect_uri}&state={state}"
        environment = jinja2.Environment()
        with open(os.path.dirname(__file__) + "/gpts-auth.html") as f:
            template = environment.from_string(f.read())
        text = template.render(verify_access_code_url=verify_access_code_url)
        return web.Response(text=text, content_type="text/html")

    async def handle_verify_access_code(self, request: web.Request):
        redirect_uri: str = request.query["redirect_uri"]
        form = await request.post()
        access_code = form["access_code"]
        state = request.query["state"]
        LOGGER.debug(
            f"OAuth: Verify-Access-Code redirect_uri={redirect_uri} state={state} access_code={access_code}"
        )
        if access_code != self.access_code:
            LOGGER.debug(f"OAuth: Invalid access code {self.access_code}")
            return web.HTTPFound(
                f"{self.base_url}/oauth/auth?invalid=1&redirect_uri={redirect_uri}&state={state}"
            )
        LOGGER.debug(f"OAuth: Access code verified")
        assert isinstance(access_code, str)
        url: str = redirect_uri + "?state=" + state
        return web.HTTPFound(url)

    async def verify(self, req: web.Request) -> web.Response | None:
        if self.access_code is None:
            return None
        auth = req.headers.get("Authorization", None)
        if auth is None:
            return web.json_response({"error": "No Authorization header"}, status=401)
        if not auth.startswith("Bearer "):
            return web.json_response(
                {"error": "Invalid Authorization header"}, status=401
            )
        token = auth.split(" ")[1]
        if token not in self.valid_tokens:
            LOGGER.debug(f"OAuth: Invalid token {token}")
            return web.json_response({"error": "Invalid token"}, status=401)
        return None

    async def handle_token(self, request: web.Request):
        # generate a token
        token = uuid.uuid4().hex
        LOGGER.debug(f"OAuth: Generated token {token}")
        # save the token
        if self.token_storage is not None:
            if not os.path.exists(self.token_storage):
                with open(self.token_storage, "w") as f:
                    json.dump([token], f)
            else:
                with open(self.token_storage, "r") as f:
                    tokens = json.load(f)
                tokens.append(token)
                with open(self.token_storage, "w") as f:
                    json.dump(tokens, f)
        self.valid_tokens.add(token)
        return web.json_response({"access_token": token})

    async def handle(self, request: web.Request):
        return web.json_response({"error": "oops"})

    def __register_routes(self):
        self.app.add_routes(
            [
                web.get("", self.actions_schema),
                web.get("/", self.actions_schema),
                web.get("/actions/{name}", self.handle_action),
                web.get("/oauth/auth", self.handle_auth),
                web.post("/oauth/token", self.handle_token),
                web.post("/oauth/verify_access_code", self.handle_verify_access_code),
            ]
        )

    async def run(self) -> None:
        LOGGER.info(
            f"Starting GPTs actions server on {self.host}:{self.port}{self.base_url}"
        )
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 3000)
        await site.start()


async def start_gpts_action_server(
    tools: ToolRegistry,
    url: str,
    port: int = 6001,
    access_code: str | None = None,
    token_storage: Path | str | None = None,
):
    server = GPTsActionServer(
        tools,
        url=url,
        port=port,
        access_code=access_code,
        token_storage=token_storage,
    )
    await server.run()
