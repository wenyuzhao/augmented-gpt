from augmented_gpt.augmented_gpt import ToolRegistry
from aiohttp import web
import asyncio
from urllib.parse import urlparse
import uuid, jinja2
import os.path


class __GPTsActionServer:
    def __init__(
        self,
        tools: ToolRegistry,
        host: str,
        port: int,
        base_url: str,
        access_code: str | None,
    ):
        self.host = host
        self.base_url = base_url
        self.port = port
        self.tools = tools
        self.access_code = access_code
        self.valid_tokens: set[str] = set()

    async def actions_schema(self, request: web.Request):
        return web.json_response(self.tools.to_gpts_json(self.host + self.base_url))

    async def handle_action(self, request: web.Request):
        err = await self.verify(request)
        if err is not None:
            return err
        print("handle_action", request.query, request.match_info)
        tool_name = request.match_info.get("name", None)
        if tool_name is None:
            return web.json_response({"error": "Invalid tool name"}, status=400)
        args = {k: v for k, v in request.rel_url.query.items()}
        res = await self.tools.call_function_raw(name=tool_name, args=args, tool_id="")
        return web.json_response(res)

    async def handle_auth(self, request: web.Request):
        redirect_uri: str = request.query["redirect_uri"]
        state: str = request.query["state"]
        print(f"[OAuth] Auth redirect_uri={redirect_uri} state={state}")
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
        print(
            f"[OAuth] Verify-Access-Code redirect_uri={redirect_uri} state={state} access_code={access_code}"
        )
        if access_code != self.access_code:
            print("Invalid access code", self.access_code)
            return web.HTTPFound(
                f"{self.base_url}/oauth/auth?invalid=1&redirect_uri={redirect_uri}&state={state}"
            )
        print("Valid access code")
        assert isinstance(access_code, str)
        url: str = redirect_uri + "?state=" + state
        return web.HTTPFound(url)

    async def verify(self, req: web.Request) -> web.Response | None:
        if self.access_code is None:
            return None
        auth = req.headers.get("Authorization", None)
        if auth is None:
            return web.json_response({"error": "No Authorization header"}, status=401)
        print("auth", auth)
        if not auth.startswith("Bearer "):
            return web.json_response(
                {"error": "Invalid Authorization header"}, status=401
            )
        token = auth.split(" ")[1]
        if token not in self.valid_tokens:
            return web.json_response({"error": "Invalid token"}, status=401)
        return None

    async def handle_token(self, request: web.Request):
        print("handle_token", request.query, request.match_info)
        form = await request.post()
        print(f"[OAuth] Token", form)
        # generate a token
        token = uuid.uuid4().hex
        print("Generated token", token)
        # save the token
        self.valid_tokens.add(token)
        return web.json_response({"access_token": token})

    async def handle(self, request: web.Request):
        print("Handling request", request.query, request.match_info)
        return web.json_response({"error": "oops"})

    async def run(self) -> None:
        print(f"Starting GPTs frontend on port {self.port}")
        app = web.Application()
        app.router.add_get(self.base_url, self.actions_schema)
        app.router.add_get(self.base_url + "/", self.actions_schema)
        app.router.add_get(self.base_url + "/actions/{name}", self.handle_action)
        app.router.add_get(self.base_url + "/oauth/auth", self.handle_auth)
        app.router.add_post(self.base_url + "/oauth/token", self.handle_token)
        app.router.add_post(
            self.base_url + "/oauth/verify_access_code", self.handle_verify_access_code
        )
        handler = app.make_handler()
        loop = asyncio.get_event_loop()
        await loop.create_server(handler, "0.0.0.0", self.port)


async def start_gpts_action_server(
    tools: ToolRegistry, url: str, port: int = 6001, access_code: str | None = None
):
    while url.endswith("/"):
        url = url[:-1]
    o = urlparse(url)
    host = o.scheme + "://" + o.netloc
    base_url = o.path
    server = __GPTsActionServer(
        tools, host=host, base_url=base_url, port=port, access_code=access_code
    )
    await server.run()
