from typing import Annotated, Any
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Header,
    WebSocket,
    WebSocketDisconnect,
)
import rich
import uvicorn

from agentia.agent import Agent, CommunicationEvent, ToolCallEvent
from agentia.message import UserMessage

from dataclasses import dataclass, asdict, field

from agentia.tools import ClientTool


@dataclass
class BaseMessage:
    def to_json(self) -> Any:
        return asdict(self)


@dataclass
class Prompt(BaseMessage):
    content: str
    files: list[str] = field(default_factory=list)
    type: str = "prompt"


@dataclass
class ResponseStart(BaseMessage):
    type: str = "response.start"


@dataclass
class ToolStart(BaseMessage):
    tool: str
    id: str
    type: str = "response.tool.start"


@dataclass
class ToolEnd(BaseMessage):
    tool: str
    id: str
    result: Any
    type: str = "response.tool.end"


@dataclass
class CommunicationStart(BaseMessage):
    id: str
    parent: str
    child: str
    message: str
    type: str = "response.communication.start"


@dataclass
class CommunicationEnd(BaseMessage):
    id: str
    parent: str
    child: str
    message: str
    response: str
    type: str = "response.communication.end"


@dataclass
class UserConsent(BaseMessage):
    message: str
    type: str = "response.user-consent"


@dataclass
class UserConsentResult(BaseMessage):
    result: bool
    type: str = "user-consent"


@dataclass
class MessageStart(BaseMessage):
    type: str = "response.message.start"


@dataclass
class MessageEnd(BaseMessage):
    type: str = "response.message.end"


@dataclass
class MessageDelta(BaseMessage):
    content: str
    type: str = "response.message.delta"


@dataclass
class ResponseEnd(BaseMessage):
    type: str = "response.end"


@dataclass
class Error(BaseMessage):
    message: str
    type: str = "error"


@dataclass
class SetupClientTools(BaseMessage):
    tools: list[Any]
    type: str = "setup.client.tools"


@dataclass
class ClientToolCall(BaseMessage):
    name: str
    args: Any
    type: str = "client.tool.call"


@dataclass
class ClientToolCallResult(BaseMessage):
    name: str
    result: Any
    type: str = "client.tool.result"


async def __chat_server(agent_name: str, websocket: WebSocket):
    await websocket.accept()
    agent = Agent.load_from_config(agent_name)

    async def send(m: BaseMessage):
        await websocket.send_json(m.to_json())

    @agent.on_tool_start
    async def notify_tool_start(e: ToolCallEvent):
        await send(ToolStart(tool=e.tool.display_name, id=e.id))

    @agent.on_tool_end
    async def notify_tool_end(e: ToolCallEvent):
        await send(ToolEnd(tool=e.tool.display_name, id=e.id, result=e.result))

    @agent.on_user_consent
    async def notify_user_consent(msg: str):
        await send(UserConsent(message=msg))
        while True:
            response = await websocket.receive_json()
            if response["type"] == "user-consent":
                return response["result"]
            else:
                await send(Error(message="Invalid message"))

    @agent.on_commuication_start
    async def notify_communication_start(e: CommunicationEvent):
        await send(
            CommunicationStart(
                id=e.id, parent=e.parent.name, child=e.child.name, message=e.message
            )
        )

    @agent.on_commuication_end
    async def notify_communication_end(e: CommunicationEvent):
        await send(
            CommunicationEnd(
                id=e.id,
                parent=e.parent.name,
                child=e.child.name,
                message=e.message,
                response=e.response or "",
            )
        )

    async def execute_client_tool_call(tool: str, args: Any):
        await send(ClientToolCall(name=tool, args=args))
        res = await websocket.receive_json()
        while True:
            if res["type"] == "client.tool.result":
                return res["result"]
            else:
                await send(Error(message="Invalid message"))

    async def receive_request() -> Prompt | None:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "setup.client.tools":
                agent.tools.add_client_tools([ClientTool(**t) for t in data["tools"]])
                agent.on_client_tool_call(execute_client_tool_call)
            elif data["type"] == "prompt":
                return Prompt(content=data["content"], files=data.get("files", []))
            else:
                await send(Error(message="Invalid message"))

    while True:
        msg = await receive_request()
        if msg is None:
            continue
        response = agent.chat_completion([UserMessage(msg.content)], stream=True)
        await send(ResponseStart())
        async for stream in response:
            message_started = False
            async for delta in stream:
                if len(delta) == 0:
                    continue
                if not message_started:
                    await send(MessageStart())
                    message_started = True
                await send(MessageDelta(content=delta))
            if message_started:
                await send(MessageEnd())
        await send(ResponseEnd())


def run(agent: str):
    app = FastAPI()

    agent_instance = Agent.load_from_config(agent)

    if token := agent_instance.original_config.get("access_code"):
        known_tokens = {token} if isinstance(token, str) else set(token)
    else:
        known_tokens = set()

    if len(known_tokens) == 0:
        rich.print(
            "[bold red]WARNING: No access code provided, API server will be open to the public.[/bold red]"
        )

    def get_token(authorization: Annotated[str | None, Header()] = None) -> str | None:
        if len(known_tokens) == 0:
            return None
        token = (
            None
            if authorization is None or not authorization.startswith("Bearer ")
            else authorization[7:]
        )
        if token is None or token not in known_tokens:
            raise HTTPException(
                status_code=401, detail="Bearer token missing or unknown"
            )
        return token

    @app.get("/")
    async def get_agent_info(token: str | None = Depends(get_token)):
        return {
            "name": agent_instance.name,
            "icon": agent_instance.icon,
            "description": agent_instance.original_config.get("description"),
            "tools": list(agent_instance.original_config.get("tools", {}).keys()),
            "colleagues": [c.name for c in agent_instance.colleagues.values()],
        }

    @app.websocket("/chat")
    async def chat(websocket: WebSocket, token: str | None = Depends(get_token)):
        try:
            await __chat_server(agent, websocket)
        except WebSocketDisconnect as e:
            return

    uvicorn.run(app, host="localhost", port=8000)
