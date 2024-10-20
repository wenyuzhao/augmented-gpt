import asyncio
from pathlib import Path
from typing import Annotated
import typer
import agentia.utils
from agentia.agent import Agent

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
)


@app.command(help="Start the command line REPL")
def repl(agent: str):
    agentia.utils.repl.run(agent)


@app.command(help="Start GPTs action server")
def gpts(agent: str, access_code: Annotated[str | None, "The access code"] = None):
    a = Agent.load_from_config(agent)
    access_code = access_code or a.original_config.get("access_code")
    token_storage = Path.cwd() / ".cache" / "gpts" / "tokens.json"
    if not token_storage.parent.exists():
        token_storage.parent.mkdir(parents=True)
    asyncio.run(
        agentia.utils.gpts.start_gpts_action_server(
            tools=a.tools,
            url="gpts.wenyu.me",
            access_code=access_code,
            token_storage=token_storage,
        )
    )


@app.callback()
def callback():
    pass


def main():
    app()
