import typer
import agentia.utils.repl

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
)


@app.command(help="Start the REPL")
def repl(agent: str):
    agentia.utils.repl.run(agent)


@app.callback()
def callback():
    pass


def main():
    app()
