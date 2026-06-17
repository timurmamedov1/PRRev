# single command cli, uses callback bc theres no subcommands

import typer

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    target: str = typer.Argument(..., help="Local repo path, commit, or GitHub PR URL"),
    commit: str | None = typer.Option(None, help="Review a specific commit"),
    range: str | None = typer.Option(None, help="Review a commit range (abc..def)"),
    staged: bool = typer.Option(False, help="Review only staged changes"),
    provider: str = typer.Option(None, help="LLM provider: anthropic or openai"),
    model: str = typer.Option(None, help="Model override"),
    post: bool = typer.Option(False, help="Post review as GitHub PR comment"),
    output: str | None = typer.Option(None, help="Write review to markdown file"),
    fail_on: str | None = typer.Option(
        None, help="Exit non-zero if issues at this severity or above (critical, warning)"
    ),
) -> None:
    raise NotImplementedError
