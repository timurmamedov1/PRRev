# single command cli, uses callback bc theres no subcommands

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from prrev.formatter import print_review, to_markdown
from prrev.git import get_diff
from prrev.llm.anthropic import AnthropicProvider
from prrev.reviewer import review_diff

app = typer.Typer(add_completion=False)
console = Console()


def _is_github_url(target: str) -> bool:
    return target.startswith("https://github.com/") and "/pull/" in target


@app.callback(invoke_without_command=True)
def main(
    target: str = typer.Argument(..., help="Local repo path or GitHub PR URL"),
    commit: str | None = typer.Option(None, help="Review a specific commit"),
    range: str | None = typer.Option(None, help="Review a commit range (abc..def)"),
    staged: bool = typer.Option(False, help="Review only staged changes"),
    provider: str | None = typer.Option(None, help="LLM provider: anthropic or openai"),
    model: str | None = typer.Option(None, help="Model override"),
    post: bool = typer.Option(False, help="Post review as GitHub PR comment"),
    output: str | None = typer.Option(None, help="Write review to markdown file"),
    fail_on: str | None = typer.Option(
        None, help="Exit non-zero if issues at this severity or above (critical, warning)"
    ),
) -> None:
    # github pr support comes later
    if _is_github_url(target):
        console.print("github PR review not implemented yet", style="red")
        raise typer.Exit(2)

    # local repo path
    try:
        diff = get_diff(target, commit=commit, range=range, staged=staged)
    except ValueError as e:
        console.print(f"error: {e}", style="red")
        raise typer.Exit(2)

    # pick provider, only anthropic for now
    prov = provider or "anthropic"
    if prov != "anthropic":
        console.print(f"provider '{prov}' not implemented yet", style="red")
        raise typer.Exit(2)

    try:
        llm = AnthropicProvider(model=model) if model else AnthropicProvider()
    except ValueError as e:
        console.print(f"error: {e}", style="red")
        raise typer.Exit(2)

    # run the review
    try:
        result = asyncio.run(review_diff(llm, diff))
    except Exception as e:
        console.print(f"review failed: {e}", style="red")
        raise typer.Exit(2)

    # count files in the diff for the header
    file_count = diff.count("diff --git ")
    print_review(result, file_count=file_count)

    # markdown output
    if output:
        Path(output).write_text(to_markdown(result))
        console.print(f"\nreview written to {output}", style="dim")

    # --post comes later with github integration
    if post:
        console.print("--post not implemented yet", style="red")
        raise typer.Exit(2)
