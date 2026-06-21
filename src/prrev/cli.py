# single command cli, uses callback bc theres no subcommands

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from prrev.config import load_config
from prrev.formatter import print_review, to_markdown
from prrev.git import get_diff
from prrev.github import fetch_pr, parse_pr_url, post_review
from prrev.llm.anthropic import AnthropicProvider
from prrev.llm.openai import OpenAIProvider
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
    # cli flags override config, config fills in defaults
    repo_path = target if not _is_github_url(target) else None
    cfg = load_config(repo_path=repo_path)
    prov = provider or cfg.provider
    mdl = model or cfg.model

    # route based on target type
    if _is_github_url(target):
        if not cfg.github_token:
            console.print("error: GITHUB_TOKEN not set", style="red")
            raise typer.Exit(2)
        try:
            owner, repo, number = parse_pr_url(target)
            pr = asyncio.run(fetch_pr(owner, repo, number, cfg.github_token))
            diff = pr.diff
            console.print(f"reviewing PR #{pr.number}: {pr.title}", style="bold")
        except ValueError as e:
            console.print(f"error: {e}", style="red")
            raise typer.Exit(2)
        except Exception as e:
            console.print(f"failed to fetch PR: {e}", style="red")
            raise typer.Exit(2)
    else:
        try:
            diff = get_diff(target, commit=commit, range=range, staged=staged)
        except ValueError as e:
            console.print(f"error: {e}", style="red")
            raise typer.Exit(2)

    # pick provider
    try:
        if prov == "openai":
            llm = OpenAIProvider(model=mdl, api_key=cfg.openai_api_key) if mdl else OpenAIProvider(api_key=cfg.openai_api_key)
        elif prov == "anthropic":
            llm = AnthropicProvider(model=mdl, api_key=cfg.anthropic_api_key) if mdl else AnthropicProvider(api_key=cfg.anthropic_api_key)
        else:
            console.print(f"unknown provider: {prov}", style="red")
            raise typer.Exit(2)
    except ValueError as e:
        console.print(f"error: {e}", style="red")
        raise typer.Exit(2)

    # run the review
    try:
        result = asyncio.run(review_diff(llm, diff, max_items=cfg.max_items))
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

    # post review as github pr comment
    if post:
        if not _is_github_url(target):
            console.print("error: --post only works with github PR urls", style="red")
            raise typer.Exit(2)
        if not cfg.github_token:
            console.print("error: GITHUB_TOKEN not set", style="red")
            raise typer.Exit(2)
        try:
            items_for_api = [
                {"file": i.file, "line": i.line, "severity": i.severity,
                 "summary": i.summary, "explanation": i.explanation}
                for i in result.items
            ]
            body = to_markdown(result)
            asyncio.run(post_review(owner, repo, number, body, cfg.github_token, items=items_for_api))
            console.print("\nreview posted to PR", style="bold green")
        except Exception as e:
            console.print(f"failed to post review: {e}", style="red")
            raise typer.Exit(2)
