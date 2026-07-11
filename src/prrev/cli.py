# single command cli, typer runs the lone command directly so options
# parse on either side of the target

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from prrev.config import load_config
from prrev.formatter import print_review, to_markdown
from prrev.git import find_repo_root, get_diff
from prrev.github import fetch_pr, parse_pr_url, post_review
from prrev.llm.anthropic import AnthropicProvider
from prrev.llm.base import ReviewResult
from prrev.llm.openai import OpenAIProvider
from prrev.reviewer import review_diff

app = typer.Typer(add_completion=False)
console = Console()


def _fail(message: str, *, debug: bool) -> None:
    # only call from inside an except block. --debug swaps the one-line
    # message for the real traceback
    if debug:
        raise
    console.print(message, style="red")
    raise typer.Exit(2) from None


def _is_github_url(target: str) -> bool:
    return target.startswith("https://github.com/") and "/pull/" in target


# provider name -> (class, config field holding its api key)
PROVIDERS = {
    "openai": (OpenAIProvider, "openai_api_key"),
    "anthropic": (AnthropicProvider, "anthropic_api_key"),
}


def _make_provider(name: str, model: str | None, cfg):
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider: {name}")
    cls, key_field = PROVIDERS[name]
    kwargs = {"api_key": getattr(cfg, key_field)}
    if model:
        kwargs["model"] = model
    return cls(**kwargs)


async def _run(
    target: str,
    cfg,
    llm,
    *,
    post: bool,
    owner: str | None = None,
    repo: str | None = None,
    number: int | None = None,
):
    # fetch diff
    if _is_github_url(target):
        pr = await fetch_pr(owner, repo, number, cfg.github_token)
        diff = pr.diff
        console.print(
            f"reviewing PR #{pr.number}: {pr.title}",
            style="bold",
            markup=False,
        )
    else:
        raise RuntimeError("_run should not be called for local diffs")

    result = await review_diff(llm, diff, max_items=cfg.max_items)

    if post:
        items_for_api = [
            {
                "file": i.file,
                "line": i.line,
                "severity": i.severity,
                "summary": i.summary,
                "explanation": i.explanation,
                "side": i.side,
            }
            for i in result.items
        ]
        # located items become inline comments, so the body only carries the
        # summary plus items that cant be placed on a line
        body_items = [i for i in result.items if not (i.file and i.line)]
        body = to_markdown(ReviewResult(items=body_items, summary=result.summary))
        await post_review(
            owner,
            repo,
            number,
            body,
            cfg.github_token,
            items=items_for_api,
        )
        console.print("\nreview posted to PR", style="bold green")

    return diff, result


@app.command()
def main(
    target: str = typer.Argument(..., help="Local repo path or GitHub PR URL"),
    commit: str | None = typer.Option(None, help="Review a specific commit"),
    commit_range: str | None = typer.Option(
        None,
        "--range",
        help="Review a commit range (abc..def)",
    ),
    staged: bool = typer.Option(False, help="Review only staged changes"),
    provider: str | None = typer.Option(None, help="LLM provider: anthropic or openai"),
    model: str | None = typer.Option(None, help="Model override"),
    post: bool = typer.Option(False, help="Post review as GitHub PR comment"),
    output: str | None = typer.Option(None, help="Write review to markdown file"),
    fail_on: str | None = typer.Option(
        None,
        help="Exit 1 if issues at this severity or above",
    ),
    debug: bool = typer.Option(False, help="Show full tracebacks instead of short errors"),
) -> None:
    # validate --fail-on early. tuple keeps the error message order stable
    valid_severities = ("critical", "warning", "suggestion")
    if fail_on and fail_on not in valid_severities:
        msg = f"error: --fail-on must be one of: {', '.join(valid_severities)}"
        console.print(msg, style="red")
        raise typer.Exit(2)

    # mode flags pick one diff source, so combinations are rejected up front
    is_url = _is_github_url(target)
    mode_flags = [
        name
        for name, value in (("--commit", commit), ("--range", commit_range), ("--staged", staged))
        if value
    ]
    if len(mode_flags) > 1:
        console.print(f"error: {' and '.join(mode_flags)} are mutually exclusive", style="red")
        raise typer.Exit(2)
    if is_url and mode_flags:
        console.print(f"error: {mode_flags[0]} only applies to local repos", style="red")
        raise typer.Exit(2)
    if post and not is_url:
        console.print("error: --post requires a github pr url target", style="red")
        raise typer.Exit(2)

    # cli flags override config, config fills in defaults. repo config lives
    # at the repo root, which may be above the target path
    repo_path = None
    if not is_url:
        repo_path = find_repo_root(target) or target
    try:
        cfg = load_config(repo_path=repo_path)
    except ValueError as e:
        # also covers TOMLDecodeError from a malformed config file
        _fail(f"error: invalid config: {e}", debug=debug)
    prov = provider or cfg.provider
    mdl = model or cfg.model

    # pick provider
    try:
        llm = _make_provider(prov, mdl, cfg)
    except ValueError as e:
        _fail(f"error: {e}", debug=debug)

    # route based on target type
    if is_url:
        if not cfg.github_token:
            console.print("error: GITHUB_TOKEN not set", style="red")
            raise typer.Exit(2)
        try:
            owner, repo, number = parse_pr_url(target)
        except ValueError as e:
            _fail(f"error: {e}", debug=debug)

        try:
            diff, result = asyncio.run(
                _run(
                    target,
                    cfg,
                    llm,
                    post=post,
                    owner=owner,
                    repo=repo,
                    number=number,
                )
            )
        except ValueError as e:
            _fail(f"error: {e}", debug=debug)
        except Exception as e:
            _fail(f"error: review failed: {e}", debug=debug)
    else:
        try:
            diff = get_diff(
                target,
                commit=commit,
                commit_range=commit_range,
                staged=staged,
            )
        except ValueError as e:
            _fail(f"error: {e}", debug=debug)

        try:
            result = asyncio.run(
                review_diff(llm, diff, max_items=cfg.max_items),
            )
        except Exception as e:
            _fail(f"error: review failed: {e}", debug=debug)

    # count files in the diff for the header. only header lines count,
    # added content can legitimately contain the same text
    file_count = sum(1 for line in diff.splitlines() if line.startswith("diff --git "))
    print_review(result, file_count=file_count)

    # markdown output
    if output:
        out_path = Path(output).resolve()
        if not out_path.parent.is_dir():
            console.print(
                f"error: directory does not exist: {out_path.parent}",
                style="red",
            )
            raise typer.Exit(2)
        out_path.write_text(to_markdown(result), encoding="utf-8")
        console.print(f"\nreview written to {output}", style="dim")

    # exit code based on --fail-on threshold. tool notices (skipped files,
    # failed chunks) dont count as findings for ci purposes
    if fail_on and result.items:
        severity_rank = {"critical": 0, "warning": 1, "suggestion": 2}
        threshold = severity_rank[fail_on]
        findings = (i for i in result.items if not i.notice)
        if any(severity_rank.get(i.severity, 2) <= threshold for i in findings):
            raise typer.Exit(1)
