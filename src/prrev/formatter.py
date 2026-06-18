# rich terminal output and markdown file export

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from prrev.llm.base import ReviewItem, ReviewResult

console = Console()

SEVERITY_STYLES = {
    "critical": ("red", "CRITICAL"),
    "warning": ("yellow", "WARNING"),
    "suggestion": ("green", "SUGGESTION"),
}


def _format_item(item: ReviewItem) -> Text:
    color, label = SEVERITY_STYLES.get(item.severity, ("white", item.severity.upper()))

    text = Text()
    text.append(f"{label:10s}", style=f"bold {color}")
    text.append(f"  {item.file}", style="bold")
    if item.line is not None:
        text.append(f":{item.line}", style="bold")
    text.append("\n")
    text.append(f"   {item.summary}\n", style="bold")
    text.append(f"   {item.explanation}\n")

    return text


def print_review(result: ReviewResult, file_count: int = 0) -> None:
    # header panel
    subtitle = f"Reviewed {file_count} files" if file_count else "Review"
    console.print(Panel(
        Text("PRRev", style="bold white"),
        subtitle=subtitle,
        border_style="blue",
    ))
    console.print()

    if not result.items:
        console.print("  No issues found.", style="bold green")
        console.print()
    else:
        for item in result.items:
            console.print(_format_item(item))

    # summary at the bottom
    console.print(Panel(result.summary, title="Summary", border_style="dim"))


def to_markdown(result: ReviewResult) -> str:
    lines = ["# PRRev Code Review\n"]

    if not result.items:
        lines.append("No issues found.\n")
    else:
        for item in result.items:
            color, label = SEVERITY_STYLES.get(item.severity, ("white", item.severity.upper()))
            location = item.file
            if item.line is not None:
                location += f":{item.line}"
            lines.append(f"### {label} — {location}\n")
            lines.append(f"**{item.summary}**\n")
            lines.append(f"{item.explanation}\n")

    lines.append(f"## Summary\n")
    lines.append(f"{result.summary}\n")

    return "\n".join(lines)
