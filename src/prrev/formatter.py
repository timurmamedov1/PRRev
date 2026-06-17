# rich terminal output and markdown file export

from prrev.llm.base import ReviewResult


def print_review(result: ReviewResult) -> None:
    raise NotImplementedError


def to_markdown(result: ReviewResult) -> str:
    raise NotImplementedError
