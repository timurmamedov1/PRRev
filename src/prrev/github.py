# github api, fetch pr diffs and post review comments

from dataclasses import dataclass


@dataclass
class PRInfo:
    owner: str
    repo: str
    number: int
    title: str
    diff: str


def parse_pr_url(url: str) -> tuple[str, str, int]:
    raise NotImplementedError


async def fetch_pr(owner: str, repo: str, number: int, token: str) -> PRInfo:
    raise NotImplementedError


async def post_review(
    owner: str, repo: str, number: int, body: str, token: str
) -> None:
    raise NotImplementedError
