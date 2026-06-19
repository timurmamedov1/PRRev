# github api, fetch pr diffs and post review comments
# post_review comes in a later commit

import re
from dataclasses import dataclass

import httpx

API_BASE = "https://api.github.com"

# matches urls like https://github.com/owner/repo/pull/42
PR_URL_PATTERN = re.compile(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)")


@dataclass
class PRInfo:
    owner: str
    repo: str
    number: int
    title: str
    diff: str


def parse_pr_url(url: str) -> tuple[str, str, int]:
    match = PR_URL_PATTERN.match(url)
    if not match:
        raise ValueError(f"invalid github PR url: {url}")
    return match.group(1), match.group(2), int(match.group(3))


async def fetch_pr(owner: str, repo: str, number: int, token: str) -> PRInfo:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(base_url=API_BASE, headers=headers) as client:
        # get pr metadata
        resp = await client.get(f"/repos/{owner}/{repo}/pulls/{number}")
        resp.raise_for_status()
        pr_data = resp.json()

        # get the full diff using the diff accept header
        diff_resp = await client.get(
            f"/repos/{owner}/{repo}/pulls/{number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        diff_resp.raise_for_status()

    return PRInfo(
        owner=owner,
        repo=repo,
        number=number,
        title=pr_data.get("title", ""),
        diff=diff_resp.text,
    )


async def post_review(
    owner: str, repo: str, number: int, body: str, token: str
) -> None:
    raise NotImplementedError
