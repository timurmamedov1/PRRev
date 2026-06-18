# orchestrator, takes diff + provider, returns structured review
# chunking comes later, this is single pass for now

import asyncio

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult

# split delimiter in unified diffs
DIFF_HEADER = "diff --git "


def _split_by_file(diff: str) -> list[str]:
    # splits a multi-file diff into per-file chunks
    chunks = []
    current: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith(DIFF_HEADER) and current:
            chunks.append("".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append("".join(current))

    return chunks


async def review_diff(
    provider: LLMProvider,
    diff: str,
    *,
    max_items: int = 20,
    chunk: bool = False,
) -> ReviewResult:
    if not diff.strip():
        raise ValueError("empty diff")

    if not chunk:
        result = await provider.review(diff)
        return _truncate(result, max_items)

    # chunked mode, review each file in parallel
    file_diffs = _split_by_file(diff)
    if len(file_diffs) <= 1:
        result = await provider.review(diff)
        return _truncate(result, max_items)

    results = await asyncio.gather(*[provider.review(d) for d in file_diffs])
    merged = _merge_results(results)
    return _truncate(merged, max_items)


def _merge_results(results: list[ReviewResult]) -> ReviewResult:
    all_items: list[ReviewItem] = []
    summaries: list[str] = []

    for r in results:
        all_items.extend(r.items)
        if r.summary:
            summaries.append(r.summary)

    summary = " ".join(summaries) if summaries else "No issues found."
    return ReviewResult(items=all_items, summary=summary)


def _truncate(result: ReviewResult, max_items: int) -> ReviewResult:
    if len(result.items) <= max_items:
        return result

    # keep criticals first, then warnings, drop suggestions
    severity_order = {"critical": 0, "warning": 1, "suggestion": 2}
    sorted_items = sorted(result.items, key=lambda i: severity_order.get(i.severity, 2))
    return ReviewResult(items=sorted_items[:max_items], summary=result.summary)
