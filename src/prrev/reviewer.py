# orchestrator, takes diff + provider, returns structured review
# auto-chunks when diff exceeds 80% of the providers context window

import asyncio

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult

DIFF_HEADER = "diff --git "

# chunk when diff uses more than 80% of the providers max input tokens
CHUNK_THRESHOLD = 0.8


def _split_by_file(diff: str) -> list[str]:
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
) -> ReviewResult:
    if not diff.strip():
        raise ValueError("empty diff")

    # check if we need to chunk based on token count
    token_count = provider.count_tokens(diff)
    threshold = int(provider.max_input_tokens * CHUNK_THRESHOLD)

    if token_count <= threshold:
        result = await provider.review(diff)
        return _truncate(result, max_items)

    # diff is too big, split by file and review in parallel
    file_diffs = _split_by_file(diff)
    if len(file_diffs) <= 1:
        # single file thats too big, just send it and hope for the best.
        # TODO: split within file by hunk
        result = await provider.review(diff)
        return _truncate(result, max_items)

    # skip files that individually exceed the threshold
    reviewable = []
    skipped_files = []
    for chunk in file_diffs:
        if provider.count_tokens(chunk) > threshold:
            # grab filename from the diff header for the warning
            first_line = chunk.split("\n", 1)[0]
            skipped_files.append(first_line)
        else:
            reviewable.append(chunk)

    results = await asyncio.gather(*[provider.review(d) for d in reviewable])
    merged = _merge_results(list(results))

    # add warnings for skipped files
    for skipped in skipped_files:
        merged.items.append(ReviewItem(
            severity="warning",
            file=skipped,
            line=None,
            summary="file skipped, too large for context window",
            explanation="this files diff exceeded the models token limit and was not reviewed.",
        ))

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
