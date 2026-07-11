# orchestrator, takes diff + provider, returns structured review
# auto-chunks when diff exceeds 80% of the providers context window

import asyncio
import re

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult

DIFF_HEADER = "diff --git "

_HEADER_PATH = re.compile(r"^diff --git a/.* b/(.*)$")


def _path_from_header(header_line: str) -> str:
    # pull the new-side path out of a "diff --git a/x b/x" header so
    # warnings show a filename instead of the raw header
    m = _HEADER_PATH.match(header_line)
    return m.group(1) if m else header_line


# chunk when diff uses more than 80% of the providers max input tokens
CHUNK_THRESHOLD = 0.8

# dont fire every chunk at the api at once, big diffs would trip rate limits
MAX_CONCURRENT_REVIEWS = 5


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


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
    if max_items < 1:
        raise ValueError(f"max_items must be positive, got {max_items}")

    if not diff.strip():
        raise ValueError("empty diff")

    threshold = int(provider.max_input_tokens * CHUNK_THRESHOLD)

    # utf-8 byte count is a safe upper bound on tokens bc tokenizers use
    # byte-level bpe, no token encodes less than 1 byte. when the diff
    # clearly fits we can skip the count_tokens call, which for anthropic
    # is a network round-trip on every review
    if _byte_len(diff) <= threshold or (await provider.count_tokens(diff)) <= threshold:
        result = await provider.review(diff)
        return _truncate(result, max_items)

    # diff is too big, split by file and review in parallel
    file_diffs = _split_by_file(diff)

    # skip files that individually exceed the threshold, same byte-length
    # shortcut so we dont call count_tokens for every small chunk.
    # TODO: split oversized files by hunk instead of skipping them
    reviewable = []
    skipped_files = []
    for chunk in file_diffs:
        if _byte_len(chunk) > threshold and (await provider.count_tokens(chunk)) > threshold:
            skipped_files.append(_path_from_header(chunk.split("\n", 1)[0]))
        else:
            reviewable.append(chunk)

    # review chunks in parallel. one chunk failing shouldnt sink the others,
    # so failures are collected and surfaced as warnings on the merged result
    sem = asyncio.Semaphore(MAX_CONCURRENT_REVIEWS)

    async def _review_chunk(chunk: str) -> ReviewResult:
        async with sem:
            return await provider.review(chunk)

    results = await asyncio.gather(
        *[_review_chunk(d) for d in reviewable],
        return_exceptions=True,
    )

    ok_results: list[ReviewResult] = []
    failed_chunks: list[tuple[str, Exception]] = []
    for chunk, res in zip(reviewable, results, strict=True):
        if isinstance(res, ReviewResult):
            ok_results.append(res)
        elif isinstance(res, Exception):
            failed_chunks.append((chunk, res))
        else:
            # cancellation and other BaseExceptions still propagate
            raise res

    merged = _merge_results(ok_results)

    for chunk, exc in failed_chunks:
        merged.items.append(
            ReviewItem(
                severity="warning",
                file=_path_from_header(chunk.split("\n", 1)[0]),
                line=None,
                summary="file not reviewed, provider call failed",
                explanation=f"reviewing this files diff failed: {exc}",
                notice=True,
            )
        )

    # add warnings for skipped files
    for skipped in skipped_files:
        merged.items.append(
            ReviewItem(
                severity="warning",
                file=skipped,
                line=None,
                summary="file skipped, too large for context window",
                explanation="this files diff exceeded the models token limit and was not reviewed.",
                notice=True,
            )
        )

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
