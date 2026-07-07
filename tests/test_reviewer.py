# tests for reviewer truncation and merging

import asyncio

import pytest

from prrev.llm.base import LLMProvider, ReviewItem, ReviewResult
from prrev.reviewer import _merge_results, _path_from_header, _truncate, review_diff


class FakeProvider(LLMProvider):
    # in-memory provider that tracks call counts, so we can prove the
    # byte-length shortcut avoids the count_tokens network round-trip.
    # count_by_length makes token counts follow text size for chunking tests,
    # fail_on makes review raise for chunks containing that substring
    def __init__(self, *, max_tokens=100_000, token_count=50, count_by_length=False, fail_on=None):
        self.max_input_tokens = max_tokens
        self._token_count = token_count
        self._count_by_length = count_by_length
        self._fail_on = fail_on
        self.count_tokens_calls = 0
        self.review_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls += 1
        if self._count_by_length:
            return len(text)
        return self._token_count

    async def review(self, diff: str) -> ReviewResult:
        self.review_calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.001)
            if self._fail_on and self._fail_on in diff:
                raise RuntimeError("provider blew up")
            return ReviewResult(items=[], summary="ok")
        finally:
            self.in_flight -= 1


def _three_file_diff():
    # three small files, each chunk under the fake threshold but the whole
    # diff over it, so review_diff takes the split path
    return "".join(
        f"diff --git a/{name} b/{name}\n" + "+line\n" * 8 for name in ("foo.py", "bar.py", "baz.py")
    )


def _item(severity="suggestion", file="test.py", line=1):
    return ReviewItem(
        severity=severity,
        file=file,
        line=line,
        summary=f"{severity} issue",
        explanation="test explanation",
    )


class TestTruncate:
    def test_under_limit(self):
        result = ReviewResult(items=[_item(), _item()], summary="ok")
        truncated = _truncate(result, max_items=5)
        assert len(truncated.items) == 2

    def test_at_limit(self):
        items = [_item() for _ in range(5)]
        result = ReviewResult(items=items, summary="ok")
        truncated = _truncate(result, max_items=5)
        assert len(truncated.items) == 5

    def test_over_limit_drops_suggestions_first(self):
        items = [
            _item("critical"),
            _item("suggestion"),
            _item("warning"),
            _item("suggestion"),
            _item("critical"),
        ]
        result = ReviewResult(items=items, summary="ok")
        truncated = _truncate(result, max_items=3)
        assert len(truncated.items) == 3
        severities = [i.severity for i in truncated.items]
        assert "suggestion" not in severities

    def test_keeps_criticals_over_warnings(self):
        items = [
            _item("warning"),
            _item("critical"),
            _item("warning"),
            _item("critical"),
            _item("warning"),
        ]
        result = ReviewResult(items=items, summary="ok")
        truncated = _truncate(result, max_items=2)
        assert all(i.severity == "critical" for i in truncated.items)

    def test_preserves_summary(self):
        items = [_item() for _ in range(5)]
        result = ReviewResult(items=items, summary="important summary")
        truncated = _truncate(result, max_items=2)
        assert truncated.summary == "important summary"


class TestMergeResults:
    def test_merges_items(self):
        r1 = ReviewResult(items=[_item("critical")], summary="bad")
        r2 = ReviewResult(items=[_item("suggestion")], summary="ok")
        merged = _merge_results([r1, r2])
        assert len(merged.items) == 2

    def test_merges_summaries(self):
        r1 = ReviewResult(items=[], summary="first")
        r2 = ReviewResult(items=[], summary="second")
        merged = _merge_results([r1, r2])
        assert "first" in merged.summary
        assert "second" in merged.summary

    def test_empty_results(self):
        merged = _merge_results([])
        assert len(merged.items) == 0
        assert merged.summary == "No issues found."

    def test_single_result(self):
        r = ReviewResult(items=[_item("warning")], summary="one")
        merged = _merge_results([r])
        assert len(merged.items) == 1
        assert merged.summary == "one"

    def test_skips_empty_summaries(self):
        r1 = ReviewResult(items=[], summary="")
        r2 = ReviewResult(items=[], summary="actual summary")
        merged = _merge_results([r1, r2])
        assert merged.summary == "actual summary"


class TestReviewDiff:
    async def test_small_diff_skips_count_tokens(self):
        # perf fix: byte-length shortcut skips the api round-trip for
        # anything that clearly fits under the threshold
        provider = FakeProvider(max_tokens=100_000)
        diff = "diff --git a/x b/x\n+hello\n"
        await review_diff(provider, diff)
        assert provider.count_tokens_calls == 0
        assert provider.review_calls == 1

    async def test_empty_diff_raises(self):
        provider = FakeProvider()
        with pytest.raises(ValueError, match="empty diff"):
            await review_diff(provider, "")

    async def test_zero_max_items_raises(self):
        provider = FakeProvider()
        with pytest.raises(ValueError, match="max_items must be positive"):
            await review_diff(provider, "diff --git a/x b/x\n+hi\n", max_items=0)

    async def test_negative_max_items_raises(self):
        provider = FakeProvider()
        with pytest.raises(ValueError, match="max_items must be positive"):
            await review_diff(provider, "diff --git a/x b/x\n+hi\n", max_items=-1)

    async def test_large_diff_falls_back_to_real_count(self):
        # byte-length exceeds threshold so we still call the real api to
        # decide, this covers the near-boundary path
        provider = FakeProvider(max_tokens=100, token_count=50)  # threshold=80
        big_diff = "diff --git a/x b/x\n" + "+padding line\n" * 100
        await review_diff(provider, big_diff)
        assert provider.count_tokens_calls >= 1


class TestPathFromHeader:
    def test_extracts_new_side_path(self):
        assert _path_from_header("diff --git a/src/app.py b/src/app.py") == "src/app.py"

    def test_rename_uses_new_path(self):
        assert _path_from_header("diff --git a/old.py b/new.py") == "new.py"

    def test_unparseable_header_falls_back(self):
        assert _path_from_header("Binary files differ") == "Binary files differ"


class TestChunkedReview:
    # max_tokens=100 gives threshold 80, count_by_length makes each ~77 byte
    # chunk fit while the full diff overflows, forcing the split path

    async def test_splits_and_reviews_each_file(self):
        provider = FakeProvider(max_tokens=100, count_by_length=True)
        result = await review_diff(provider, _three_file_diff())
        assert provider.review_calls == 3
        assert result.summary == "ok ok ok"

    async def test_failed_chunk_becomes_warning(self):
        provider = FakeProvider(max_tokens=100, count_by_length=True, fail_on="bar.py")
        result = await review_diff(provider, _three_file_diff())
        assert provider.review_calls == 3
        warnings = [i for i in result.items if i.severity == "warning"]
        assert len(warnings) == 1
        assert warnings[0].file == "bar.py"
        assert "provider blew up" in warnings[0].explanation
        assert warnings[0].notice is True

    async def test_oversized_file_skipped_with_warning(self):
        # one file blows past the threshold, the others still get reviewed
        big = "diff --git a/big.py b/big.py\n" + "+pad\n" * 40
        provider = FakeProvider(max_tokens=100, count_by_length=True)
        result = await review_diff(provider, _three_file_diff() + big)
        assert provider.review_calls == 3
        skipped = [i for i in result.items if "skipped" in i.summary]
        assert len(skipped) == 1
        assert skipped[0].file == "big.py"
        assert skipped[0].notice is True

    async def test_single_oversized_file_skipped(self):
        # a lone file over the threshold gets the skip warning, not an api call
        big = "diff --git a/big.py b/big.py\n" + "+pad\n" * 40
        provider = FakeProvider(max_tokens=100, count_by_length=True)
        result = await review_diff(provider, big)
        assert provider.review_calls == 0
        assert len(result.items) == 1
        assert "skipped" in result.items[0].summary

    async def test_concurrency_is_capped(self):
        diff = "".join(f"diff --git a/f{i}.py b/f{i}.py\n+line\n" for i in range(12))
        provider = FakeProvider(max_tokens=100, count_by_length=True)
        await review_diff(provider, diff)
        assert provider.review_calls == 12
        assert provider.max_in_flight == 5
