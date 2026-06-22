# tests for reviewer truncation and merging

from prrev.llm.base import ReviewItem, ReviewResult
from prrev.reviewer import _truncate, _merge_results


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
