# tests for markdown and rich formatter output

from prrev.formatter import to_markdown, _format_item
from prrev.llm.base import ReviewItem, ReviewResult


def _item(severity="warning", file="app.py", line=10, summary="something wrong", explanation="here is why"):
    return ReviewItem(
        severity=severity,
        file=file,
        line=line,
        summary=summary,
        explanation=explanation,
    )


class TestToMarkdown:
    def test_has_header(self):
        result = ReviewResult(items=[], summary="all good")
        md = to_markdown(result)
        assert md.startswith("# PRRev Code Review")

    def test_no_issues(self):
        result = ReviewResult(items=[], summary="clean")
        md = to_markdown(result)
        assert "No issues found." in md

    def test_item_severity_label(self):
        result = ReviewResult(items=[_item("critical")], summary="bad")
        md = to_markdown(result)
        assert "CRITICAL" in md

    def test_item_file_and_line(self):
        result = ReviewResult(items=[_item(file="main.py", line=42)], summary="ok")
        md = to_markdown(result)
        assert "main.py:42" in md

    def test_item_no_line(self):
        result = ReviewResult(items=[_item(line=None)], summary="ok")
        md = to_markdown(result)
        assert "app.py\n" in md
        assert "app.py:" not in md

    def test_item_summary_bold(self):
        item = _item(summary="use a set here")
        result = ReviewResult(items=[item], summary="ok")
        md = to_markdown(result)
        assert "**use a set here**" in md

    def test_item_explanation(self):
        item = _item(explanation="sets have O(1) lookup")
        result = ReviewResult(items=[item], summary="ok")
        md = to_markdown(result)
        assert "sets have O(1) lookup" in md

    def test_summary_section(self):
        result = ReviewResult(items=[], summary="no problems")
        md = to_markdown(result)
        assert "## Summary" in md
        assert "no problems" in md

    def test_multiple_items(self):
        items = [_item("critical"), _item("suggestion"), _item("warning")]
        result = ReviewResult(items=items, summary="mixed")
        md = to_markdown(result)
        assert "CRITICAL" in md
        assert "SUGGESTION" in md
        assert "WARNING" in md


class TestFormatItem:
    def test_contains_severity_label(self):
        text = _format_item(_item("critical"))
        plain = text.plain
        assert "CRITICAL" in plain

    def test_contains_file(self):
        text = _format_item(_item(file="utils.py"))
        assert "utils.py" in text.plain

    def test_contains_line(self):
        text = _format_item(_item(line=99))
        assert ":99" in text.plain

    def test_no_line_no_colon(self):
        text = _format_item(_item(line=None))
        assert ":None" not in text.plain

    def test_contains_summary_and_explanation(self):
        text = _format_item(_item())
        plain = text.plain
        assert "something wrong" in plain
        assert "here is why" in plain
