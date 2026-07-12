# tests for markdown and rich formatter output

import io

from rich.console import Console

from prrev import formatter as fmt
from prrev.formatter import _format_item, print_review, to_markdown
from prrev.llm.base import ReviewItem, ReviewResult


def _item(
    severity="warning",
    file="app.py",
    line=10,
    summary="something wrong",
    explanation="here is why",
):
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

    def test_empty_notice_suppressed(self):
        # posting path suppresses the notice when findings went inline
        result = ReviewResult(items=[], summary="found a bug on line 7")
        md = to_markdown(result, show_empty_notice=False)
        assert "No issues found." not in md
        assert "found a bug on line 7" in md

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


class TestPrintReview:
    def test_renders_items_and_file_count(self, monkeypatch):
        buf = io.StringIO()
        monkeypatch.setattr(fmt, "console", Console(file=buf, force_terminal=True))
        result = ReviewResult(items=[_item("critical")], summary="one bad thing")
        print_review(result, file_count=2)
        out = buf.getvalue()
        assert "Reviewed 2 files" in out
        assert "CRITICAL" in out
        assert "one bad thing" in out


class TestPrintReviewMarkupSafety:
    def test_summary_with_markup_like_text_does_not_raise(self, monkeypatch):
        # llm-produced summary can contain [/tag]-shaped substrings that
        # rich would try to parse as markup and raise MarkupError on
        monkeypatch.setattr(fmt, "console", Console(file=io.StringIO(), force_terminal=True))
        result = ReviewResult(items=[], summary="Removed the [/tool] wrapper")
        print_review(result)
