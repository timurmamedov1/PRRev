# tests for url parsing, diff splitting, github url detection, and review posting

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prrev.cli import _is_github_url
from prrev.github import parse_pr_url, post_review
from prrev.reviewer import _split_by_file


class TestParsePrUrl:
    def test_basic_url(self):
        owner, repo, number = parse_pr_url("https://github.com/user/repo/pull/42")
        assert owner == "user"
        assert repo == "repo"
        assert number == 42

    def test_url_with_trailing_path(self):
        # should still match, regex uses match() not fullmatch()
        owner, repo, number = parse_pr_url("https://github.com/org/project/pull/123/files")
        assert owner == "org"
        assert repo == "project"
        assert number == 123

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="invalid github PR url"):
            parse_pr_url("https://github.com/user/repo/issues/5")

    def test_not_github(self):
        with pytest.raises(ValueError, match="invalid github PR url"):
            parse_pr_url("https://gitlab.com/user/repo/pull/1")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_pr_url("")


class TestSplitByFile:
    def test_single_file(self):
        diff = "diff --git a/foo.py b/foo.py\n+hello\n"
        chunks = _split_by_file(diff)
        assert len(chunks) == 1
        assert "foo.py" in chunks[0]

    def test_multiple_files(self):
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "+line1\n"
            "diff --git a/bar.py b/bar.py\n"
            "+line2\n"
            "diff --git a/baz.py b/baz.py\n"
            "+line3\n"
        )
        chunks = _split_by_file(diff)
        assert len(chunks) == 3
        assert "foo.py" in chunks[0]
        assert "bar.py" in chunks[1]
        assert "baz.py" in chunks[2]

    def test_empty_diff(self):
        chunks = _split_by_file("")
        assert len(chunks) == 0

    def test_preserves_content(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,4 @@\n"
            " existing\n"
            "+new line\n"
            "diff --git a/b.py b/b.py\n"
            "+stuff\n"
        )
        chunks = _split_by_file(diff)
        assert len(chunks) == 2
        assert "+new line\n" in chunks[0]
        assert "+stuff\n" in chunks[1]


class TestIsGithubUrl:
    def test_valid_pr_url(self):
        assert _is_github_url("https://github.com/user/repo/pull/42") is True

    def test_issues_url(self):
        assert _is_github_url("https://github.com/user/repo/issues/42") is False

    def test_local_path(self):
        assert _is_github_url(".") is False
        assert _is_github_url("/home/user/project") is False

    def test_other_host(self):
        assert _is_github_url("https://gitlab.com/user/repo/pull/1") is False


class TestPostReview:
    async def test_inline_comments_posted(self):
        items = [
            {
                "file": "app.py",
                "line": 10,
                "severity": "warning",
                "summary": "unused import",
                "explanation": "os is unused",
            }
        ]

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            await post_review("o", "r", 1, "body", "tok", items=items)
            payload = ctx.post.call_args[1]["json"]
            assert len(payload["comments"]) == 1
            assert payload["comments"][0]["path"] == "app.py"

    async def test_422_falls_back_to_body(self):
        items = [
            {
                "file": "app.py",
                "line": 999,
                "severity": "warning",
                "summary": "bad line",
                "explanation": "not in diff",
            }
        ]

        rejected = AsyncMock()
        rejected.status_code = 422

        ok = AsyncMock()
        ok.status_code = 200
        ok.raise_for_status = lambda: None

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(side_effect=[rejected, ok])
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            await post_review("o", "r", 1, "body", "tok", items=items)
            assert ctx.post.call_count == 2
            fallback_payload = ctx.post.call_args_list[1][1]["json"]
            assert "comments" not in fallback_payload
            assert "app.py:999" in fallback_payload["body"]

    async def test_deletion_comment_uses_left_side(self):
        items = [
            {
                "file": "app.py",
                "line": 5,
                "severity": "warning",
                "summary": "removed guard",
                "explanation": "was needed",
                "side": "LEFT",
            }
        ]

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            await post_review("o", "r", 1, "body", "tok", items=items)
            payload = ctx.post.call_args[1]["json"]
            assert payload["comments"][0]["side"] == "LEFT"

    async def test_no_fallback_without_comments(self):
        # 422 without comments shouldnt retry, just raise
        rejected = AsyncMock()
        rejected.status_code = 422
        rejected.raise_for_status = lambda: (_ for _ in ()).throw(
            httpx.HTTPStatusError(
                "422",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(422),
            )
        )

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=rejected)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.HTTPStatusError):
                await post_review("o", "r", 1, "body", "tok", items=None)
