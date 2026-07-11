# tests for url parsing, diff splitting, github url detection, and review posting

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prrev.cli import _is_github_url
from prrev.github import _neutralize_mentions, parse_pr_url, post_review
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

    def test_repo_with_dots_ok(self):
        owner, repo, number = parse_pr_url("https://github.com/user/my.repo/pull/2")
        assert repo == "my.repo"

    def test_owner_with_traversal_rejected(self):
        with pytest.raises(ValueError, match="invalid github PR url"):
            parse_pr_url("https://github.com/../repo/pull/1")

    def test_repo_dotdot_rejected(self):
        with pytest.raises(ValueError, match="invalid github PR url"):
            parse_pr_url("https://github.com/user/../pull/1")

    def test_percent_encoded_rejected(self):
        with pytest.raises(ValueError, match="invalid github PR url"):
            parse_pr_url("https://github.com/user/repo%2Fother/pull/1")


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


class TestPostSanitization:
    def test_mentions_get_backticked(self):
        assert _neutralize_mentions("cc @octocat please") == "cc `@octocat` please"

    def test_emails_untouched(self):
        assert _neutralize_mentions("mail user@example.com") == "mail user@example.com"

    def test_already_backticked_untouched(self):
        assert _neutralize_mentions("`@user`") == "`@user`"

    async def test_posted_body_and_comments_sanitized(self):
        items = [
            {
                "file": "app.py",
                "line": 3,
                "severity": "warning",
                "summary": "ping @someone here",
                "explanation": "written by @octocat",
            }
        ]

        ok = AsyncMock()
        ok.status_code = 200
        ok.raise_for_status = lambda: None

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=ok)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            await post_review("o", "r", 1, "summary mentions @admin", "tok", items=items)
            payload = ctx.post.call_args[1]["json"]
            assert "`@admin`" in payload["body"]
            assert "Review generated by" in payload["body"]
            comment_body = payload["comments"][0]["body"]
            assert "`@someone`" in comment_body
            assert "`@octocat`" in comment_body


class TestRetry:
    def _client_with(self, responses):
        mock_client = patch("prrev.github.httpx.AsyncClient").start()
        ctx = AsyncMock()
        ctx.post = AsyncMock(side_effect=responses)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        return ctx

    def teardown_method(self):
        patch.stopall()

    async def test_retries_500_then_succeeds(self):
        req = httpx.Request("POST", "https://api.github.com/x")
        flaky = httpx.Response(500, request=req, headers={"retry-after": "0"})
        ok = AsyncMock()
        ok.status_code = 200
        ok.raise_for_status = lambda: None

        ctx = self._client_with([flaky, ok])
        await post_review("o", "r", 1, "body", "tok", items=None)
        assert ctx.post.call_count == 2

    async def test_retries_403_rate_limit(self):
        req = httpx.Request("POST", "https://api.github.com/x")
        limited = httpx.Response(
            403,
            request=req,
            headers={"retry-after": "0"},
            json={"message": "API rate limit exceeded for user"},
        )
        ok = AsyncMock()
        ok.status_code = 200
        ok.raise_for_status = lambda: None

        ctx = self._client_with([limited, ok])
        await post_review("o", "r", 1, "body", "tok", items=None)
        assert ctx.post.call_count == 2

    async def test_gives_up_after_max_retries(self):
        req = httpx.Request("POST", "https://api.github.com/x")
        down = httpx.Response(500, request=req, headers={"retry-after": "0"})

        ctx = self._client_with([down, down, down])
        with pytest.raises(httpx.HTTPStatusError):
            await post_review("o", "r", 1, "body", "tok", items=None)
        assert ctx.post.call_count == 3  # first try + MAX_RETRIES


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

    async def test_github_error_body_lands_in_message(self):
        # githubs json error detail should surface in the raised error
        req = httpx.Request("POST", "https://api.github.com/repos/o/r/pulls/1/reviews")
        resp = httpx.Response(
            422,
            request=req,
            json={
                "message": "Validation Failed",
                "errors": [{"resource": "PullRequestReview", "field": "line"}],
            },
        )

        with patch("prrev.github.httpx.AsyncClient") as mock_client:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=resp)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.HTTPStatusError, match="Validation Failed"):
                await post_review("o", "r", 1, "body", "tok", items=None)

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
