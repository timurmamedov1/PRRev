# cli integration tests using typer's test runner

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from prrev.cli import app
from prrev.llm.base import ReviewItem, ReviewResult

runner = CliRunner()


def _mock_review_result(items=None):
    return ReviewResult(
        items=items or [],
        summary="looks fine",
    )


def _warning_item():
    return ReviewItem(
        severity="warning",
        file="app.py",
        line=5,
        summary="unused var",
        explanation="x is never read",
    )


def _mock_config(**overrides):
    defaults = dict(
        provider="anthropic",
        model=None,
        anthropic_api_key="sk-test",
        openai_api_key=None,
        github_token=None,
        max_items=20,
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestLocalReview:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_reviews_local_repo(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result()
        result = runner.invoke(app, ["."])
        assert result.exit_code == 0

    @patch("prrev.cli.get_diff", side_effect=ValueError("no changes found"))
    @patch("prrev.cli.load_config")
    def test_no_changes_exits_2(self, mock_config, mock_diff):
        mock_config.return_value = _mock_config()
        result = runner.invoke(app, ["."])
        assert result.exit_code == 2
        assert "no changes found" in result.output

    @patch("prrev.cli.load_config", side_effect=ValueError("max_items must be an integer"))
    def test_invalid_config_exits_2(self, mock_config):
        result = runner.invoke(app, ["."])
        assert result.exit_code == 2
        assert "invalid config" in result.output

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.find_repo_root", return_value="/repo/root")
    @patch("prrev.cli.load_config")
    def test_config_loaded_from_repo_root(self, mock_config, mock_root, mock_diff, mock_review):
        # repo config is read from the discovered root, not the target path
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result()
        result = runner.invoke(app, ["sub/dir"])
        assert result.exit_code == 0
        mock_config.assert_called_once_with(repo_path="/repo/root")


class TestDebug:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_debug_reraises_the_real_error(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.side_effect = RuntimeError("boom")
        result = runner.invoke(app, [".", "--debug"])
        assert isinstance(result.exception, RuntimeError)

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_without_debug_prints_short_error(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.side_effect = RuntimeError("boom")
        result = runner.invoke(app, ["."])
        assert result.exit_code == 2
        assert "error: review failed: boom" in result.output


class TestFileCount:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff")
    @patch("prrev.cli.load_config")
    def test_added_diff_text_not_counted(self, mock_config, mock_diff, mock_review):
        # an added line containing diff header text isnt a file boundary
        mock_config.return_value = _mock_config()
        mock_diff.return_value = (
            "diff --git a/fixture.txt b/fixture.txt\n"
            "+diff --git a/embedded.py b/embedded.py\n"
            "+more added content\n"
        )
        mock_review.return_value = _mock_review_result()
        result = runner.invoke(app, ["."])
        assert result.exit_code == 0
        assert "Reviewed 1 files" in result.output


class TestFlagConflicts:
    @patch("prrev.cli.load_config")
    def test_commit_and_staged_conflict(self, mock_config):
        mock_config.return_value = _mock_config()
        result = runner.invoke(app, [".", "--commit", "abc", "--staged"])
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    @patch("prrev.cli.load_config")
    def test_commit_and_range_conflict(self, mock_config):
        mock_config.return_value = _mock_config()
        result = runner.invoke(app, [".", "--commit", "abc", "--range", "a..b"])
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    @patch("prrev.cli.load_config")
    def test_mode_flag_with_url_rejected(self, mock_config):
        mock_config.return_value = _mock_config(github_token="tok")
        result = runner.invoke(app, ["https://github.com/user/repo/pull/1", "--staged"])
        assert result.exit_code == 2
        assert "local repos" in result.output

    @patch("prrev.cli.load_config")
    def test_post_requires_url(self, mock_config):
        mock_config.return_value = _mock_config()
        result = runner.invoke(app, [".", "--post"])
        assert result.exit_code == 2
        assert "--post requires" in result.output


class TestArgumentOrder:
    # readme examples put the target first, so options must parse on
    # either side of it

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_options_after_target(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result()
        result = runner.invoke(app, [".", "--fail-on", "critical"])
        assert result.exit_code == 0

    @patch("prrev.cli.get_diff", side_effect=ValueError("no staged changes found"))
    @patch("prrev.cli.load_config")
    def test_staged_after_target(self, mock_config, mock_diff):
        mock_config.return_value = _mock_config()
        result = runner.invoke(app, [".", "--staged"])
        assert result.exit_code == 2
        assert "no staged changes" in result.output
        assert mock_diff.call_args.kwargs["staged"] is True


class TestFailOn:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_exits_1_when_threshold_met(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result([_warning_item()])
        result = runner.invoke(app, ["--fail-on", "warning", "."])
        assert result.exit_code == 1

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_exits_0_when_below_threshold(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result([_warning_item()])
        result = runner.invoke(app, ["--fail-on", "critical", "."])
        assert result.exit_code == 0

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_notices_dont_trip_threshold(self, mock_config, mock_diff, mock_review):
        # a skipped-file notice is severity warning but shouldnt fail ci
        mock_config.return_value = _mock_config()
        notice = ReviewItem(
            severity="warning",
            file="big.py",
            line=None,
            summary="file skipped, too large for context window",
            explanation="not reviewed",
            notice=True,
        )
        mock_review.return_value = _mock_review_result([notice])
        result = runner.invoke(app, ["--fail-on", "warning", "."])
        assert result.exit_code == 0

    def test_invalid_fail_on_exits_2(self):
        result = runner.invoke(app, ["--fail-on", "banana", "."])
        assert result.exit_code == 2


class TestOutput:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_writes_markdown_file(self, mock_config, mock_diff, mock_review, tmp_path):
        mock_config.return_value = _mock_config()
        mock_review.return_value = _mock_review_result()
        out = tmp_path / "review.md"
        result = runner.invoke(app, ["--output", str(out), "."])
        assert result.exit_code == 0
        assert out.exists()
        assert "PRRev" in out.read_text()


class TestProviderRouting:
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.get_diff", return_value="diff content")
    @patch("prrev.cli.load_config")
    def test_unknown_provider_exits_2(self, mock_config, mock_diff, mock_review):
        mock_config.return_value = _mock_config(provider="gemini")
        result = runner.invoke(app, ["."])
        assert result.exit_code == 2
        assert "error: unknown provider: gemini" in result.output


class TestGitHubUrl:
    @patch("prrev.cli.load_config")
    def test_missing_github_token_exits_2(self, mock_config):
        mock_config.return_value = _mock_config(github_token=None)
        result = runner.invoke(app, ["https://github.com/user/repo/pull/1"])
        assert result.exit_code == 2
        assert "GITHUB_TOKEN" in result.output

    @patch("prrev.cli.post_review", new_callable=AsyncMock)
    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.fetch_pr", new_callable=AsyncMock)
    @patch("prrev.cli.load_config")
    def test_body_excludes_inline_items(self, mock_config, mock_fetch, mock_review, mock_post):
        # located items go inline, the body carries only unplaceable ones
        mock_config.return_value = _mock_config(github_token="tok")
        mock_fetch.return_value = MagicMock(number=1, title="t", diff="diff --git a/x b/x\n")
        inline = ReviewItem(
            severity="warning",
            file="app.py",
            line=5,
            summary="inline finding",
            explanation="on a line",
        )
        unplaced = ReviewItem(
            severity="suggestion",
            file="app.py",
            line=None,
            summary="general finding",
            explanation="no line",
        )
        mock_review.return_value = ReviewResult(items=[inline, unplaced], summary="sum")
        result = runner.invoke(app, ["https://github.com/user/repo/pull/1", "--post"])
        assert result.exit_code == 0
        body = mock_post.call_args.args[3]
        assert "general finding" in body
        assert "inline finding" not in body
        assert len(mock_post.call_args.kwargs["items"]) == 2

    @patch("prrev.cli.review_diff", new_callable=AsyncMock)
    @patch("prrev.cli.fetch_pr", new_callable=AsyncMock)
    @patch("prrev.cli.load_config")
    def test_markup_in_pr_title_does_not_crash(self, mock_config, mock_fetch, mock_review):
        # pr title is attacker controlled, a bracket-tag-shaped title used to
        # crash the rich print with MarkupError before the review even ran
        mock_config.return_value = _mock_config(github_token="tok")
        pr = MagicMock(number=12, title="Fix the [/x] parser", diff="diff --git a/x b/x\n")
        mock_fetch.return_value = pr
        mock_review.return_value = _mock_review_result()
        result = runner.invoke(app, ["https://github.com/user/repo/pull/12"])
        assert result.exit_code == 0, result.exception
