# cli integration tests using typer's test runner

from unittest.mock import AsyncMock, patch, MagicMock

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
        provider="anthropic", model=None,
        anthropic_api_key="sk-test", openai_api_key=None,
        github_token=None, max_items=20,
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
        assert "unknown provider" in result.output


class TestGitHubUrl:
    @patch("prrev.cli.load_config")
    def test_missing_github_token_exits_2(self, mock_config):
        mock_config.return_value = _mock_config(github_token=None)
        result = runner.invoke(app, ["https://github.com/user/repo/pull/1"])
        assert result.exit_code == 2
        assert "GITHUB_TOKEN" in result.output
