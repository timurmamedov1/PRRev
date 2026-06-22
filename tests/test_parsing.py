# tests for url parsing, diff splitting, and github url detection

import pytest

from prrev.github import parse_pr_url
from prrev.reviewer import _split_by_file
from prrev.cli import _is_github_url


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
