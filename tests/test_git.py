# tests for local diff extraction

from pathlib import Path

import pytest
from git import Repo

from prrev.git import find_repo_root, get_diff


@pytest.fixture
def git_repo(tmp_path):
    """creates a real git repo with one commit and a tracked file"""
    repo = Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    f = tmp_path / "file.txt"
    f.write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")
    return tmp_path


class TestUncommittedChanges:
    def test_detects_unstaged_changes(self, git_repo):
        (git_repo / "file.txt").write_text("changed\n")
        diff = get_diff(str(git_repo))
        assert "changed" in diff

    def test_no_changes_raises(self, git_repo):
        with pytest.raises(ValueError, match="no changes found"):
            get_diff(str(git_repo))

    def test_detects_new_untracked_staged_file(self, git_repo):
        new = git_repo / "new.txt"
        new.write_text("new file\n")
        repo = Repo(git_repo)
        repo.index.add(["new.txt"])
        diff = get_diff(str(git_repo))
        assert "new file" in diff


class TestStagedOnly:
    def test_returns_staged_diff(self, git_repo):
        (git_repo / "file.txt").write_text("staged change\n")
        repo = Repo(git_repo)
        repo.index.add(["file.txt"])
        diff = get_diff(str(git_repo), staged=True)
        assert "staged change" in diff

    def test_no_staged_raises(self, git_repo):
        (git_repo / "file.txt").write_text("unstaged only\n")
        with pytest.raises(ValueError, match="no staged changes"):
            get_diff(str(git_repo), staged=True)


class TestCommit:
    def test_shows_commit_diff(self, git_repo):
        (git_repo / "file.txt").write_text("v2\n")
        repo = Repo(git_repo)
        repo.index.add(["file.txt"])
        c = repo.index.commit("second")
        diff = get_diff(str(git_repo), commit=c.hexsha)
        assert "v2" in diff

    def test_root_commit(self, git_repo):
        repo = Repo(git_repo)
        root = list(repo.iter_commits())[-1]
        diff = get_diff(str(git_repo), commit=root.hexsha)
        assert "hello" in diff

    def test_dash_prefixed_commit_rejected(self, git_repo):
        # dont let a leading-dash value flow into git as a flag
        with pytest.raises(ValueError, match="invalid commit"):
            get_diff(str(git_repo), commit="--output=/tmp/pwn")

    def test_unknown_commit_raises_clean_error(self, git_repo):
        # a rev that doesnt resolve should surface as ValueError, not a
        # gitpython exception the cli cant catch
        with pytest.raises(ValueError, match="unknown commit"):
            get_diff(str(git_repo), commit="deadbeef123")


class TestRange:
    def test_range_diff(self, git_repo):
        repo = Repo(git_repo)
        first = repo.head.commit.hexsha

        (git_repo / "file.txt").write_text("range change\n")
        repo.index.add(["file.txt"])
        second = repo.index.commit("second")

        diff = get_diff(str(git_repo), commit_range=f"{first}..{second.hexsha}")
        assert "range change" in diff

    def test_invalid_range_raises(self, git_repo):
        with pytest.raises(ValueError, match="invalid range format"):
            get_diff(str(git_repo), commit_range="abc123")

    def test_unknown_range_raises_clean_error(self, git_repo):
        with pytest.raises(ValueError, match="cannot resolve range"):
            get_diff(str(git_repo), commit_range="foo..bar")

    def test_dash_prefixed_range_rejected(self, git_repo, tmp_path):
        # git treats --output=<file> as a flag that overwrites files, and the
        # ".." check is trivially satisfied. reject before it hits git and
        # confirm nothing got written
        target = tmp_path / "pwned"
        with pytest.raises(ValueError, match="invalid range format"):
            get_diff(str(git_repo), commit_range=f"--output={target}..HEAD")
        assert not target.exists()


class TestRepoRoot:
    def test_get_diff_from_subdir(self, git_repo):
        (git_repo / "sub").mkdir()
        (git_repo / "file.txt").write_text("changed\n")
        diff = get_diff(str(git_repo / "sub"))
        assert "changed" in diff

    def test_find_repo_root_from_subdir(self, git_repo):
        (git_repo / "sub").mkdir()
        root = find_repo_root(str(git_repo / "sub"))
        assert root is not None
        assert Path(root).resolve() == git_repo.resolve()

    def test_find_repo_root_outside_repo(self, tmp_path):
        assert find_repo_root(str(tmp_path)) is None


class TestErrors:
    def test_not_a_repo(self, tmp_path):
        with pytest.raises(ValueError, match="not a git repository"):
            get_diff(str(tmp_path))

    def test_path_does_not_exist(self):
        with pytest.raises(ValueError, match="path does not exist"):
            get_diff("/nonexistent/fake/path")
