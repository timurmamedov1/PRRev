# local diff extraction via gitpython

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo
from gitdb.exc import BadName, BadObject


def get_diff(
    repo_path: str,
    *,
    commit: str | None = None,
    commit_range: str | None = None,
    staged: bool = False,
) -> str:
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        raise ValueError(f"not a git repository: {repo_path}") from None
    except NoSuchPathError:
        raise ValueError(f"path does not exist: {repo_path}") from None

    if repo.bare:
        raise ValueError(f"cannot diff a bare repository: {repo_path}")

    # specific commit, show its diff against parent
    if commit:
        # reject dash-prefixed values, git would treat them as options
        if commit.startswith("-"):
            raise ValueError(f"invalid commit: {commit}")
        try:
            commit_obj = repo.commit(commit)
        except (BadName, BadObject, ValueError):
            raise ValueError(f"unknown commit: {commit}") from None
        if commit_obj.parents:
            return repo.git.diff(commit_obj.parents[0].hexsha, commit_obj.hexsha)
        # root commit, diff against empty tree
        empty_tree = repo.git.hash_object("-t", "tree", "/dev/null")
        return repo.git.diff(empty_tree, commit_obj.hexsha)

    # commit range like abc123..def456. reject dash-prefixed values and pass
    # --end-of-options so git cant interpret the range as a flag like
    # --output=/some/path which would overwrite arbitrary files
    if commit_range:
        if commit_range.startswith("-") or ".." not in commit_range:
            raise ValueError(f"invalid range format, expected 'a..b': {commit_range}")
        try:
            return repo.git.diff("--end-of-options", commit_range)
        except GitCommandError:
            raise ValueError(f"cannot resolve range: {commit_range}") from None

    # staged only
    if staged:
        diff = repo.git.diff("--cached")
        if not diff:
            raise ValueError("no staged changes found")
        return diff

    # default: all uncommitted changes (staged + unstaged)
    # diff HEAD to catch both, but if theres no commits yet diff the index
    diff = repo.git.diff("HEAD") if repo.head.is_valid() else repo.git.diff("--cached")

    if not diff:
        raise ValueError("no changes found")

    return diff
