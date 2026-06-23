# local diff extraction via gitpython

from git import Repo, InvalidGitRepositoryError, NoSuchPathError


def get_diff(
    repo_path: str,
    *,
    commit: str | None = None,
    range: str | None = None,
    staged: bool = False,
) -> str:
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        raise ValueError(f"not a git repository: {repo_path}")
    except NoSuchPathError:
        raise ValueError(f"path does not exist: {repo_path}")

    if repo.bare:
        raise ValueError(f"cannot diff a bare repository: {repo_path}")

    # specific commit, show its diff against parent
    if commit:
        commit_obj = repo.commit(commit)
        if commit_obj.parents:
            return repo.git.diff(commit_obj.parents[0].hexsha, commit_obj.hexsha)
        # root commit, diff against empty tree
        empty_tree = repo.git.hash_object("-t", "tree", "/dev/null")
        return repo.git.diff(empty_tree, commit_obj.hexsha)

    # commit range like abc123..def456
    if range:
        if ".." not in range:
            raise ValueError(f"invalid range format, expected 'a..b': {range}")
        return repo.git.diff(range)

    # staged only
    if staged:
        diff = repo.git.diff("--cached")
        if not diff:
            raise ValueError("no staged changes found")
        return diff

    # default: all uncommitted changes (staged + unstaged)
    # diff HEAD to catch both, but if theres no commits yet diff the index
    if repo.head.is_valid():
        diff = repo.git.diff("HEAD")
    else:
        # no commits yet, show whats staged
        diff = repo.git.diff("--cached")

    if not diff:
        raise ValueError("no changes found")

    return diff
