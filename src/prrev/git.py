# local diff extraction via gitpython


def get_diff(
    repo_path: str,
    *,
    commit: str | None = None,
    range: str | None = None,
    staged: bool = False,
) -> str:
    raise NotImplementedError
