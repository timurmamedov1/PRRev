# config file loading and env var handling

from dataclasses import dataclass


@dataclass
class Config:
    provider: str = "anthropic"
    model: str | None = None
    github_token: str | None = None
    max_items: int = 20


def load_config(repo_path: str | None = None) -> Config:
    raise NotImplementedError
