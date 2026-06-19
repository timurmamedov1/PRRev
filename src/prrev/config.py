# config file loading and env var handling
# toml config comes later, just env vars for now

import os
from dataclasses import dataclass


@dataclass
class Config:
    provider: str = "anthropic"
    model: str | None = None
    github_token: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    max_items: int = 20


def load_config() -> Config:
    return Config(
        provider=os.environ.get("PRREV_PROVIDER", "anthropic"),
        model=os.environ.get("PRREV_MODEL"),
        github_token=os.environ.get("GITHUB_TOKEN"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        max_items=int(os.environ.get("PRREV_MAX_ITEMS", "20")),
    )
