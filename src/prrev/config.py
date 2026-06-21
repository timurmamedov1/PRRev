# config file loading and env var handling
# precedence: cli flags > env vars > repo .prrev.toml > global config > defaults
# tokens only from env vars or global config, never repo config (security)

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

GLOBAL_CONFIG_PATH = Path.home() / ".config" / "prrev" / "config.toml"
REPO_CONFIG_NAME = ".prrev.toml"


@dataclass
class Config:
    provider: str = "anthropic"
    model: str | None = None
    github_token: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    max_items: int = 20


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(repo_path: str | None = None) -> Config:
    # start with defaults
    cfg = Config()

    # global config
    global_data = _load_toml(GLOBAL_CONFIG_PATH)
    _apply_toml(cfg, global_data, allow_tokens=True)

    # repo config, tokens not allowed here
    if repo_path:
        repo_data = _load_toml(Path(repo_path) / REPO_CONFIG_NAME)
        _apply_toml(cfg, repo_data, allow_tokens=False)

    # env vars override everything
    if v := os.environ.get("PRREV_PROVIDER"):
        cfg.provider = v
    if v := os.environ.get("PRREV_MODEL"):
        cfg.model = v
    if v := os.environ.get("PRREV_MAX_ITEMS"):
        cfg.max_items = int(v)
    if v := os.environ.get("GITHUB_TOKEN"):
        cfg.github_token = v
    if v := os.environ.get("ANTHROPIC_API_KEY"):
        cfg.anthropic_api_key = v
    if v := os.environ.get("OPENAI_API_KEY"):
        cfg.openai_api_key = v

    return cfg


def _apply_toml(cfg: Config, data: dict, *, allow_tokens: bool) -> None:
    llm = data.get("llm", {})
    if v := llm.get("provider"):
        cfg.provider = v
    if v := llm.get("model"):
        cfg.model = v

    review = data.get("review", {})
    if v := review.get("max_items"):
        cfg.max_items = int(v)

    # tokens only from global config, not repo config
    if allow_tokens:
        github = data.get("github", {})
        if v := github.get("token"):
            cfg.github_token = v
        llm_keys = data.get("llm", {})
        if v := llm_keys.get("anthropic_api_key"):
            cfg.anthropic_api_key = v
        if v := llm_keys.get("openai_api_key"):
            cfg.openai_api_key = v
