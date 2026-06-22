# tests for config loading and precedence

import os
from pathlib import Path
from unittest.mock import patch

from prrev.config import Config, load_config, _apply_toml


class TestDefaults:
    def test_default_provider(self):
        cfg = Config()
        assert cfg.provider == "anthropic"

    def test_default_max_items(self):
        cfg = Config()
        assert cfg.max_items == 20

    def test_default_tokens_are_none(self):
        cfg = Config()
        assert cfg.github_token is None
        assert cfg.anthropic_api_key is None
        assert cfg.openai_api_key is None


class TestApplyToml:
    def test_sets_provider(self):
        cfg = Config()
        _apply_toml(cfg, {"llm": {"provider": "openai"}}, allow_tokens=False)
        assert cfg.provider == "openai"

    def test_sets_model(self):
        cfg = Config()
        _apply_toml(cfg, {"llm": {"model": "gpt-4o-mini"}}, allow_tokens=False)
        assert cfg.model == "gpt-4o-mini"

    def test_sets_max_items(self):
        cfg = Config()
        _apply_toml(cfg, {"review": {"max_items": 5}}, allow_tokens=False)
        assert cfg.max_items == 5

    def test_tokens_allowed(self):
        cfg = Config()
        data = {
            "github": {"token": "ghp_abc"},
            "llm": {"anthropic_api_key": "sk-ant", "openai_api_key": "sk-oai"},
        }
        _apply_toml(cfg, data, allow_tokens=True)
        assert cfg.github_token == "ghp_abc"
        assert cfg.anthropic_api_key == "sk-ant"
        assert cfg.openai_api_key == "sk-oai"

    def test_tokens_blocked_from_repo_config(self):
        cfg = Config()
        data = {
            "github": {"token": "ghp_abc"},
            "llm": {"anthropic_api_key": "sk-ant"},
        }
        _apply_toml(cfg, data, allow_tokens=False)
        assert cfg.github_token is None
        assert cfg.anthropic_api_key is None


class TestEnvVarOverride:
    def test_env_overrides_provider(self):
        env = {"PRREV_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.provider == "openai"

    def test_env_overrides_model(self):
        env = {"PRREV_MODEL": "gpt-4o-mini"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.model == "gpt-4o-mini"

    def test_env_overrides_max_items(self):
        env = {"PRREV_MAX_ITEMS": "5"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.max_items == 5

    def test_env_sets_api_keys(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENAI_API_KEY": "sk-oai-test",
            "GITHUB_TOKEN": "ghp_test",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.anthropic_api_key == "sk-ant-test"
        assert cfg.openai_api_key == "sk-oai-test"
        assert cfg.github_token == "ghp_test"


class TestLoadConfig:
    def test_missing_toml_returns_defaults(self):
        cfg = load_config(repo_path="/nonexistent/path")
        assert cfg.provider == "anthropic"
        assert cfg.max_items == 20

    def test_repo_toml_loaded(self, tmp_path):
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text('[llm]\nprovider = "openai"\n')
        cfg = load_config(repo_path=str(tmp_path))
        assert cfg.provider == "openai"

    def test_repo_toml_blocks_tokens(self, tmp_path):
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text('[github]\ntoken = "ghp_leaked"\n')
        cfg = load_config(repo_path=str(tmp_path))
        assert cfg.github_token is None
