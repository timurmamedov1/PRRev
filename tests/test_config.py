# tests for config loading and precedence

import os
from unittest.mock import patch

import pytest

from prrev import config as config_mod
from prrev.config import Config, _apply_toml, load_config

# vars load_config reads from the env, cleared before each test so the
# developers real shell env (a live GITHUB_TOKEN etc) cant leak into asserts
_CONFIG_ENV_VARS = (
    "PRREV_PROVIDER",
    "PRREV_MODEL",
    "PRREV_MAX_ITEMS",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    # clean env every test, otherwise a real GITHUB_TOKEN in the shell overrides
    # repo config and masks the token-blocking guarantee were trying to verify
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # dont read the real ~/.config/prrev/config.toml, it may hold live tokens
    monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", tmp_path / "no-global.toml")


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


class TestInvalidConfig:
    def test_non_integer_env_max_items_raises(self):
        env = {"PRREV_MAX_ITEMS": "abc"}
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(ValueError, match="PRREV_MAX_ITEMS"),
        ):
            load_config()

    def test_non_integer_toml_max_items_raises(self, tmp_path):
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text('[review]\nmax_items = "lots"\n')
        with pytest.raises(ValueError, match="max_items"):
            load_config(repo_path=str(tmp_path))

    def test_zero_toml_max_items_rejected(self, tmp_path):
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text("[review]\nmax_items = 0\n")
        with pytest.raises(ValueError, match="positive"):
            load_config(repo_path=str(tmp_path))

    def test_zero_env_max_items_rejected(self):
        env = {"PRREV_MAX_ITEMS": "0"}
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(ValueError, match="positive"),
        ):
            load_config()

    def test_empty_provider_surfaces_instead_of_falling_back(self, tmp_path):
        # empty string reaches the provider check and errors there, rather
        # than silently reverting to the default provider
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text('[llm]\nprovider = ""\n')
        cfg = load_config(repo_path=str(tmp_path))
        assert cfg.provider == ""

    def test_malformed_toml_raises(self, tmp_path):
        # TOMLDecodeError subclasses ValueError so the cli catch covers it
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text("not [ valid toml")
        with pytest.raises(ValueError):
            load_config(repo_path=str(tmp_path))


class TestPrecedence:
    def test_env_overrides_repo_config(self, tmp_path):
        (tmp_path / ".prrev.toml").write_text('[llm]\nprovider = "openai"\n')
        with patch.dict(os.environ, {"PRREV_PROVIDER": "anthropic"}, clear=False):
            cfg = load_config(repo_path=str(tmp_path))
        assert cfg.provider == "anthropic"

    def test_global_config_loads_tokens_and_settings(self, tmp_path, monkeypatch):
        global_toml = tmp_path / "global.toml"
        global_toml.write_text('[github]\ntoken = "ghp_global"\n[llm]\nprovider = "openai"\n')
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", global_toml)
        cfg = load_config()
        assert cfg.github_token == "ghp_global"
        assert cfg.provider == "openai"

    def test_repo_config_overrides_global(self, tmp_path, monkeypatch):
        global_toml = tmp_path / "global.toml"
        global_toml.write_text('[llm]\nprovider = "openai"\n')
        monkeypatch.setattr(config_mod, "GLOBAL_CONFIG_PATH", global_toml)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".prrev.toml").write_text('[llm]\nprovider = "anthropic"\n')
        cfg = load_config(repo_path=str(repo))
        assert cfg.provider == "anthropic"


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
        # security guarantee: repo config can never inject any token, even if
        # someone drops a malicious .prrev.toml into a repo you clone
        toml_file = tmp_path / ".prrev.toml"
        toml_file.write_text(
            '[github]\ntoken = "ghp_leaked"\n'
            '[llm]\nanthropic_api_key = "sk-ant-leaked"\n'
            'openai_api_key = "sk-oai-leaked"\n'
        )
        cfg = load_config(repo_path=str(tmp_path))
        assert cfg.github_token is None
        assert cfg.anthropic_api_key is None
        assert cfg.openai_api_key is None
