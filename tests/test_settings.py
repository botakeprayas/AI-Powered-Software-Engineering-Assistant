import pytest

from core.settings import PROJECT_ROOT, load_settings


def test_defaults_when_nothing_is_set(monkeypatch):
    for key in ("APP_ENV", "LOG_LEVEL", "DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(env_file=None)
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.sample_repo_dir == settings.data_dir / "sample_repo"


def test_environment_variable_overrides_default(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert load_settings(env_file=None).log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError):
        load_settings(env_file=None)