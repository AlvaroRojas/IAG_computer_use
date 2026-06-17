"""Config is a system boundary — verify env parsing, the CSV-list field, and
fail-fast on missing secrets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iag_sim.config import Settings

REQUIRED = {
    "OPENAI_API_KEY": "sk-test",
    "MUREX_BEFORE_URL": "https://before",
    "MUREX_AFTER_URL": "https://after",
    "MUREX_USER": "u",
    "MUREX_PASS": "p",
}


def _set(monkeypatch, **extra):
    for k, v in {**REQUIRED, **extra}.items():
        monkeypatch.setenv(k, v)
    # ignore any real .env on disk during the test
    return Settings(_env_file=None)


def test_join_columns_parsed_from_csv_string(monkeypatch):
    s = _set(monkeypatch, DIFF_JOIN_COLUMNS="trade_id,gl_account,currency")
    assert s.diff_join_columns == ["trade_id", "gl_account", "currency"]


def test_url_for_selects_environment(monkeypatch):
    s = _set(monkeypatch)
    assert s.url_for("before") == "https://before"
    assert s.url_for("after") == "https://after"


def test_secrets_not_plaintext_in_repr(monkeypatch):
    s = _set(monkeypatch)
    assert "sk-test" not in repr(s)


def test_missing_required_raises(monkeypatch):
    for k in REQUIRED:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_channel_defaults_to_web(monkeypatch):
    s = _set(monkeypatch)
    assert s.channel_for("before") == "web"
    assert s.channel_for("after") == "web"


def test_channel_global_override(monkeypatch):
    s = _set(monkeypatch, MUREX_CHANNEL="thick")
    assert s.channel_for("before") == "thick"
    assert s.channel_for("after") == "thick"


def test_channel_per_env_override_wins(monkeypatch):
    # mixed: before on the web UI, after on the thick client
    s = _set(monkeypatch, MUREX_CHANNEL="web", MUREX_AFTER_CHANNEL="thick")
    assert s.channel_for("before") == "web"
    assert s.channel_for("after") == "thick"


def test_invalid_channel_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, MUREX_CHANNEL="citrix")


def test_cua_environment_derived_from_channel(monkeypatch):
    s = _set(monkeypatch, MUREX_BEFORE_CHANNEL="web", MUREX_AFTER_CHANNEL="thick")
    assert s.cua_environment_for("before") == "browser"
    assert s.cua_environment_for("after") == "ubuntu"


def test_cua_environment_explicit_override(monkeypatch):
    s = _set(monkeypatch, MUREX_CHANNEL="thick", CUA_ENVIRONMENT="windows")
    assert s.cua_environment_for("before") == "windows"


def test_docker_run_extra_parsed_from_csv(monkeypatch):
    s = _set(monkeypatch, MUREX_DOCKER_RUN_EXTRA="--shm-size=2g,--network=host")
    assert s.docker_run_extra == ["--shm-size=2g", "--network=host"]


def test_docker_resource_caps_default_1cpu_512m(monkeypatch):
    s = _set(monkeypatch)
    assert s.murex_docker_cpus == "1"
    assert s.murex_docker_memory == "512m"


def test_docker_resource_caps_overridable(monkeypatch):
    s = _set(monkeypatch, MUREX_DOCKER_CPUS="2", MUREX_DOCKER_MEMORY="1g")
    assert s.murex_docker_cpus == "2"
    assert s.murex_docker_memory == "1g"


def test_playwright_memory_cap_default(monkeypatch):
    s = _set(monkeypatch)
    assert s.playwright_max_memory_mb == 512


def test_login_group_defaults_empty(monkeypatch):
    s = _set(monkeypatch)
    assert s.group_for("before") == ""
    assert s.group_for("after") == ""
    assert s.murex_llm_login is False


def test_login_group_global_applies_to_both(monkeypatch):
    s = _set(monkeypatch, MUREX_LOGIN_GROUP="ACCT_DESK_01")
    assert s.group_for("before") == "ACCT_DESK_01"
    assert s.group_for("after") == "ACCT_DESK_01"


def test_login_group_per_env_override_wins(monkeypatch):
    s = _set(
        monkeypatch,
        MUREX_LOGIN_GROUP="GLOBAL",
        MUREX_AFTER_GROUP="AFTER_DESK",
    )
    assert s.group_for("before") == "GLOBAL"
    assert s.group_for("after") == "AFTER_DESK"
