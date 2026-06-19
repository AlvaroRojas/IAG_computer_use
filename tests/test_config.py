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


def test_ignore_https_errors_defaults_true_and_override(monkeypatch):
    # On-prem Murex serves a self-signed cert, so the default must be permissive.
    assert _set(monkeypatch).murex_ignore_https_errors is True
    assert (
        _set(monkeypatch, MUREX_IGNORE_HTTPS_ERRORS="false").murex_ignore_https_errors
        is False
    )


def test_web_interaction_tuning_defaults_and_override(monkeypatch):
    # Web-SPA click delay + settle pause; non-zero defaults so the web channel
    # registers widget clicks and screenshots settled frames out of the box.
    s = _set(monkeypatch)
    assert s.cua_web_click_delay_ms == 60
    assert s.cua_web_settle_ms == 400
    o = _set(monkeypatch, CUA_WEB_CLICK_DELAY_MS="0", CUA_WEB_SETTLE_MS="250")
    assert o.cua_web_click_delay_ms == 0
    assert o.cua_web_settle_ms == 250


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


# --- Model provider selection + per-provider credential fail-fast ---


def test_provider_defaults_to_openai(monkeypatch):
    s = _set(monkeypatch)
    assert s.cua_provider == "openai"
    assert s.is_anthropic_provider() is False


def test_openai_provider_requires_openai_key(monkeypatch):
    for k in REQUIRED:
        monkeypatch.delenv(k, raising=False)
    # provide the non-secret required fields, omit OPENAI_API_KEY
    for k, v in {k: v for k, v in REQUIRED.items() if k != "OPENAI_API_KEY"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_anthropic_provider_requires_key(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_PROVIDER="anthropic")


def test_anthropic_provider_ok_with_key(monkeypatch):
    s = _set(monkeypatch, CUA_PROVIDER="anthropic", ANTHROPIC_API_KEY="ak-test")
    assert s.cua_provider == "anthropic"
    assert s.is_anthropic_provider() is True


def test_bedrock_provider_ok_with_token_and_region(monkeypatch):
    s = _set(
        monkeypatch, CUA_PROVIDER="bedrock",
        AWS_REGION="eu-west-1", AWS_BEARER_TOKEN_BEDROCK="ABSK-token",
    )
    assert s.cua_provider == "bedrock"
    assert s.is_anthropic_provider() is True


def test_bedrock_provider_missing_region_raises(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_PROVIDER="bedrock", AWS_BEARER_TOKEN_BEDROCK="ABSK-token")


def test_bedrock_provider_missing_token_raises(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_PROVIDER="bedrock", AWS_REGION="eu-west-1")


def test_bedrock_openai_provider_ok_with_token_and_base_url(monkeypatch):
    s = _set(
        monkeypatch, CUA_PROVIDER="bedrock-openai",
        AWS_BEARER_TOKEN_BEDROCK="ABSK-token", CUA_MODEL="openai.gpt-5.5",
        CUA_OPENAI_BASE_URL="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    assert s.cua_provider == "bedrock-openai"
    # OpenAI-shaped provider, NOT the Anthropic Messages family
    assert s.is_anthropic_provider() is False


def test_bedrock_openai_provider_missing_base_url_raises(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_PROVIDER="bedrock-openai", AWS_BEARER_TOKEN_BEDROCK="ABSK-token")


def test_bedrock_openai_provider_missing_token_raises(monkeypatch):
    with pytest.raises(ValidationError):
        _set(
            monkeypatch, CUA_PROVIDER="bedrock-openai",
            CUA_OPENAI_BASE_URL="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        )


def test_prompt_cache_retention_default_and_override(monkeypatch):
    s = _set(monkeypatch)
    assert s.cua_openai_prompt_cache_retention == "in_memory"
    s2 = _set(monkeypatch, CUA_OPENAI_PROMPT_CACHE_RETENTION="24h")
    assert s2.cua_openai_prompt_cache_retention == "24h"


def test_anthropic_tool_version_defaults_and_override(monkeypatch):
    s = _set(monkeypatch)
    assert s.cua_anthropic_tool_version == "computer_20251124"
    assert s.cua_anthropic_beta == "computer-use-2025-11-24"
    s2 = _set(
        monkeypatch,
        CUA_ANTHROPIC_TOOL_VERSION="computer_20250124",
        CUA_ANTHROPIC_BETA="computer-use-2025-01-24",
    )
    assert s2.cua_anthropic_tool_version == "computer_20250124"
    assert s2.cua_anthropic_beta == "computer-use-2025-01-24"


def test_invalid_provider_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_PROVIDER="gemini")


def test_reasoning_effort_defaults_none_and_override(monkeypatch):
    assert _set(monkeypatch).cua_reasoning_effort is None
    assert _set(monkeypatch, CUA_REASONING_EFFORT="high").cua_reasoning_effort == "high"


def test_invalid_reasoning_effort_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, CUA_REASONING_EFFORT="ludicrous")


def test_reasoning_effort_accepts_xhigh_and_max(monkeypatch):
    assert _set(monkeypatch, CUA_REASONING_EFFORT="xhigh").cua_reasoning_effort == "xhigh"
    # normalized to lowercase
    assert _set(monkeypatch, CUA_REASONING_EFFORT="MAX").cua_reasoning_effort == "max"


def test_prompt_cache_defaults_true_and_override(monkeypatch):
    assert _set(monkeypatch).cua_prompt_cache is True
    assert _set(monkeypatch, CUA_PROMPT_CACHE="false").cua_prompt_cache is False


def test_max_retries_defaults_and_override(monkeypatch):
    assert _set(monkeypatch).cua_max_retries == 8
    assert _set(monkeypatch, CUA_MAX_RETRIES="3").cua_max_retries == 3


# --- Export reality gate ---


def test_export_gate_defaults(monkeypatch):
    s = _set(monkeypatch)
    assert s.export_wait_secs == 20
    assert s.export_poll_secs == 0.5
    assert s.export_stable_polls == 2
    # Default 0: a zero-posting sim is a valid empty result (Murex still emits the header).
    assert s.export_min_rows == 0
    assert s.export_require_trade_id is True
    assert s.export_trade_id_columns == ["Trade nb", "Origin Trade nb"]


def test_export_gate_overrides(monkeypatch):
    s = _set(
        monkeypatch,
        EXPORT_WAIT_SECS="5",
        EXPORT_MIN_ROWS="2",
        EXPORT_REQUIRE_TRADE_ID="false",
        EXPORT_TRADE_ID_COLUMN="Trade nb, Origin Trade nb",
    )
    assert s.export_wait_secs == 5
    assert s.export_min_rows == 2
    assert s.export_require_trade_id is False
    # comma-separated -> list, whitespace trimmed (NoDecode + _split_csv)
    assert s.export_trade_id_columns == ["Trade nb", "Origin Trade nb"]


def test_export_poll_secs_must_be_positive(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, EXPORT_POLL_SECS="0")


def test_export_wait_secs_non_negative(monkeypatch):
    with pytest.raises(ValidationError):
        _set(monkeypatch, EXPORT_WAIT_SECS="-1")
