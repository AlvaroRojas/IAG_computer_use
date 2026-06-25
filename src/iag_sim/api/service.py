"""Translation between the API request and the engine.

Three pure helpers:
- `build_settings_from_request` — turn a `RunRequest` into a per-request `Settings`,
  passing ONLY the payload fields as alias kwargs (provider creds, OUTPUT_DIR, diff
  tuning, etc. fall back to the server's env/.env).
- `mint_run_id` — new run-dir folder name (UTC), same format as the CLI.
- `derive_result_code` — map the engine summary dict to MATCH/DIFFERENCES/NO_COMPARISON.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError

from ..config import Settings
from .schemas import ResultCode, RunRequest


def build_settings_from_request(req: RunRequest) -> Settings:
    """Construct a per-request `Settings`. Only the payload's Murex/engine fields
    are passed (by their UPPERCASE alias); every other field — CUA_PROVIDER, the
    provider API keys, OUTPUT_DIR, MUREX_DOCKER_IMAGE, diff/export tuning — falls
    back to the server's own environment / `.env`.

    Thick channel cannot script its login, so MUREX_LLM_LOGIN is forced True when
    either resolved channel is "thick" (never forced False — a server-side default
    still wins for web)."""
    kw: dict[str, object] = {
        "MUREX_BEFORE_URL": req.murex_before_url,
        "MUREX_AFTER_URL": req.murex_after_url,
        "MUREX_USER": req.murex_user,
        "MUREX_PASS": req.murex_pass,
        "MUREX_LOGIN_GROUP": req.murex_login_group,
        "MUREX_CHANNEL": req.murex_channel,
        "MAX_CONCURRENCY": req.max_concurrency,
    }
    # Optional per-env overrides: only pass when set, so an unset override keeps
    # the Settings fallback (group_for / channel_for resolve the shared default).
    for alias, val in (
        ("MUREX_BEFORE_GROUP", req.murex_before_group),
        ("MUREX_AFTER_GROUP", req.murex_after_group),
        ("MUREX_BEFORE_CHANNEL", req.murex_before_channel),
        ("MUREX_AFTER_CHANNEL", req.murex_after_channel),
    ):
        if val is not None:
            kw[alias] = val

    before_ch = req.murex_before_channel or req.murex_channel
    after_ch = req.murex_after_channel or req.murex_channel
    if "thick" in (before_ch, after_ch):
        kw["MUREX_LLM_LOGIN"] = True

    try:
        return Settings(**kw)  # type: ignore[arg-type]
    except ValidationError as exc:
        # Missing provider creds, bad channel value, etc. -> 422 with field detail.
        # Drop context (carries non-JSON ValueError objects) and input (may carry
        # the MUREX_PASS value) so the detail is JSON-safe and leaks nothing.
        raise HTTPException(
            status_code=422,
            detail=exc.errors(
                include_url=False, include_context=False, include_input=False
            ),
        )


def mint_run_id() -> str:
    """New run-dir folder name, e.g. run-20260617-170647 (UTC). Same format as
    `cli._run_id` so the API and CLI mint identical run ids."""
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def derive_result_code(summary: dict) -> ResultCode:
    """Map the engine summary to a result code, the same way the CLI derives its
    exit code: no diff -> NO_COMPARISON; diff.matches -> MATCH; else DIFFERENCES."""
    diff = summary.get("diff")
    if diff is None:
        return ResultCode.NO_COMPARISON
    if diff.get("matches"):
        return ResultCode.MATCH
    return ResultCode.DIFFERENCES
