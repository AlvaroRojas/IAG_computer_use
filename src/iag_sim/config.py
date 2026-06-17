"""Runtime configuration. Loaded from environment / .env, validated at startup.

Secrets (OpenAI key, Murex password) live only in the environment — never in
source. Import `get_settings()` to fail fast if anything required is missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OpenAI
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    cua_model: str = Field(default="gpt-5.5", alias="CUA_MODEL")
    # Optional explicit override of the computer-use `environment`. If unset it
    # is derived per channel: web -> "browser", thick -> "ubuntu".
    cua_environment: str | None = Field(default=None, alias="CUA_ENVIRONMENT")

    # Murex
    murex_before_url: str = Field(alias="MUREX_BEFORE_URL")
    murex_after_url: str = Field(alias="MUREX_AFTER_URL")
    murex_user: str = Field(alias="MUREX_USER")
    murex_pass: SecretStr = Field(alias="MUREX_PASS")

    # Login group / desk / entity context the model selects after authenticating.
    # Scoped PER ENVIRONMENT (not per trade): a global default plus optional
    # before/after overrides, resolved by group_for(env) like channel_for/url_for.
    murex_login_group: str = Field(default="", alias="MUREX_LOGIN_GROUP")
    murex_before_group: str | None = Field(default=None, alias="MUREX_BEFORE_GROUP")
    murex_after_group: str | None = Field(default=None, alias="MUREX_AFTER_GROUP")
    # When true the computer-use model performs the login itself (types
    # MUREX_USER/MUREX_PASS) and selects the group, once per trade session, on
    # BOTH channels. Required for thick (its login cannot be scripted); optional
    # for web. WARNING: this places the credentials into the model context and
    # into every screenshot — it deliberately breaks the "credentials stay out of
    # LLM context" invariant. Leave false to keep the deterministic logins
    # (web: murex/login.py; thick: container entrypoint).
    murex_llm_login: bool = Field(default=False, alias="MUREX_LLM_LOGIN")

    # Access channel: "web" (Playwright/Chromium) or "thick" (Murex Java client
    # in a Linux Docker container). Set globally and/or override per environment.
    murex_channel: str = Field(default="web", alias="MUREX_CHANNEL")
    murex_before_channel: str | None = Field(default=None, alias="MUREX_BEFORE_CHANNEL")
    murex_after_channel: str | None = Field(default=None, alias="MUREX_AFTER_CHANNEL")

    # Thick-client (Docker) settings
    murex_docker_image: str | None = Field(default=None, alias="MUREX_DOCKER_IMAGE")
    murex_display: str = Field(default=":99", alias="MUREX_DISPLAY")
    murex_container_export_dir: str = Field(
        default="/exports", alias="MUREX_CONTAINER_EXPORT_DIR"
    )
    # Boot readiness is PROBED, not slept: new_session waits for the Murex client
    # window (WM_CLASS = container_login_window_class) to appear, then for the
    # screen to stop changing — returning as soon as the login/app screen is
    # painted. container_ready_secs is the HARD CAP on that wait (proceed anyway
    # on timeout), NOT a fixed sleep. Set 0 to skip the probe entirely (tests).
    container_ready_secs: int = Field(default=90, alias="MUREX_CONTAINER_READY_SECS", ge=0)
    container_ready_poll_secs: float = Field(
        default=2.0, alias="MUREX_CONTAINER_READY_POLL_SECS", gt=0
    )
    # How many consecutive identical screen signatures mark "done painting".
    container_ready_stable_polls: int = Field(
        default=2, alias="MUREX_CONTAINER_READY_STABLE_POLLS", ge=1
    )
    # WM_CLASS substring identifying the Murex client window. The window TITLE
    # carries the fileserver address (env-specific); the CLASS does not, so it is
    # the stable, environment-independent readiness key.
    container_login_window_class: str = Field(
        default="murex-rmi-loader", alias="MUREX_LOGIN_WINDOW_CLASS"
    )
    # Per-container resource caps (cgroup limits via `docker run --cpus/--memory`).
    # Empty string disables that flag. Defaults bound each trade's container to
    # 1 CPU / 512MB so N parallel containers stay predictable on the host.
    murex_docker_cpus: str = Field(default="1", alias="MUREX_DOCKER_CPUS")
    murex_docker_memory: str = Field(default="512m", alias="MUREX_DOCKER_MEMORY")
    docker_run_extra: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="MUREX_DOCKER_RUN_EXTRA"
    )

    # Run tuning
    max_concurrency: int = Field(default=4, alias="MAX_CONCURRENCY", ge=1, le=64)
    display_width: int = Field(default=1280, alias="DISPLAY_WIDTH")
    display_height: int = Field(default=800, alias="DISPLAY_HEIGHT")
    headless: bool = Field(default=True, alias="HEADLESS")
    # Web-channel (Chromium) memory cap. Playwright contexts SHARE one browser
    # process, so this is NOT a per-session cgroup like the thick container's
    # --memory: it is applied as a V8 heap cap (--max-old-space-size, in MB) on
    # launch, bounding EACH Chromium renderer process. CPU cannot be capped per
    # session through Playwright — use OS controls (cgroups / Job Objects) for a
    # hard CPU bound. None / 0 disables the cap.
    playwright_max_memory_mb: int | None = Field(
        default=512, alias="PLAYWRIGHT_MAX_MEMORY_MB", ge=0
    )
    max_turns: int = Field(default=60, alias="MAX_TURNS", ge=1, le=500)
    output_dir: Path = Field(default=Path("data/out"), alias="OUTPUT_DIR")

    # Diff
    # Delimiter of the RAW Murex export. The Mx.3 "Download as CSV" writes
    # SEMICOLON-separated values (confirmed from a live accounting-simulation
    # export), with the Comment column double-quoted around embedded ';'.
    csv_delimiter: str = Field(default=";", alias="CSV_DELIMITER")
    # NoDecode stops pydantic-settings from JSON-decoding the env value; the
    # before-validator below splits the comma-separated string instead.
    # Default = the real accounting-simulation posting key (trade_id is added by
    # aggregate.py; the rest are Murex column names). Confirmed against trade 594
    # (2 postings, distinct Rule nb) — re-verify on a multi-cashflow trade.
    diff_join_columns: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "trade_id", "Rule nb", "Value date", "Debit account",
            "Credit account", "Cur.",
        ],
        alias="DIFF_JOIN_COLUMNS",
    )
    diff_abs_tol: float = Field(default=0.01, alias="DIFF_ABS_TOL", ge=0)
    diff_rel_tol: float = Field(default=0.0, alias="DIFF_REL_TOL", ge=0)

    @field_validator("diff_join_columns", "docker_run_extra", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    @field_validator("murex_channel", "murex_before_channel", "murex_after_channel")
    @classmethod
    def _valid_channel(cls, v: str | None) -> str | None:
        if v is not None and v not in ("web", "thick"):
            raise ValueError(f"channel must be 'web' or 'thick', got {v!r}")
        return v

    def url_for(self, env: str) -> str:
        return self.murex_before_url if env == "before" else self.murex_after_url

    def channel_for(self, env: str) -> str:
        override = self.murex_before_channel if env == "before" else self.murex_after_channel
        return override or self.murex_channel

    def group_for(self, env: str) -> str:
        override = self.murex_before_group if env == "before" else self.murex_after_group
        return override or self.murex_login_group

    def cua_environment_for(self, env: str) -> str:
        if self.cua_environment:
            return self.cua_environment
        return "browser" if self.channel_for(env) == "web" else "ubuntu"

    def effective_concurrency(self) -> int:
        # Both channels parallelize (web: contexts, thick: containers).
        return self.max_concurrency


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Raises pydantic ValidationError if required
    secrets/URLs are missing — that is the intended fail-fast behaviour."""
    return Settings()  # type: ignore[call-arg]
