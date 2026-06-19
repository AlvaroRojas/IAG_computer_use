"""Runtime configuration. Loaded from environment / .env, validated at startup.

Secrets (model-provider keys, Murex password) live only in the environment —
never in source. Import `get_settings()` to fail fast if anything required is
missing. The computer-use loop is model-agnostic: `cua_provider` selects OpenAI
(Responses API), direct Anthropic, or Anthropic over AWS Bedrock.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Model provider (computer-use) ---
    # Which API drives the computer-use loop. "openai" = Responses API (default,
    # unchanged); "anthropic" = direct Anthropic Messages API; "bedrock" = the same
    # Messages API over AWS Bedrock; "bedrock-openai" = a GPT model on AWS Bedrock via
    # the OpenAI-compatible "mantle" Responses endpoint, which has NO native computer
    # tool, so computer-use is emulated with a custom function tool (see
    # cua/openai_custom_backend.py). The execution layer (Computer / dispatch /
    # harness) is provider-neutral; only the agent loop + client construction differ.
    cua_provider: Literal["openai", "anthropic", "bedrock", "bedrock-openai"] = Field(
        default="openai", alias="CUA_PROVIDER"
    )
    # The computer-use model id. Provider-specific:
    #   openai        -> e.g. gpt-5.5 / computer-use-preview
    #   anthropic     -> e.g. claude-opus-4-8
    #   bedrock       -> an inference-profile id, e.g. eu.anthropic.claude-opus-4-8
    #   bedrock-openai -> a Bedrock OpenAI model id, e.g. openai.gpt-5.5
    cua_model: str = Field(default="gpt-5.5", alias="CUA_MODEL")
    # Completion-token budget per turn. The Anthropic Messages API REQUIRES
    # max_tokens; ignored by the OpenAI Responses loop.
    cua_max_tokens: int = Field(default=4096, alias="CUA_MAX_TOKENS", ge=1)
    # Reasoning effort applied to EVERY provider. Unset -> provider default (nothing
    # sent). OpenAI -> reasoning.effort; Anthropic/Bedrock -> output_config.effort
    # (newest models, via adaptive thinking) or extended-thinking budget_tokens
    # (older models). Availability is model-specific and the API validates it:
    # `none`/`minimal` are OpenAI-only; `xhigh` = newer OpenAI + Claude Opus only;
    # `max` = Claude Sonnet 4.6+/Opus. Validated to a known set here to catch typos.
    cua_reasoning_effort: str | None = Field(
        default=None, alias="CUA_REASONING_EFFORT"
    )

    # OpenAI credentials (required only when cua_provider == "openai").
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    # Optional explicit override of the OpenAI computer-use `environment`. If unset
    # it is derived per channel: web -> "browser", thick -> "ubuntu". OpenAI-only.
    cua_environment: str | None = Field(default=None, alias="CUA_ENVIRONMENT")
    # OpenAI-compatible Responses base URL. Used (and REQUIRED) only by the
    # "bedrock-openai" provider, where it points at the Bedrock mantle endpoint, e.g.
    # https://bedrock-mantle.us-east-2.api.aws/openai/v1 . The OpenAI SDK appends
    # "/responses"; auth is the Bedrock bearer token (AWS_BEARER_TOKEN_BEDROCK).
    cua_openai_base_url: str | None = Field(default=None, alias="CUA_OPENAI_BASE_URL")
    # Prompt-cache retention for the bedrock-openai provider. GPT-5.5 prompt caching is
    # OFF unless this is sent (the "borked GPT-5 caching" gotcha): "in_memory" gave
    # consistent ~90% hits on a stable resent prefix; "24h" flaked; automatic /
    # previous_response_id / prompt_cache_key alone all returned cached=0. Empty/unset =
    # no caching. Works WITH pruning: a stubbed screenshot stays a (byte-identical) stub
    # forever, so the task preamble + already-pruned history form a stable cacheable
    # prefix billed at 0.1x; only the last CUA_KEEP_LAST_SCREENSHOTS real images + the
    # new one fall past the divergence and bill full. Only the bedrock-openai backend
    # uses this.
    cua_openai_prompt_cache_retention: str | None = Field(
        default="in_memory", alias="CUA_OPENAI_PROMPT_CACHE_RETENTION"
    )

    # Direct Anthropic credentials (required only when cua_provider == "anthropic").
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # AWS Bedrock (required only when cua_provider == "bedrock"). Auth is via a
    # Bedrock API key (bearer token) in AWS_BEARER_TOKEN_BEDROCK — AsyncAnthropicBedrock
    # sends it as `Authorization: Bearer` and skips SigV4, so no AWS access key/secret
    # is needed. aws_region selects the endpoint (e.g. eu-west-1).
    aws_region: str | None = Field(default=None, alias="AWS_REGION")
    aws_bearer_token_bedrock: SecretStr | None = Field(
        default=None, alias="AWS_BEARER_TOKEN_BEDROCK"
    )

    # Anthropic computer-use tool generation. Defaults target the newest models
    # (Opus 4.8/4.7/4.6, Sonnet 4.6). Override the pair for older models — e.g.
    # Bedrock Sonnet 4.5 needs computer_20250124 / computer-use-2025-01-24.
    cua_anthropic_tool_version: str = Field(
        default="computer_20251124", alias="CUA_ANTHROPIC_TOOL_VERSION"
    )
    cua_anthropic_beta: str = Field(
        default="computer-use-2025-11-24", alias="CUA_ANTHROPIC_BETA"
    )
    # How many of the most recent screenshots to keep in the Anthropic message
    # history (older image blocks are pruned to bound token cost). 0 = keep all.
    cua_keep_last_screenshots: int = Field(
        default=3, alias="CUA_KEEP_LAST_SCREENSHOTS", ge=0
    )
    # Insert Anthropic prompt-cache breakpoints (the tool def + a rolling anchor at
    # the screenshot-prune frontier) so the byte-stable prefix bills at the 0.1x
    # cache-read rate on every turn after the first. 5-min ephemeral cache; no beta
    # header needed (caching is GA on both direct Anthropic and Bedrock). OpenAI
    # caches automatically, so this flag only affects anthropic/bedrock. Raising
    # CUA_KEEP_LAST_SCREENSHOTS enlarges the cacheable prefix (less pruning churn).
    cua_prompt_cache: bool = Field(default=True, alias="CUA_PROMPT_CACHE")
    # Max automatic retries the provider SDK makes per request, with exponential
    # backoff. Covers transient 429/5xx — notably Bedrock's 503 "unable to process
    # your request" (on-demand capacity/throttle dips, common on Opus profiles).
    # Retrying at the SDK layer rides out a capacity blip WITHOUT losing in-session
    # progress (the coarse worker-level retry re-boots the whole trade session).
    # SDK default is 2; bumped here since computer-use sessions are long (N x 2 x
    # up to MAX_TURNS calls) so any single 503 shouldn't sink a trade.
    cua_max_retries: int = Field(default=8, alias="CUA_MAX_RETRIES", ge=0)

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
    # Web channel only: tell Chromium to ignore TLS certificate errors. On-prem
    # Murex web servers typically present a self-signed / internal-CA certificate,
    # so a fresh Playwright context aborts navigation with ERR_CERT_AUTHORITY_INVALID.
    # Defaults true because that is the norm for the internal hosts this tool targets;
    # set false when pointing at a server with a publicly trusted certificate. No
    # effect on the thick channel.
    murex_ignore_https_errors: bool = Field(
        default=True, alias="MUREX_IGNORE_HTTPS_ERRORS"
    )
    # Web-channel interaction fidelity (no effect on the thick channel, which
    # drives Java Swing via xdotool and repaints synchronously). The MX.3 web SPA
    # animates menus/dropdowns and uses custom comboboxes, so coordinate clicks
    # need a small mousedown->up gap to register, and the page needs a beat to
    # repaint before the agent loop screenshots — else the model sees a stale /
    # mid-transition frame and mis-clicks (observed: dropdown selections not
    # taking, typed text not landing). Both in milliseconds; 0 disables.
    cua_web_click_delay_ms: int = Field(
        default=60, alias="CUA_WEB_CLICK_DELAY_MS", ge=0
    )
    cua_web_settle_ms: int = Field(default=400, alias="CUA_WEB_SETTLE_MS", ge=0)

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
    # Graceful Murex logout on container teardown. The backend caps concurrent
    # sessions PER USER and reaps a session only when its client disconnects; a
    # hard `docker stop` (SIGKILL after the grace window) can leave the session
    # lingering server-side, so rapid retries/relaunches pile up and trip the cap.
    # `docker stop -t <secs>` gives a SIGTERM-trapping client time to log out and
    # disconnect cleanly before SIGKILL.
    container_stop_timeout_secs: int = Field(
        default=10, alias="MUREX_CONTAINER_STOP_TIMEOUT_SECS", ge=0
    )
    # Optional shell command run inside the container (`docker exec sh -c`) just
    # before stop, to trigger a clean logout/disconnect when the client does NOT
    # exit gracefully on SIGTERM (e.g. the Murex JVM is not PID 1: "pkill -TERM
    # java", or an image-provided logout script). Empty = skip. Best-effort and
    # time-bounded — never blocks teardown.
    container_logout_cmd: str = Field(default="", alias="MUREX_CONTAINER_LOGOUT_CMD")
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
    # Turn budget for the second, logout-only computer-use phase that runs AFTER
    # the export has been validated. Kept small — clean log-off is a few clicks;
    # it only releases the Murex session so the next trade doesn't hit the
    # per-user session cap, and it is best-effort (failure never changes the
    # WorkerResult). 0 disables the logout phase entirely.
    logoff_max_turns: int = Field(default=8, alias="LOGOFF_MAX_TURNS", ge=0, le=60)
    # Max seconds the model should wait for the accounting-simulation postings table
    # to populate after clicking 'Proceed' before treating a STILL-EMPTY table as a
    # zero-posting result and exporting the header-only CSV. Injected into the task
    # prompt (the model is the only observer of the table — Python never sees it, so
    # this is guidance, not a deterministic timer). The model decomposes it into a
    # few ~15s 'wait' actions (cua/actions.py caps one wait at 15s). Set GENEROUSLY:
    # too short and a slow-but-real sim gets exported header-only and is trusted as a
    # FALSE empty; if rows appear sooner the model exports immediately, so a large
    # value only costs turns on genuinely empty sims. Re-tune against real Murex
    # accounting-sim latency before the first real run.
    sim_result_wait_secs: int = Field(default=45, alias="SIM_RESULT_WAIT_SECS", ge=0)
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

    # Export reality gate — proves the CSV the model "exported" is a real, non-empty,
    # parseable file for THIS trade before ok=True (else the worker retries it). The
    # model's "DONE" text is never trusted; only a validated on-disk artifact passes.
    # Max time to wait for the export to APPEAR after the loop ends (the model's last
    # action may trigger the download/file just as the loop returns). 0 = collect once.
    export_wait_secs: int = Field(default=20, alias="EXPORT_WAIT_SECS", ge=0)
    # Poll interval while waiting for the file to appear / its size to settle.
    export_poll_secs: float = Field(default=0.5, alias="EXPORT_POLL_SECS", gt=0)
    # Thick channel only: consecutive identical file sizes that mark "done writing"
    # (the bind-mounted file can be observed mid-write). Web downloads complete via
    # Playwright save_as, so this does not apply there.
    export_stable_polls: int = Field(default=2, alias="EXPORT_STABLE_POLLS", ge=1)
    # Minimum DATA rows (excluding the header) a valid export must contain.
    # Default 0: a zero-posting accounting simulation is a LEGITIMATE result —
    # Murex still exports the header row, so a header-only CSV is a trusted empty
    # outcome (empty-before vs empty-after => MATCH; empty vs non-empty => a real
    # present/missing difference). Set to 1 only when every trade must have postings.
    export_min_rows: int = Field(default=0, alias="EXPORT_MIN_ROWS", ge=0)
    # Enforce that every posting row references THIS trade (catches the model querying
    # or exporting the wrong trade). Disable only if a real export omits the columns.
    export_require_trade_id: bool = Field(default=True, alias="EXPORT_REQUIRE_TRADE_ID")
    # Comma-separated list of export columns that may carry the queried trade ref. A
    # row passes if its id matches ANY of these columns: the queried id lands in
    # "Trade nb" for a normal trade, but in "Origin Trade nb" for an origin/novated
    # trade whose postings are booked under a different (resolved) "Trade nb". Matching
    # any one of them keeps the anti-hallucination guarantee (a wrong export matches
    # none) without rejecting legitimate origin trades. NoDecode + _split_csv as for
    # DIFF_JOIN_COLUMNS. Re-verify the column names against a real export.
    export_trade_id_columns: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["Trade nb", "Origin Trade nb"],
        alias="EXPORT_TRADE_ID_COLUMN",
    )

    @field_validator(
        "diff_join_columns", "docker_run_extra", "export_trade_id_columns", mode="before"
    )
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

    @field_validator("cua_reasoning_effort", mode="before")
    @classmethod
    def _valid_effort(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        s = str(v).strip().lower()
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        if s not in allowed:
            raise ValueError(
                f"CUA_REASONING_EFFORT must be one of {sorted(allowed)} or unset"
            )
        return s

    @model_validator(mode="after")
    def _require_provider_credentials(self) -> "Settings":
        """Fail fast if the credentials for the selected provider are missing."""
        if self.cua_provider == "openai" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when CUA_PROVIDER=openai")
        if self.cua_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when CUA_PROVIDER=anthropic")
        if self.cua_provider == "bedrock":
            if self.aws_bearer_token_bedrock is None:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK is required when CUA_PROVIDER=bedrock"
                )
            if not self.aws_region:
                raise ValueError("AWS_REGION is required when CUA_PROVIDER=bedrock")
        if self.cua_provider == "bedrock-openai":
            if self.aws_bearer_token_bedrock is None:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK is required when CUA_PROVIDER=bedrock-openai"
                )
            if not self.cua_openai_base_url:
                raise ValueError(
                    "CUA_OPENAI_BASE_URL is required when CUA_PROVIDER=bedrock-openai"
                )
        return self

    def is_anthropic_provider(self) -> bool:
        """True for the Anthropic Messages-API providers (direct or Bedrock)."""
        return self.cua_provider in ("anthropic", "bedrock")

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
