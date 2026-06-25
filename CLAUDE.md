# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **before/after accounting comparator** for Murex. For *N* trades it runs the
Murex *accounting simulation* in a **before-changes** and an **after-changes**
environment, exports each result to CSV, aggregates, and **diffs them
deterministically** to prove a config change did not alter accounting output.

Murex is driven by a **computer-use model** — OpenAI (Responses API `computer`)
or Anthropic / AWS Bedrock (Messages API `computer` tool), selected by
`CUA_PROVIDER` behind a provider-neutral **backend seam** (`cua/backend.py`) — over
one of two interchangeable **channels** (see Architecture). Read `README.md` for
the user-facing overview; this file is the working contract for editing.

## Commands

Windows / PowerShell. Always invoke the venv interpreter explicitly
(`.\.venv\Scripts\python.exe`) — there is no activated shell assumption.

```powershell
# Setup
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m playwright install chromium   # web channel only
Copy-Item .env.example .env                                 # then fill real values

# Run (both engines share the same work unit)
.\.venv\Scripts\python.exe -m iag_sim run --trades data/trades.csv
.\.venv\Scripts\python.exe -m iag_sim run --trades data/trades.csv --engine langgraph
.\.venv\Scripts\python.exe -m iag_sim run --headed --max-concurrency 2   # web debug

# Serve the run API (langgraph engine over REST) — needs $env:IAG_SIM_API_KEY
.\.venv\Scripts\python.exe -m iag_sim.server                # or: iag-sim-api  (:8000)

# Tests (deterministic — no browser, Docker, or network needed)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_diff.py -q          # one file
.\.venv\Scripts\python.exe -m pytest tests/test_diff.py::test_identical_frames_match -q   # one test
.\.venv\Scripts\python.exe -m pytest -k aggregate -q               # by keyword
```

`iag-sim` is also installed as a console script (`pyproject` `[project.scripts]`),
equivalent to `python -m iag_sim`.

**Run exit codes** (`cli._cmd_run`): `0` = before/after match, `2` = differences
found, `1` = no comparison produced (a side yielded no data). Artifacts land in
`data/out/run-<UTC timestamp>/`.

## Architecture

The whole pipeline is one work unit fanned out two ways:

```
trades.csv → fan-out one worker per (trade × {before, after}), bounded by MAX_CONCURRENCY
  worker → Harness.new_session(trade) → a Computer + CSV-export collector
         → computer-use loop drives the UI → CSV exported
  → aggregate per-env (pandas) → before_aggregated.csv / after_aggregated.csv
  → diff (datacompy, deterministic) → comparison/{report.txt, mismatches.csv, summary.json}
```

### The two seams that matter

**1. Channel seam (`harness/base.py`).** A `Harness` = one access channel bound to
one environment; it mints a `TradeSession` (a `Computer` + display size + CSV
collector) per trade. Two implementations, same `Computer` interface so the agent
loop is identical:
- `harness/browser.py` — **web** channel: Murex web UI via Playwright Chromium
  contexts. `cua` environment = `browser`.
- `harness/docker.py` — **thick** channel: Murex Java client in a Linux Docker
  container, one container per trade, driven by `docker exec … xdotool` / ImageMagick
  `import`. `cua` environment = `ubuntu`.

Adding a channel = add a harness; nothing else changes. Channel is resolved per
environment in `config.py` (`channel_for`), so you can diff a web "before" against
a thick "after".

**2. Engine seam.** Both orchestrators wrap the *same* `orchestration/worker.py`
(`run_worker`), so the unit of work is identical:
- `orchestration/runner.py` — default **async** engine: `asyncio.gather` + semaphore.
- `orchestration/graph.py` — **langgraph** engine: `StateGraph` + Send-API fan-out +
  **durable `AsyncSqliteSaver`** checkpointer at `<run_dir>/checkpoints.sqlite`
  (`thread_id` = run-dir name). Resume a crashed run with
  `iag-sim run --resume <run_dir>`: completed workers replay from the on-disk
  checkpoint, only unfinished ones re-run (resources + Murex login always
  re-established). `build_graph` falls back to in-memory `MemorySaver` when no
  checkpointer is passed (tests). Needs `langgraph-checkpoint-sqlite`.

### HTTP API surface (`api/`, `server.py`)

A thin FastAPI layer (`iag-sim-api` console script) drives the **langgraph** engine
over REST without changing it — it just builds a **per-request `Settings`** and
calls `run_graph_async`. This works because `get_settings()`' `lru_cache` singleton
is used ONLY by the CLI; every engine/orchestration layer takes `settings` as a
parameter, so a fresh `Settings(**alias_kwargs)` threads through with zero global
bleed. The request body's `MUREX_*` + `MAX_CONCURRENCY` keys are the `Settings`
field aliases verbatim (`api/service.py::build_settings_from_request`), so they
override those fields for that run while provider creds / `OUTPUT_DIR` / diff tuning
fall back to the server's env/`.env`. `POST /runs` (empty `run_id` mints a new
`run-<UTC>` dir; non-empty resumes that folder's checkpoint) launches the run as a
detached `asyncio.create_task` and returns `202` + the run id; `GET /runs/{id}`
polls. `api/run_manager.py` enforces **one run at a time** (an `asyncio.Lock` guards
the busy-check + slot claim; a second `POST` gets `409`) and keeps an in-memory
status registry — the durable on-disk checkpoint is the recovery story across a
process restart (re-`POST` the run id). Auth is a shared `X-API-Key` vs
`IAG_SIM_API_KEY` (`api/security.py`); `thick` channel forces `MUREX_LLM_LOGIN=true`.
The status `result_code` (`MATCH`/`DIFFERENCES`/`NO_COMPARISON`) mirrors the CLI exit
codes `0`/`2`/`1`. Never call `asyncio.run` inside the API loop — `run_graph_async`
is awaited directly and opens its own resources + sqlite saver.

### Computer-use loop (`cua/`)

**3. Provider seam (`cua/backend.py`).** An `AgentBackend` owns the model client +
model id and runs the loop against a `Computer`; `simulate.py` calls `backend.run(...)`
so the unit of work is identical across providers. `build_backend(settings)` picks
the impl from `CUA_PROVIDER` (lazy per-provider client imports). Three impls, same
`Computer`:
- `cua/loop.py` + `cua/openai_backend.py` — OpenAI **Responses API** `computer`
  contract (computer_call → execute → screenshot back via `computer_call_output` →
  carry `previous_response_id` until no computer_call returns). Safety checks via an
  injected `on_safety_check` callback (default: deny).
- `cua/anthropic_backend.py` — Anthropic **Messages API** loop (direct Anthropic OR
  AWS Bedrock — the client differs, the loop is identical: `AsyncAnthropic` vs
  `AsyncAnthropicBedrock`, the latter authed by a Bedrock API key in
  `AWS_BEARER_TOKEN_BEDROCK`). Stateless: a growing `messages` list (no
  `previous_response_id`). Each `tool_use` action is translated by
  `cua/anthropic_actions.py` into the SAME canonical action dicts and run through
  `cua/actions.py`; the post-action screenshot is fed back as a `tool_result` image.
  Assistant content (incl. thinking blocks) is echoed back verbatim; older
  screenshots are pruned (`CUA_KEEP_LAST_SCREENSHOTS`) to bound tokens. Tool
  generation defaults to `computer_20251124` / beta `computer-use-2025-11-24`
  (override via `CUA_ANTHROPIC_TOOL_VERSION` / `CUA_ANTHROPIC_BETA` for older models).
- `cua/openai_custom_backend.py` — **custom-tool computer-use emulation** for GPT on
  AWS Bedrock (`CUA_PROVIDER=bedrock-openai`): the mantle Responses endpoint
  (`CUA_OPENAI_BASE_URL`, authed by the SAME `AWS_BEARER_TOKEN_BEDROCK` bearer)
  exposes **no native `computer` tool**, so this declares ONE custom function tool
  (`computer`, the canonical action vocab) and feeds screenshots as `input_image`.
  Stateless like the Anthropic loop (no `previous_response_id`): output items are
  echoed back, the post-action screenshot rides a fresh `user` message because a
  Responses `function_call_output` is text-only, and older screenshots are pruned
  (`CUA_KEEP_LAST_SCREENSHOTS`). **Caveat:** GPT is not GUI-grounding-fine-tuned, so
  expect more mis-clicks on Murex's dense UI than Opus 4.8 / `computer-use-preview`.

`cua/actions.py` is the pure action-dict → `Computer` dispatch shared by ALL loops;
`cua/base.py` is the dependency-free `Computer` protocol. The canonical action
vocabulary is OpenAI-shaped; Anthropic actions (xdotool keysyms, `coordinate` pairs,
`scroll_direction`/`scroll_amount`) are normalized to it in `anthropic_actions.py`.

**Reasoning effort** is one knob across providers: `CUA_REASONING_EFFORT`
(`none|minimal|low|medium|high|xhigh|max`, unset = provider default; the API
validates per-model availability) maps to OpenAI `reasoning.effort` and, on
Anthropic/Bedrock, to **adaptive thinking** (`thinking:{type:"adaptive"}`) + a
top-level `output_config.effort` on the newest gen (tool `computer_20251124`;
manual `budget_tokens` is a 400 on Opus 4.8/4.7), falling back to manual
`budget_tokens` for older models (tool `computer_20250124`, e.g. Sonnet 4.5). **Scroll** deltas are canonical **wheel notches**:
the thick channel consumes them 1:1 (`xdotool click --repeat`), the browser scales
notches→pixels (`_WHEEL_PX_PER_NOTCH`) — no per-channel env knob.

**Prompt caching** (`CUA_PROMPT_CACHE`, default on) is automatic for OpenAI (the
Responses loop chains `previous_response_id`, so the prefix is cached server-side —
no code) and `cache_control`-driven for Anthropic/Bedrock (stateless Messages API).
The latter sets a static breakpoint on the tool def + ONE rolling breakpoint
anchored at the **screenshot-prune frontier** (`_cache_anchor`): pruning mutates an
old image→stub roughly every turn, so anchoring at the newest message would pay the
1.25× cache-write penalty with no read; anchoring strictly before the oldest still-
real screenshot keeps the cached prefix byte-stable across turns. Never put
`cache_control` on assistant blocks (SDK objects echoed verbatim — mutating them
breaks the thinking-block signature contract). Per-turn `cache_read`/`cache_write`
counts go to the trace (`usage` event).

## Invariants — do not break these

- **The LLM never computes the diff.** It only operates the UI. Aggregation
  (`aggregate.py`) and comparison (`diff.py`, datacompy) are pure, reproducible
  Python suitable for audit. Never route the comparison through the model.
- **Credentials stay out of LLM context** *(default; opt-out exists)*. Thick channel:
  the container entrypoint logs in from `$MUREX_USER`/`$MUREX_PASS` — the model never
  types them. Web channel: login is deterministic Playwright in `murex/login.py`. The
  task prompt assumes the app is already logged in. **Escape hatch — `MUREX_LLM_LOGIN=true`:**
  the model logs in (types `$MUREX_USER`/`$MUREX_PASS`) and selects the login group itself,
  **once per trade session, on both channels**. The login *action* is per-session (sessions
  stay cold/parallel — one container per trade, one web context per trade); the *config*
  (`group_for(env)` via `MUREX_LOGIN_GROUP[_BEFORE|_AFTER]`, and the creds) is per-environment,
  never per-trade. `simulate.py` injects creds+group into the task prompt; `browser.py` skips
  the pre-auth so each context lands on the login page; `docker.py` omits the creds env so the
  container boots to the login screen. This deliberately exposes credentials to the model
  context and screenshots — only enable when a deterministic login can't be used (thick login
  cannot be scripted, so thick requires this mode).
- **`simulate_trade` never raises for automation failures** — it returns
  `WorkerResult(ok=False, error=…)` so the orchestrator can record/retry.
  `worker.py` retries on `not result.ok` (tenacity, 3 attempts, exponential backoff).
- **An export is trusted only when it's a real artifact, never the model's word.**
  The model's "DONE" text is never a success signal — success requires a CSV on disk
  that passes the reality gate (`murex/export_validate.py`): exists, non-empty, parses
  with `CSV_DELIMITER`, has ≥ `EXPORT_MIN_ROWS` rows, and every row matches the trade id
  in AT LEAST ONE of the `EXPORT_TRADE_ID_COLUMN` columns (a comma-separated list, default
  `Trade nb,Origin Trade nb`: a normal trade carries the id in `Trade nb`, an origin/novated
  trade in `Origin Trade nb` while `Trade nb` holds the resolved trade — matching any one
  avoids rejecting legitimate origin trades while still catching a wrong export). **A zero-posting
  simulation is a LEGITIMATE result, not a failure:** `EXPORT_MIN_ROWS` defaults to `0`,
  so a header-only CSV (parses, has the trade-id column, 0 data rows) passes as a trusted
  *empty* export (`ExportCheck.empty` → `WorkerResult.empty=True`). `postprocess.py`'s
  coverage ledger then proves empty-before vs empty-after a MATCH and surfaces
  empty-vs-non-empty as a present/missing difference (datacompy `rows_only_*`). Set
  `EXPORT_MIN_ROWS=1` to require postings on every trade. `collect_export`
  first WAITS up to `EXPORT_WAIT_SECS` for the file/download to appear (web: the
  Playwright download event; thick: the bind-mounted file, size-stable for
  `EXPORT_STABLE_POLLS` polls), since the model's last action can fire it just as the
  loop returns. A failed check returns `ok=False` and rides the SAME tenacity retry —
  no separate retry path. Thick reuses one per-trade export dir across attempts, so
  `new_session` clears stale `*.csv` first. Don't reintroduce trusting `final_text`.
- **Runtime resources are non-serializable and passed via closures** (`resources.py`),
  never placed in checkpointed graph state. Playwright launches only if some env uses
  the web channel.
- **A dead process never orphans a container.** `DockerHarness` tracks every container
  it starts (`self._containers` + module-level `_LIVE_CONTAINERS`). `aclose()` (registered
  in `open_resources`' `AsyncExitStack`, so it runs on normal exit, exception, OR Ctrl+C
  cancellation) shielded-stops all tracked containers; an `atexit` hook synchronously
  force-stops any remainder if the interpreter exits abnormally. A container is untracked
  only AFTER a successful `docker stop`, so a failed/cancelled stop is retried. The only
  uncoverable case is SIGKILL / `taskkill /F` (nothing can intercept it). Don't make
  `aclose` a no-op again, and don't untrack before the stop confirms.
- **Domain models (`models.py`) stay small and JSON-serializable** so they flow
  through async results and LangGraph state.
- **Config validates at startup, fail-fast** (`get_settings()`, pydantic-settings).
  Secrets are `SecretStr`; missing required values raise `ValidationError`. Comma-
  separated env lists (`DIFF_JOIN_COLUMNS`, `MUREX_DOCKER_RUN_EXTRA`) use `NoDecode` +
  the `_split_csv` validator — do not switch them to JSON decoding.

## Before the first real run — required tuning

The scaffold is complete and tested, but four things depend on the **actual Murex
UI** and are currently placeholders/guesses:
1. **Diff key + export gate columns** — open a real exported CSV; set
   `DIFF_JOIN_COLUMNS` (columns that uniquely identify a posting) and `DIFF_ABS_TOL`
   (defaults `trade_id,gl_account,currency`, `0.01` are a guess). On the same CSV,
   confirm `CSV_DELIMITER` and the `EXPORT_TRADE_ID_COLUMN` columns (comma-separated,
   default `Trade nb,Origin Trade nb`) carrying the trade reference — the reality gate
   matches every row against them, passing if any one matches.
   A zero-posting sim is treated as a valid empty export (`EXPORT_MIN_ROWS=0`); confirm
   Murex still emits the header row for an empty result, and set `EXPORT_MIN_ROWS=1`
   only if every trade must have postings.
2. **Web login** — deterministic mode: set the placeholder selectors in
   `murex/login.py` against the real login page. LLM-login mode (`MUREX_LLM_LOGIN=true`):
   no selectors needed — the model logs in; tune the login wording in `simulate.py`
   (`_PREAMBLE_LLM_LOGIN`) and set the per-env group (`MUREX_LOGIN_GROUP[_BEFORE|_AFTER]`).
3. **Thick container image** — build/supply `MUREX_DOCKER_IMAGE` honouring the
   contract in `harness/docker.py` (boot X on `$DISPLAY` at `$SCREEN_GEOMETRY`,
   launch the client, ship `import` + `xdotool`, export to
   `$MUREX_CONTAINER_EXPORT_DIR`). Boot readiness is **probed**, not slept
   (`DockerHarness._wait_ready`): it waits for the client window
   (`WM_CLASS` ⊇ `MUREX_LOGIN_WINDOW_CLASS`, default `murex-rmi-loader`) then for
   the screen to settle, capped by `MUREX_CONTAINER_READY_SECS` (a timeout, not a
   fixed wait). If your image's window class differs, set `MUREX_LOGIN_WINDOW_CLASS`.
   Thick login cannot be scripted → run with `MUREX_LLM_LOGIN=true` so the image
   boots to the login screen and the model logs in + picks the group.
4. **Sim-result wait** — `SIM_RESULT_WAIT_SECS` (default `45`) is the max the model
   waits after 'Proceed' for the postings table to populate before treating a
   still-empty table as a zero-posting result and exporting the header-only CSV. It
   is prompt guidance (`_GOAL` step 4 in `simulate.py`), not a Python timer — the
   model is the only observer of the table. Tune to real accounting-sim latency: too
   short exports a slow-but-real sim as a FALSE empty (the reality gate trusts it);
   rows appearing sooner export immediately, so a generous value only costs turns on
   genuinely empty sims. Sites that never expect empties can hard-fail via `EXPORT_MIN_ROWS=1`.

Also confirm `CUA_MODEL` is a computer-use-capable model on your key.

## Config

All runtime config flows through `config.py` (`Settings`, loaded from env/`.env`).
`.env` is gitignored — see `.env.example` for the full annotated list. Cost note:
computer-use screenshots are token-heavy and there are `N × 2` sessions; bound
`MAX_CONCURRENCY` to stay under OpenAI TPM/RPM and (thick) host CPU/RAM.
