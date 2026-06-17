# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **before/after accounting comparator** for Murex. For *N* trades it runs the
Murex *accounting simulation* in a **before-changes** and an **after-changes**
environment, exports each result to CSV, aggregates, and **diffs them
deterministically** to prove a config change did not alter accounting output.

Murex is driven by the OpenAI **computer-use** tool (Responses API `computer`)
over one of two interchangeable **channels** (see Architecture). Read `README.md`
for the user-facing overview; this file is the working contract for editing.

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

### Computer-use loop (`cua/`)

`cua/loop.py` implements the Responses API `computer` tool contract (computer_call →
execute actions → screenshot back via `computer_call_output` → carry
`previous_response_id` until no computer_call returns). `cua/actions.py` is the pure
action-dict → `Computer` dispatch; `cua/base.py` is the dependency-free `Computer`
protocol. Safety checks are acknowledged through an injected `on_safety_check`
callback (default: deny).

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

The scaffold is complete and tested, but three things depend on the **actual Murex
UI** and are currently placeholders/guesses:
1. **Diff key** — open a real exported CSV; set `DIFF_JOIN_COLUMNS` (columns that
   uniquely identify a posting) and `DIFF_ABS_TOL`. Defaults
   (`trade_id,gl_account,currency`, `0.01`) are a guess.
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

Also confirm `CUA_MODEL` is a computer-use-capable model on your key.

## Config

All runtime config flows through `config.py` (`Settings`, loaded from env/`.env`).
`.env` is gitignored — see `.env.example` for the full annotated list. Cost note:
computer-use screenshots are token-heavy and there are `N × 2` sessions; bound
`MAX_CONCURRENCY` to stay under OpenAI TPM/RPM and (thick) host CPU/RAM.
