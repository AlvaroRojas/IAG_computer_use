# IAG Accounting Simulation — Before/After Comparator

Validates that a Murex configuration change does **not** unintentionally alter
accounting output. For *N* trades, it runs the **accounting simulation** in a
**before-changes** and an **after-changes** Murex environment, exports each
result to CSV, aggregates into two CSVs, and **diffs them deterministically**.

Murex is driven by a **computer-use model** — OpenAI (Responses API `computer`
tool) or Anthropic / AWS Bedrock (Messages API `computer` tool), chosen with
`CUA_PROVIDER` — through one of two interchangeable **channels**:

| Channel | Driver | Parallelism | `environment` |
|---------|--------|-------------|---------------|
| **web** | Murex **web UI** via Playwright (Chromium) | N browser contexts | `browser` |
| **thick** | Murex **Java client** in a Linux **Docker** container | N containers (one per trade) | `ubuntu` |

Both are first-class. Pick globally with `MUREX_CHANNEL`, or per environment
(`MUREX_BEFORE_CHANNEL` / `MUREX_AFTER_CHANNEL`) — you can even diff a web
"before" against a thick "after". The model only operates the UI; the
comparison is plain, reproducible Python (`pandas` + `datacompy`). **The LLM
never computes the diff.**

The `environment` column is OpenAI's `computer` tool hint; the Anthropic backend
ignores it. The provider is orthogonal to the channel — any provider drives any
channel through the same `Computer` interface (provider seam: `cua/backend.py`).

## Architecture

```
trades.csv
   │
   ▼  fan-out: one worker per (trade × {before, after}), bounded by MAX_CONCURRENCY
 worker ──▶ Harness.new_session(trade) ──▶ a Computer + CSV-export collector
   │          ├─ web channel:   Playwright context (reuse per-env login) → Murex URL
   │          └─ thick channel: `docker run` a Murex-client container (Xvfb + xdotool)
   │          computer-use loop: model emits actions → Computer executes
   │          → screenshot back → … → CSV exported
   │          reality gate: real parseable CSV for THIS trade (else retry) ◀─ not the model's word
   │          (a header-only CSV = a trusted zero-posting result, not a failure)
   ▼
 aggregate (pandas) ──▶ before_aggregated.csv / after_aggregated.csv
   ▼
 diff (datacompy, deterministic) ──▶ comparison/{report.txt, mismatches.csv, summary.json}
```

The computer-use loop and the diff are **channel-agnostic**: a `Harness`
abstraction supplies the right `Computer` (Playwright page actions, or
`docker exec` + `xdotool`/`import`) and the CSV-export collection. Adding a
channel means adding a harness, nothing else changes.

Two engines, same work unit:
- **`async`** (default) — `asyncio.gather` + semaphore. Simple, observable.
- **`langgraph`** — `StateGraph` + Send-API fan-out + checkpointer (resumable).

## Module map

| Path | Responsibility |
|------|----------------|
| `src/iag_sim/config.py` | env/.env settings, channel resolution, validated at startup |
| `src/iag_sim/models.py` | `TradeTask`, `WorkerResult`, `EnvName` |
| `src/iag_sim/cua/base.py` | `Computer` protocol (no deps) |
| `src/iag_sim/cua/actions.py` | canonical action dict → `Computer` call (pure, shared) |
| `src/iag_sim/cua/computer.py` | `PlaywrightComputer` (browser + downloads) |
| `src/iag_sim/cua/backend.py` | provider seam: `AgentBackend` + `build_backend` |
| `src/iag_sim/cua/loop.py` + `openai_backend.py` | OpenAI computer-use loop (Responses API) |
| `src/iag_sim/cua/anthropic_backend.py` | Anthropic/Bedrock loop (Messages API) |
| `src/iag_sim/cua/openai_custom_backend.py` | GPT-on-Bedrock loop: custom-tool computer-use emulation |
| `src/iag_sim/cua/anthropic_actions.py` | Anthropic action → canonical translator (pure) |
| `src/iag_sim/harness/base.py` | `Harness` / `TradeSession` abstraction (channel seam) |
| `src/iag_sim/harness/browser.py` | web channel: Playwright contexts |
| `src/iag_sim/harness/docker.py` | thick channel: `DockerComputer` + `DockerHarness` |
| `src/iag_sim/murex/login.py` | web channel per-env login → saved `storage_state` |
| `src/iag_sim/murex/simulate.py` | run one trade's simulation → CSV path |
| `src/iag_sim/murex/export_validate.py` | reality gate: validate the exported CSV (pure) |
| `src/iag_sim/aggregate.py` | concat per-trade CSVs (pure core) |
| `src/iag_sim/diff.py` | datacompy before/after compare (pure) |
| `src/iag_sim/orchestration/` | resources, worker (+retry), runner, graph, postprocess |
| `src/iag_sim/cli.py` | `iag-sim run` |

## Prerequisites (deployment machine)

Host requirements depend on which **channel(s)** you run. The channel is the only
thing that changes the machine footprint — the provider (OpenAI / Anthropic /
Bedrock) is just an API client and adds no host dependency beyond credentials.

| Run | Host needs | Does **not** need |
|-----|------------|-------------------|
| **web** | Python 3.12, `playwright install chromium` | Docker |
| **thick** | Python 3.12, a running **Docker daemon** (Desktop/Engine) + socket access, a built/supplied `MUREX_DOCKER_IMAGE` | Chromium |
| **mixed** (web one env, thick the other) | both of the above | — |
| **tests** | Python 3.12 only | browser, Docker, network |

Python **3.12** is a hard floor. The thick channel shells out to the `docker`
CLI, so the daemon must be running and reachable by the user that launches the
run (one `docker run -d --rm` container per trade). The web channel launches one
shared Chromium with N contexts — no Docker.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m playwright install chromium   # web channel only
Copy-Item .env.example .env   # then fill in real values
```

`.env` (never commit — gitignored). Always:
- `CUA_PROVIDER` (`openai` | `anthropic` | `bedrock` | `bedrock-openai`, default
  `openai`) + `CUA_MODEL` (a computer-use-capable model for that provider, e.g.
  `gpt-5.5`, `claude-opus-4-8`, a Bedrock profile `eu.anthropic.claude-opus-4-8`, or
  a Bedrock OpenAI id `openai.gpt-5.5`)
- provider credentials: `OPENAI_API_KEY` (openai) **or** `ANTHROPIC_API_KEY`
  (anthropic) **or** `AWS_REGION` + `AWS_BEARER_TOKEN_BEDROCK` (bedrock; a Bedrock
  API key — no AWS access key/secret needed) **or** `AWS_BEARER_TOKEN_BEDROCK` +
  `CUA_OPENAI_BASE_URL` (`bedrock-openai`; GPT on Bedrock via the mantle Responses
  endpoint — computer-use is *emulated* with a custom function tool, no native tool
  there, so grounding is weaker than Opus 4.8)
- optional `CUA_REASONING_EFFORT` (`none|minimal|low|medium|high|xhigh|max`, model-
  dependent — the API validates) — applied to **any** provider; unset = provider
  default. OpenAI → `reasoning.effort`; Anthropic/Bedrock → adaptive thinking +
  `output_config.effort` (newest models) or `budget_tokens` (older). `xhigh` = Opus
  only; `max` = Sonnet 4.6+/Opus. Raise `CUA_MAX_TOKENS` for high+ effort.
- `MUREX_BEFORE_URL`, `MUREX_AFTER_URL`, `MUREX_USER`, `MUREX_PASS`
- `MUREX_CHANNEL` (`web` | `thick`), optional `MUREX_BEFORE_CHANNEL` / `MUREX_AFTER_CHANNEL`
- `MAX_CONCURRENCY`, `DISPLAY_WIDTH/HEIGHT`, `MAX_TURNS`, `DIFF_JOIN_COLUMNS`, `DIFF_ABS_TOL`

Web channel also uses `HEADLESS`. Thick channel also needs:
- `MUREX_DOCKER_IMAGE` (required), `MUREX_DISPLAY`, `MUREX_CONTAINER_EXPORT_DIR`,
  `MUREX_CONTAINER_READY_SECS`, optional `MUREX_DOCKER_RUN_EXTRA`

### Thick-client container contract

For the thick channel the supplied image (`MUREX_DOCKER_IMAGE`) must:
- boot an X server on `$DISPLAY` (default `:99`) at `$SCREEN_GEOMETRY`;
- launch the Murex Java client and **log in** using `$MUREX_USER` / `$MUREX_PASS`
  against `$MUREX_ENV_TARGET`, leaving the app ready. **Credentials are handled
  by the image entrypoint — never typed by the model**, so they stay out of LLM
  context;
- ship ImageMagick (`import`) and `xdotool`;
- write CSV exports into `$MUREX_CONTAINER_EXPORT_DIR` (host-mounted per trade).

The harness does `docker run -d --rm` per trade with those env vars + the export
volume, drives it via `docker exec <c> sh -c "export DISPLAY=… && …"`
(screenshots via `import -window root png:-`, input via `xdotool`), and
`docker stop`s it when the trade completes. This matches OpenAI's documented
Docker action handlers.

## Run

```powershell
.\.venv\Scripts\python.exe -m iag_sim run --trades data/trades.csv
.\.venv\Scripts\python.exe -m iag_sim run --trades data/trades.csv --engine langgraph
.\.venv\Scripts\python.exe -m iag_sim run --headed --max-concurrency 2   # debug (web)
```

Exit codes: `0` = before/after match, `2` = differences found, `1` = no
comparison produced (a side yielded no data).

Artifacts land in `data/out/run-<timestamp>/`: per-trade CSVs, the two
aggregates, and `comparison/`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Deterministic, no browser/Docker/network needed: action dispatch, aggregation,
the audit-critical diff (identical→match, delta→flagged, within-tolerance→match,
row-only-in-one-env→flagged), config + channel resolution, `DockerComputer`
command construction (via a fake runner), and LangGraph wiring.

## ⚠️ Before the first real run — required tuning

This scaffold is complete and tested, but a few things depend on the **actual
Murex UI** and must be confirmed once (per the plan's pre-build steps):

1. **CSV schema → diff key + export gate** (both channels) — open a real exported
   CSV and set `DIFF_JOIN_COLUMNS` (columns that uniquely identify a posting) and
   `DIFF_ABS_TOL` (defaults `trade_id,gl_account,currency`, `0.01` are a guess). On
   the same file confirm `CSV_DELIMITER` and `EXPORT_TRADE_ID_COLUMN` — a
   comma-separated list of columns the reality gate matches each row against, passing
   if any matches (default `Trade nb,Origin Trade nb`: a normal trade carries the id
   in `Trade nb`, an origin/novated trade in `Origin Trade nb`).
   A zero-posting simulation is treated as a valid empty result (header-only CSV);
   set `EXPORT_MIN_ROWS=1` only if you want every trade to require postings.
2. **Web channel: login selectors** — `src/iag_sim/murex/login.py` has
   **placeholder** selectors (`USERNAME_SELECTOR`, `PASSWORD_SELECTOR`,
   `SUBMIT_SELECTOR`, `LOGGED_IN_SELECTOR`). Walk the real login page once and
   set them, or drive login via computer-use too.
3. **Thick channel: container image** — build/supply `MUREX_DOCKER_IMAGE`
   honouring the contract above and tune `MUREX_CONTAINER_READY_SECS` to the
   client's real startup + login time.

Also verify `CUA_MODEL` is a computer-use-capable model for your `CUA_PROVIDER`
and available on your key/account (OpenAI:
https://developers.openai.com/api/docs/guides/tools-computer-use; Anthropic/Bedrock:
the `computer_20251124` tool + `computer-use-2025-11-24` beta — for older Bedrock
models set `CUA_ANTHROPIC_TOOL_VERSION` / `CUA_ANTHROPIC_BETA`).

### Cost / reliability note
Computer-use screenshots are token-heavy; `N × 2` sessions add up. For stable
steps (web: login/navigation/export click) prefer recorded **deterministic
Playwright actions** and reserve vision for the genuinely variable in-screen
work. Bound `MAX_CONCURRENCY` to stay under your provider's rate limits (OpenAI
TPM/RPM, Anthropic/Bedrock token+request caps) and (thick channel) host CPU/RAM —
each container is a full desktop.

**Prompt caching** is on by default to make long sessions cheaper. OpenAI caches
the prefix automatically (the Responses loop chains turns with
`previous_response_id`); Anthropic/Bedrock insert `cache_control` breakpoints (a
static one on the tool def + a rolling one anchored at the screenshot-prune
frontier) so the byte-stable prefix bills at the 0.1× cache-read rate after the
first turn. Disable with `CUA_PROMPT_CACHE=false`; per-turn `cache_read` /
`cache_write` token counts are emitted to the live trace. Raising
`CUA_KEEP_LAST_SCREENSHOTS` enlarges the cacheable prefix.
