# IAG Accounting Simulation — Before/After Comparator

Validates that a Murex configuration change does **not** unintentionally alter
accounting output. For *N* trades, it runs the **accounting simulation** in a
**before-changes** and an **after-changes** Murex environment, exports each
result to CSV, aggregates into two CSVs, and **diffs them deterministically**.

Murex is driven by OpenAI **computer-use** (Responses API `computer` tool)
through one of two interchangeable **channels**:

| Channel | Driver | Parallelism | `environment` |
|---------|--------|-------------|---------------|
| **web** | Murex **web UI** via Playwright (Chromium) | N browser contexts | `browser` |
| **thick** | Murex **Java client** in a Linux **Docker** container | N containers (one per trade) | `ubuntu` |

Both are first-class. Pick globally with `MUREX_CHANNEL`, or per environment
(`MUREX_BEFORE_CHANNEL` / `MUREX_AFTER_CHANNEL`) — you can even diff a web
"before" against a thick "after". The model only operates the UI; the
comparison is plain, reproducible Python (`pandas` + `datacompy`). **The LLM
never computes the diff.**

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
| `src/iag_sim/cua/actions.py` | action dict → `Computer` call (pure) |
| `src/iag_sim/cua/computer.py` | `PlaywrightComputer` (browser + downloads) |
| `src/iag_sim/cua/loop.py` | the computer-use agent loop (Responses API) |
| `src/iag_sim/harness/base.py` | `Harness` / `TradeSession` abstraction (channel seam) |
| `src/iag_sim/harness/browser.py` | web channel: Playwright contexts |
| `src/iag_sim/harness/docker.py` | thick channel: `DockerComputer` + `DockerHarness` |
| `src/iag_sim/murex/login.py` | web channel per-env login → saved `storage_state` |
| `src/iag_sim/murex/simulate.py` | run one trade's simulation → CSV path |
| `src/iag_sim/aggregate.py` | concat per-trade CSVs (pure core) |
| `src/iag_sim/diff.py` | datacompy before/after compare (pure) |
| `src/iag_sim/orchestration/` | resources, worker (+retry), runner, graph, postprocess |
| `src/iag_sim/cli.py` | `iag-sim run` |

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m playwright install chromium   # web channel only
Copy-Item .env.example .env   # then fill in real values
```

`.env` (never commit — gitignored). Always:
- `OPENAI_API_KEY`, `CUA_MODEL` (a computer-use-capable model, e.g. `gpt-5.5`)
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

1. **CSV schema → diff key** (both channels) — open a real exported CSV and set
   `DIFF_JOIN_COLUMNS` (columns that uniquely identify a posting) and
   `DIFF_ABS_TOL`. Defaults (`trade_id,gl_account,currency`, `0.01`) are a guess.
2. **Web channel: login selectors** — `src/iag_sim/murex/login.py` has
   **placeholder** selectors (`USERNAME_SELECTOR`, `PASSWORD_SELECTOR`,
   `SUBMIT_SELECTOR`, `LOGGED_IN_SELECTOR`). Walk the real login page once and
   set them, or drive login via computer-use too.
3. **Thick channel: container image** — build/supply `MUREX_DOCKER_IMAGE`
   honouring the contract above and tune `MUREX_CONTAINER_READY_SECS` to the
   client's real startup + login time.

Also verify `CUA_MODEL` is a computer-use-capable model available on your key
(live contract: https://developers.openai.com/api/docs/guides/tools-computer-use).

### Cost / reliability note
Computer-use screenshots are token-heavy; `N × 2` sessions add up. For stable
steps (web: login/navigation/export click) prefer recorded **deterministic
Playwright actions** and reserve vision for the genuinely variable in-screen
work. Bound `MAX_CONCURRENCY` to stay under OpenAI TPM/RPM and (thick channel)
host CPU/RAM — each container is a full desktop.
