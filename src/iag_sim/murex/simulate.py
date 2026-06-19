"""Run the Murex accounting simulation for one trade in one environment and
return the exported CSV path. Channel-agnostic: the `Harness` supplies a
`Computer` (web via Playwright, or thick client via Docker/xdotool) and the CSV
export collection; the computer-use loop drives whichever one it is.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..cua.backend import AgentBackend
from ..cua.trace import Tracer
from ..harness.base import Harness
from ..models import TradeTask, WorkerResult
from .export_validate import validate_export

# Shared goal — appended after whichever login preamble applies. The navigation
# is the exact path verified against the live Mx.3 UI (trade 594), so the model
# does not have to discover it: Trade query -> filter by Trade ID -> Search ->
# right-click the row -> Financial information -> Accounting simulation ->
# Proceed -> File -> Download as CSV. Do NOT filter by Financial contract ID.
_GOAL = (
    "Goal: export the ACCOUNTING SIMULATION postings for trade {trade_id}{extra} "
    "to CSV. Do not change, save, or validate any trade data. Steps:\n"
    "1. Open 'Trade query' (Processing / Trades, or the home History list) "
    "ONCE. Each open consumes a Murex session and there is a per-user cap (~6), "
    "so do NOT click 'Trade query' again if a window is already open. If a dialog "
    "warns you are 'not allowed to open more than 6 sessions' (or any session-limit "
    "message), click OK/Close to dismiss it, issue a single 'wait' action (~15 "
    "seconds) to let stale sessions reap, then REUSE the Trade query window that is "
    "already open instead of opening a new one.\n"
    "2. In the top-left filter field, change 'Financial contract ID' to "
    "'Trade ID' (filter by TRADE, not contract), type {trade_id}, click Search.\n"
    "3. Right-click the trade row -> hover 'Financial information' -> click "
    "'Accounting simulation'.\n"
    "4. In the Accounting simulation dialog, click 'Proceed' and wait for the "
    "postings table (Value date, Rule nb, Debit account, Credit account, "
    "Amount, ...) to populate. As soon as rows appear, continue. If the table "
    "stays EMPTY, wait at most ~{sim_wait_secs} seconds (a few 'wait' actions of "
    "~15s each) — a zero-posting simulation is a VALID result, so once that time "
    "has passed with the table still empty, STOP waiting and proceed to export "
    "the (header-only) CSV anyway.\n"
    "5. Click any cell in the postings table, then the 'File' menu (top-left) "
    "-> 'Download as CSV'. {export_hint}\n"
    "When the CSV has been saved, reply with the single word DONE. Do NOT log off "
    "yet — leave Murex open; a separate step handles log-off after the file is "
    "verified."
)

# Second, logout-only computer-use phase. Run AFTER the export has been collected
# and the reality gate has passed (or failed) — never before, so the model can't
# log off before the file is confirmed saved. Releases the Murex session cleanly
# so the next sequential trade doesn't hit the per-user session cap (~6). Driven
# as its own short backend.run on the SAME live session; best-effort (a failed
# log-off does not change the already-decided WorkerResult).
_LOGOFF_TASK = (
    "The CSV export for this trade is already saved and verified. Your ONLY "
    "remaining task is to SIGN OUT of Murex cleanly so the session is released "
    "for the next trade. Steps:\n"
    "1. In the TOP-RIGHT corner of the Mx.3 window or web client interface, click the person / user "
    "icon (a small silhouette on a grey/dark background, to the right of the info 'i' and help "
    "'?' icons). A small menu opens (Change Group/Desk, Settings, Save "
    "screenset, Lock screen, Sign Out).\n"
    "2. Click 'Sign Out' (the bottom item).\n"
    "Then in the confirmation alert click yes to sign out, after a couple of seconds murex screen will dissapear and after a few more seconds will go back to login.\n"
    "No murex window or a login murex screen means you have signed out."
    "Do NOT open 'Trade query', run another simulation, or export anything. When "
    "you have signed out (or the application has closed), reply with the single "
    "word DONE."
)

# Where the exported CSV must land so the harness can collect it.
# Thick channel: a File chooser opens (defaults to /opt/murex) -> the model MUST
# type the shared export-dir path or collect_export() never finds the file.
# Web channel: the browser download dir is what collect_export() globs, so a
# plain download is enough.
_EXPORT_HINT_THICK = (
    "When the Save / File chooser dialog opens: click the File Name field, "
    "select all (Ctrl+A) and type the FULL path "
    "'{export_dir}/accounting_{trade_id}.csv', then press ENTER to confirm. "
    "Do NOT look for or click a 'Save' button — the chooser confirms on Enter, "
    "and hunting for the button causes misclicks. The file MUST land in "
    "'{export_dir}/' (the shared export folder), not the default directory."
)
_EXPORT_HINT_WEB = (
    "Use the browser download to save the CSV; the default downloads folder is "
    "correct — do not change it."
)

# Default path: deterministic login already happened; app is authenticated.
_PREAMBLE_LOGGED_IN = (
    "You are operating Murex (the {env} environment) for the accounting team. "
    "The application is open and you are logged in. "
)
# Appended to the logged-in preamble when a group is configured but auth is
# deterministic — just make sure the right group/context is active.
_PREAMBLE_GROUP_ONLY = "Make sure the login group '{group}' is selected. "

# Opt-in path (MUREX_LLM_LOGIN): the model logs in and picks the group itself.
# NOTE: this intentionally exposes the credentials to the model + screenshots.
_PREAMBLE_LLM_LOGIN = (
    "You are operating Murex (the {env} environment) for the accounting team. "
    "The application is open at the LOGIN screen. First LOG IN with username "
    "'{user}' and password '{password}'. After authenticating, select the login "
    "group '{group}' (the desk / entity context) from the group selector. "
)

# Back-compat alias for the original single-template name.
TASK_TEMPLATE = _PREAMBLE_LOGGED_IN + _GOAL


def _build_task(trade: TradeTask, env: str, settings: Settings) -> str:
    extra = ""
    if trade.extra:
        pairs = ", ".join(f"{k}={v}" for k, v in trade.extra.items())
        extra = f" (additional identifiers: {pairs})"

    group = settings.group_for(env)
    if settings.murex_llm_login:
        preamble = _PREAMBLE_LLM_LOGIN.format(
            env=env,
            user=settings.murex_user,
            password=settings.murex_pass.get_secret_value(),
            group=group or "(MUREX_LOGIN_GROUP not set)",
        )
    else:
        preamble = _PREAMBLE_LOGGED_IN.format(env=env)
        if group:
            preamble += _PREAMBLE_GROUP_ONLY.format(group=group)

    if settings.channel_for(env) == "thick":
        export_hint = _EXPORT_HINT_THICK.format(
            export_dir=settings.murex_container_export_dir, trade_id=trade.trade_id
        )
    else:
        export_hint = _EXPORT_HINT_WEB

    return preamble + _GOAL.format(
        trade_id=trade.trade_id,
        extra=extra,
        export_hint=export_hint,
        sim_wait_secs=settings.sim_result_wait_secs,
    )


async def _attempt_logoff(
    *,
    backend: AgentBackend,
    session,
    settings: Settings,
    env,
    tracer: Tracer,
) -> None:
    """Best-effort clean Murex log-off, run as a SECOND short computer-use phase
    on the same live session AFTER the export has been collected and validated.
    Releases the Murex session so the next sequential trade doesn't hit the
    per-user session cap. Never raises: the file is already trusted (or already
    failed and headed for retry), so a failed log-off must not change the
    WorkerResult. Disabled when LOGOFF_MAX_TURNS=0."""
    if settings.logoff_max_turns <= 0:
        return
    try:
        width, height = session.display
        await backend.run(
            computer=session.computer,
            task=_LOGOFF_TASK,
            display_width=width,
            display_height=height,
            environment=settings.cua_environment_for(env.value),
            max_turns=settings.logoff_max_turns,
            tracer=tracer,
        )
        tracer.event("logoff_ok")
    except Exception as exc:  # noqa: BLE001 - best-effort, never fail the result
        tracer.event("logoff_failed", error=f"{type(exc).__name__}: {exc}")


async def simulate_trade(
    *,
    harness: Harness,
    trade: TradeTask,
    settings: Settings,
    backend: AgentBackend,
    run_dir: Path,
) -> WorkerResult:
    """Drive one (trade, env) simulation. Never raises for automation failures —
    returns a WorkerResult with ok=False so the orchestrator can retry/record."""
    env = harness.env
    # Per-session real-time action trace (append mode: retries accumulate, each
    # delimited by its own `session_start` event). The Murex password is redacted
    # since LLM-login mode types it as an action.
    tracer = Tracer(
        run_dir / env.value / trade.trade_id / "trace.jsonl",
        label=f"{env.value}:{trade.trade_id}",
        secrets=[settings.murex_pass.get_secret_value()],
    )
    try:
        try:
            session = await harness.new_session(trade)
        except Exception as exc:
            tracer.event("error", phase="session_setup", error=f"{type(exc).__name__}: {exc}")
            return WorkerResult(
                trade_id=trade.trade_id, env=env, ok=False,
                error=f"session setup failed: {type(exc).__name__}: {exc}",
            )

        # Phase 1: drive the export. Compute a WorkerResult (ok or not) WITHOUT
        # returning yet, so every session-established path can funnel through the
        # phase-2 log-off below before the session is closed/torn down.
        try:
            width, height = session.display
            result = await backend.run(
                computer=session.computer,
                task=_build_task(trade, env.value, settings),
                display_width=width,
                display_height=height,
                environment=settings.cua_environment_for(env.value),
                max_turns=settings.max_turns,
                tracer=tracer,
            )

            export = await session.collect_export(timeout=settings.export_wait_secs)
            if export is None:
                tracer.event("error", phase="collect_export", error="no CSV was exported",
                             turns=result.turns, completed=result.completed)
                worker_result = WorkerResult(
                    trade_id=trade.trade_id, env=env, ok=False,
                    error="no CSV was exported", turns=result.turns,
                )
            else:
                # Reality gate: prove the collected file is a real, parseable CSV
                # whose postings reference THIS trade before trusting it. A failure
                # yields ok=False, which the tenacity retry in worker.py re-drives.
                check = validate_export(
                    Path(export),
                    trade_id=trade.trade_id,
                    sep=settings.csv_delimiter,
                    min_rows=settings.export_min_rows,
                    require_trade_id=settings.export_require_trade_id,
                    trade_id_columns=settings.export_trade_id_columns,
                )
                if not check.ok:
                    tracer.event("error", phase="validate_export", error=check.reason,
                                 turns=result.turns, completed=result.completed)
                    worker_result = WorkerResult(
                        trade_id=trade.trade_id, env=env, ok=False,
                        error=check.reason, turns=result.turns,
                    )
                else:
                    target = run_dir / env.value / trade.trade_id / "export.csv"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if Path(export) != target:
                        target.write_bytes(Path(export).read_bytes())
                    tracer.event("export_ok", csv=str(target), rows=check.rows,
                                 empty=check.empty, turns=result.turns)
                    worker_result = WorkerResult(
                        trade_id=trade.trade_id, env=env, ok=True,
                        csv_path=str(target), empty=check.empty, turns=result.turns,
                    )
        except Exception as exc:
            tracer.event("error", phase="loop", error=f"{type(exc).__name__}: {exc}")
            worker_result = WorkerResult(
                trade_id=trade.trade_id, env=env, ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Phase 2: clean Murex log-off AFTER the export is validated (never before,
        # so the model can't log off until the file is confirmed). Runs in EVERY
        # path where a session exists — valid (clean release) or invalid (leave
        # cleanly before the retry tears the session down). Best-effort.
        await _attempt_logoff(
            backend=backend, session=session, settings=settings, env=env, tracer=tracer
        )
        try:
            await session.close()
        except Exception:
            pass
        return worker_result
    finally:
        tracer.close()
