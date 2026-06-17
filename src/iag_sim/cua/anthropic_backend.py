"""Anthropic Messages-API computer-use loop (direct Anthropic + AWS Bedrock).

Same `Computer` interface and canonical action dispatch as the OpenAI loop; only
the transport differs. The client is either `AsyncAnthropic` or
`AsyncAnthropicBedrock` — both expose `.beta.messages.stream` with identical
request/response shapes, so one loop serves both. Requests are STREAMED (the SDK
refuses a non-streaming call whose max_tokens could run past the 10-min ceiling);
`stream.get_final_message()` yields the same assembled Message. Conversation state
is a growing `messages` list (the Messages API is stateless — no previous_response_id).

Per-turn contract:
  - request: tools=[computer tool], messages=history, betas=[beta flag]
  - response.content: text / thinking / tool_use blocks (appended back VERBATIM —
    thinking blocks must be preserved or the next turn is rejected)
  - reply: a user message of tool_result blocks, each carrying the post-action
    screenshot as a base64 image, referencing the originating tool_use_id
  - done when stop_reason == "end_turn" or no tool_use blocks remain
"""

from __future__ import annotations

from typing import Any

from anthropic import NOT_GIVEN
from anthropic.lib.streaming import BetaAsyncMessageStreamManager

from .actions import dispatch, wait_duration_ms
from .anthropic_actions import to_canonical
from .base import Action, Computer
from .loop import LoopResult, SafetyHandler, _deny
from .trace import NullTracer, Tracer

# Manual extended-thinking budgets (tokens), used ONLY for older models that lack
# adaptive thinking + the effort param (e.g. Bedrock Sonnet 4.5). Newest models use
# adaptive thinking + output_config.effort instead (see run_anthropic_loop). xhigh/
# max aren't real on those old models, so they clamp to a high budget.
_THINKING_BUDGET = {
    "minimal": 1024, "low": 4096, "medium": 8192,
    "high": 16384, "xhigh": 16384, "max": 24576,
}
# Adaptive mode: floor on max_tokens (thinking + text) per effort so high-effort
# replies aren't truncated mid-thought. The docs suggest ~64k for xhigh/max; these
# conservative floors are overridden upward by a larger CUA_MAX_TOKENS.
_ADAPTIVE_MIN_MAX_TOKENS = {"high": 16000, "xhigh": 32000, "max": 32000}
# Effort levels the Anthropic ADAPTIVE API accepts in output_config.effort. It 400s
# on anything else ("unknown variant `none`, expected one of low/medium/high/xhigh/
# max"). `none`/`minimal` are OpenAI-only tiers, so they are NOT forwarded here.
_ADAPTIVE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _blocks(response) -> list:
    return list(getattr(response, "content", []) or [])


def _btype(block) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _battr(block, name: str):
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _final_text(blocks) -> str:
    parts: list[str] = []
    for b in blocks:
        if _btype(b) == "text":
            t = _battr(b, "text")
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def _screenshot_result(tool_use_id: str, b64_png: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_png,
                },
            }
        ],
    }


def _error_result(tool_use_id: str, message: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "is_error": True,
        "content": [{"type": "text", "text": message}],
    }


def _prune_screenshots(messages: list[dict], keep: int) -> None:
    """Replace all but the last `keep` screenshot image blocks (in user
    tool_result messages) with a text placeholder, to bound token cost."""
    if keep <= 0:
        return
    locations: list[tuple[int, int, int]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    for ii, ib in enumerate(inner):
                        if isinstance(ib, dict) and ib.get("type") == "image":
                            locations.append((mi, bi, ii))
    for mi, bi, ii in locations[:-keep]:
        messages[mi]["content"][bi]["content"][ii] = {
            "type": "text",
            "text": "[screenshot pruned to save tokens]",
        }


# --- Prompt caching (cache_control breakpoints) -----------------------------
# 5-min ephemeral cache. The Messages API re-sends the whole growing `messages`
# list every turn (it is stateless), so without breakpoints the entire prefix is
# re-billed at full price each turn. A breakpoint marks the cumulative prefix up
# to it as cacheable: written once (1.25x), then read at 0.1x on later turns —
# but ONLY if that prefix is byte-identical next turn (caching is prefix-exact).
_EPHEMERAL = {"type": "ephemeral"}


def _has_real_image(msg: dict) -> bool:
    """True if a user message still carries an un-pruned screenshot image block."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                if any(isinstance(ib, dict) and ib.get("type") == "image" for ib in inner):
                    return True
    return False


def _last_user_list_index(messages: list[dict], before: int | None = None) -> int | None:
    """Index of the last user message with list content (optionally strictly before
    `before`). cache_control can only attach to our own dict blocks — those live in
    user tool_result messages; assistant blocks are SDK objects we must not mutate."""
    hi = len(messages) if before is None else before
    for i in range(hi - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            return i
    return None


def _cache_anchor(messages: list[dict], keep: int) -> int | None:
    """Index of the message to carry the rolling breakpoint: the frontier of the
    byte-stable prefix that survives the NEXT turn's screenshot prune.

    Pruning keeps only the last `keep` real screenshots. Once at capacity (real
    count == keep — the steady state, since the loop prunes every turn), the OLDEST
    real screenshot is the one stubbed next turn, so the largest prefix that stays
    byte-identical is everything strictly BEFORE it (caching is prefix-exact: the
    first changed block breaks the match, so the kept screenshots after it can't be
    cached cross-turn anyway — they're fresh each turn). Below capacity (pruning
    off, or fewer than `keep` screenshots so far) nothing stubs next turn, so the
    whole history is stable -> anchor the newest user message. None when no
    eligible user message exists yet (e.g. turn 1, before any screenshot)."""
    real = [i for i, m in enumerate(messages) if m.get("role") == "user" and _has_real_image(m)]
    if keep <= 0 or len(real) < keep:
        return _last_user_list_index(messages)
    # at capacity: real[0] (oldest real screenshot) is stubbed next turn.
    return _last_user_list_index(messages, before=real[0])


def _clear_cache_control(messages: list[dict]) -> None:
    """Drop cache_control from every dict content block, so the rolling anchor does
    not accumulate past Anthropic's 4-breakpoint cap as it moves each turn."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


def _apply_cache_breakpoints(messages: list[dict], keep: int) -> None:
    """Re-place the single rolling conversation breakpoint at the prune frontier."""
    _clear_cache_control(messages)
    i = _cache_anchor(messages, keep)
    if i is None:
        return
    last = messages[i]["content"][-1]
    if isinstance(last, dict):
        last["cache_control"] = dict(_EPHEMERAL)


async def run_anthropic_loop(
    *,
    client: Any,
    model: str,
    tool_version: str,
    beta_flag: str,
    computer: Computer,
    task: str,
    display_width: int,
    display_height: int,
    max_tokens: int = 4096,
    max_turns: int = 60,
    keep_last_screenshots: int = 3,
    reasoning_effort: str | None = None,
    prompt_cache: bool = True,
    tracer: Tracer | NullTracer | None = None,
) -> LoopResult:
    """Drive `computer` to accomplish `task` via the Anthropic computer-use tool."""
    tracer = tracer or NullTracer()
    tool = {
        "type": tool_version,
        "name": "computer",
        "display_width_px": display_width,
        "display_height_px": display_height,
    }
    if prompt_cache:
        # Static breakpoint: caches the tool def (and any future system prompt).
        tool["cache_control"] = dict(_EPHEMERAL)
    # Reasoning effort -> thinking + output_config.effort. Unset = provider default
    # (no thinking, no effort sent). The thinking MODE tracks the model generation,
    # which is paired with the tool version:
    #   - computer_20251124 (Opus 4.8/4.7/4.6, Sonnet 4.6): ADAPTIVE thinking
    #     (`{"type":"adaptive"}`) + top-level `output_config.effort`. Manual
    #     budget_tokens is rejected with a 400 on Opus 4.8/4.7.
    #   - older gen (Bedrock Sonnet 4.5): manual `budget_tokens`; no effort param.
    # `effort` accepts low|medium|high|xhigh|max; availability is model-specific and
    # the API validates it (xhigh = Opus only; max = Sonnet 4.6+/Opus).
    adaptive = tool_version == "computer_20251124"
    thinking = None
    output_config = None
    effective_max_tokens = max_tokens
    # `none` means reasoning is explicitly OFF — emit no thinking/effort (identical
    # to unset). `none`/`minimal` are OpenAI-only tiers; the adaptive API 400s on
    # them ("unknown variant `none`"), so on the adaptive path only the accepted
    # efforts are forwarded and a sub-low value falls through to no thinking.
    if reasoning_effort and reasoning_effort != "none":
        if adaptive:
            if reasoning_effort in _ADAPTIVE_EFFORTS:
                thinking = {"type": "adaptive"}
                output_config = {"effort": reasoning_effort}
                # Give thinking room at higher effort so replies aren't truncated.
                effective_max_tokens = max(
                    max_tokens, _ADAPTIVE_MIN_MAX_TOKENS.get(reasoning_effort, 0)
                )
        else:
            budget = _THINKING_BUDGET.get(reasoning_effort, 16384)
            thinking = {"type": "enabled", "budget_tokens": budget}
            effective_max_tokens = budget + max_tokens
    tracer.event(
        "session_start", model=model, provider="anthropic", tool=tool_version,
        beta=beta_flag, max_turns=max_turns, task_chars=len(task),
        effort=reasoning_effort, max_tokens=effective_max_tokens,
    )

    messages: list[dict] = [{"role": "user", "content": task}]
    response = None
    for turn in range(1, max_turns + 1):
        if prompt_cache:
            # Re-place the rolling breakpoint at the (post-prune) frontier each turn.
            _apply_cache_breakpoints(messages, keep_last_screenshots)
        # Stream the request: the SDK refuses a non-streaming create() once
        # max_tokens is large enough that the worst-case generation could exceed
        # 10 min ("Streaming is required for operations that may take longer than
        # 10 minutes"), and a 60-turn high-effort loop is exactly that long-request
        # case. Direct Anthropic exposes the high-level .stream() helper; the
        # Bedrock beta resource does NOT, so fall back to create(stream=True)
        # wrapped in the SDK's stream-accumulator manager. Both assemble the same
        # final Message (content/usage/stop_reason) and preserve thinking-block
        # signatures (the SDK accumulator reassembles them).
        request_kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=effective_max_tokens,
            tools=[tool],
            messages=messages,
            betas=[beta_flag],
            **({"thinking": thinking} if thinking else {}),
            **({"output_config": output_config} if output_config else {}),
        )
        beta_messages = client.beta.messages
        if hasattr(beta_messages, "stream"):
            async with beta_messages.stream(**request_kwargs) as stream:
                response = await stream.get_final_message()
        else:
            manager = BetaAsyncMessageStreamManager(
                beta_messages.create(stream=True, **request_kwargs),
                output_format=NOT_GIVEN,
            )
            async with manager as stream:
                response = await stream.get_final_message()
        usage = getattr(response, "usage", None)
        if usage is not None:
            tracer.event(
                "usage", turn=turn,
                input=getattr(usage, "input_tokens", None),
                output=getattr(usage, "output_tokens", None),
                cache_read=getattr(usage, "cache_read_input_tokens", None),
                cache_write=getattr(usage, "cache_creation_input_tokens", None),
            )
        blocks = _blocks(response)

        for b in blocks:
            bt = _btype(b)
            if bt == "text":
                t = _battr(b, "text")
                if t:
                    tracer.event("reasoning", turn=turn, text=t)
            elif bt == "thinking":
                t = _battr(b, "thinking")
                if t:
                    tracer.event("reasoning", turn=turn, text=t)

        # Append assistant content VERBATIM (preserves thinking-block signatures).
        messages.append({"role": "assistant", "content": blocks})

        tool_uses = [b for b in blocks if _btype(b) == "tool_use"]
        stop_reason = getattr(response, "stop_reason", None)
        if not tool_uses:
            final = _final_text(blocks)
            tracer.event("done", turns=turn, completed=True, text=final, stop_reason=stop_reason)
            return LoopResult(final_text=final, turns=turn, completed=True)

        tool_results: list[dict] = []
        for tu in tool_uses:
            tu_id = _battr(tu, "id")
            raw: Action = _battr(tu, "input") or {}
            try:
                canonical = to_canonical(raw)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                tracer.event("error", turn=turn, error=msg, action=raw)
                tool_results.append(_error_result(tu_id, msg))
                continue
            for ca in canonical:
                if ca.get("type") == "wait":
                    tracer.event("action", turn=turn, action=ca, wait_ms=wait_duration_ms(ca))
                else:
                    tracer.event("action", turn=turn, action=ca)
                if ca.get("type") != "screenshot":
                    await dispatch(computer, ca)
            shot = await computer.screenshot()
            tool_results.append(_screenshot_result(tu_id, shot))

        messages.append({"role": "user", "content": tool_results})
        _prune_screenshots(messages, keep_last_screenshots)

    final = _final_text(_blocks(response)) if response is not None else ""
    tracer.event("timeout", turns=max_turns, completed=False, text=final)
    return LoopResult(final_text=final, turns=max_turns, completed=False)


class AnthropicBackend:
    """AgentBackend over the Anthropic computer-use tool (direct API or Bedrock)."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        tool_version: str,
        beta_flag: str,
        max_tokens: int = 4096,
        keep_last_screenshots: int = 3,
        reasoning_effort: str | None = None,
        prompt_cache: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.tool_version = tool_version
        self.beta_flag = beta_flag
        self.max_tokens = max_tokens
        self.keep_last_screenshots = keep_last_screenshots
        self.reasoning_effort = reasoning_effort
        self.prompt_cache = prompt_cache

    async def run(
        self,
        *,
        computer: Computer,
        task: str,
        display_width: int,
        display_height: int,
        environment: str = "browser",
        max_turns: int = 60,
        on_safety_check: SafetyHandler = _deny,
        tracer: Tracer | NullTracer | None = None,
    ) -> LoopResult:
        # `environment` and `on_safety_check` are OpenAI-loop concepts; accepted for
        # signature parity but unused (Anthropic computer use has no environment
        # parameter and no pending_safety_checks).
        return await run_anthropic_loop(
            client=self.client,
            model=self.model,
            tool_version=self.tool_version,
            beta_flag=self.beta_flag,
            computer=computer,
            task=task,
            display_width=display_width,
            display_height=display_height,
            max_tokens=self.max_tokens,
            max_turns=max_turns,
            keep_last_screenshots=self.keep_last_screenshots,
            reasoning_effort=self.reasoning_effort,
            prompt_cache=self.prompt_cache,
            tracer=tracer,
        )
