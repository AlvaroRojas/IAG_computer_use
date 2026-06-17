"""Custom-tool computer-use backend (OpenAI Responses API, any compatible endpoint).

For models that speak the Responses API and support function/custom tools + vision
but expose NO built-in `computer` tool — notably **GPT-5.x on AWS Bedrock** via the
mantle endpoint (`https://bedrock-mantle.<region>.api.aws/openai/v1`). Instead of the
native tool, we declare ONE custom function tool (`computer`) carrying the canonical
action vocabulary, feed screenshots as `input_image`, and run the model's tool calls
through the SAME `cua/actions.py` dispatcher the native loop uses. Computer-use is
thus *emulated* via function-calling + vision.

Two contract differences from the native loop (`loop.py`):
  - A Responses `function_call_output` is TEXT-only, so the post-action screenshot
    cannot ride inside it. We ack the call with a short text output and feed the new
    screenshot as a fresh `user` message with an `input_image` block.
  - No server-side state: the loop is STATELESS (a growing `input` list, like the
    Anthropic backend). The model's output items are echoed back verbatim each turn;
    older screenshots are pruned (`keep_last_screenshots`) to bound token cost.

There are no `pending_safety_checks` on a custom tool, so `on_safety_check` is unused
(kept for `AgentBackend` parity).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from openai import AsyncOpenAI

from .actions import dispatch, wait_duration_ms
from .base import Computer
from .loop import (
    LoopResult,
    SafetyHandler,
    _deny,
    _final_text,
    _image_data_url,
    _reasoning_texts,
)
from .trace import NullTracer, Tracer

# The single custom tool. Its property names mirror the canonical action dicts that
# `cua/actions.py` dispatches; `action` maps to the dispatcher's `type`. Keep the
# enum in sync with `dispatch()`.
_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "computer",
    "description": (
        "Operate the computer to accomplish the task. Each call performs ONE action "
        "on the screen shown in the most recent screenshot. Coordinates are pixels "
        "from the top-left of that screenshot."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "click",
                    "double_click",
                    "move",
                    "drag",
                    "scroll",
                    "type",
                    "keypress",
                    "wait",
                    "screenshot",
                ],
                "description": "The action to perform.",
            },
            "x": {"type": "integer", "description": "X pixel (click/double_click/move/scroll)."},
            "y": {"type": "integer", "description": "Y pixel (click/double_click/move/scroll)."},
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for click/double_click (default left).",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key names for keypress, or modifier keys held during a click.",
            },
            "text": {"type": "string", "description": "Text to type (action=type)."},
            "path": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Sequence of [x, y] points for action=drag.",
            },
            "scroll_x": {"type": "integer", "description": "Horizontal wheel notches (action=scroll)."},
            "scroll_y": {"type": "integer", "description": "Vertical wheel notches (action=scroll)."},
            "duration_ms": {"type": "integer", "description": "Pause length for action=wait (ms)."},
        },
        "required": ["action"],
    },
}

# Replaces a pruned screenshot so the message stays valid but cheap.
_PRUNED_STUB = {"type": "input_text", "text": "[screenshot pruned to save tokens]"}

# The Bedrock mantle GPT engine intermittently returns an HTTP 400 wrapping an
# INTERNAL job-registration/dispatch failure (a 404 from the engine, JSON-RPC -32602)
# even for a perfectly valid request — a server-side capacity/routing flake, not a
# request-shape error (an identical request on another session succeeds). The OpenAI
# SDK never auto-retries 400s, so without this the flake kills in-session progress and
# forces a full session restart (re-login). We retry the SAME create() call when the
# error message carries one of these markers; anything else propagates immediately.
_TRANSIENT_MARKERS = (
    "Job registration failed",
    "Task submission failed",
    "-32602",
)

# Short fixed pause between same-session retries of the transient mantle flake. The
# failure is instant engine-routing churn (a 404 to a missing node), not throttling,
# so a brief delay before re-dispatch is enough; exponential backoff only wastes time.
_TRANSIENT_RETRY_DELAY_SECS = 0.5


def _is_transient(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _to_canonical(args: dict) -> dict:
    """Map the custom tool's args to the canonical action dict `dispatch` expects:
    `action` -> `type`; every other field already matches by name."""
    action = dict(args)
    atype = action.pop("action", None)
    action["type"] = atype
    return action


def _function_calls(response, name: str = "computer") -> list:
    return [
        it
        for it in response.output
        if getattr(it, "type", None) == "function_call"
        and getattr(it, "name", None) == name
    ]


def _output_as_input(response) -> list[dict]:
    """The model's output items, serialized to dicts so they can be echoed back into
    the next turn's `input` (stateless conversation; no previous_response_id)."""
    items: list[dict] = []
    for it in response.output:
        items.append(it if isinstance(it, dict) else it.model_dump(exclude_none=True))
    return items


def _prune_screenshots(input_items: list[dict], keep_last: int) -> None:
    """Mutate `input_items` in place: keep only the most recent `keep_last`
    `input_image` blocks; replace older ones with a text stub. 0 = keep all.
    Only user messages carry images (echoed assistant items never do)."""
    if keep_last <= 0:
        return
    # Collect (message_content_list, index) for every image block, in order.
    images: list[tuple[list, int]] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "input_image":
                images.append((content, i))
    for content, i in images[:-keep_last]:
        content[i] = dict(_PRUNED_STUB)


def _emit_usage(tracer, turn: int, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    tracer.event(
        "usage",
        turn=turn,
        input=getattr(usage, "input_tokens", None),
        output=getattr(usage, "output_tokens", None),
        cache_read=cached,
    )


class OpenAICustomToolBackend:
    """AgentBackend that emulates computer-use with a custom function tool, for
    Responses-API models lacking the native `computer` tool (e.g. GPT-5.x on Bedrock)."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4096,
        keep_last_screenshots: int = 3,
        transient_retries: int = 8,
        prompt_cache_retention: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.keep_last_screenshots = keep_last_screenshots
        self.transient_retries = transient_retries
        self.prompt_cache_retention = prompt_cache_retention

    async def _create(self, tracer, turn: int, **kwargs):
        """Call responses.create, retrying ONLY the mantle engine's transient
        job-registration 400s (see _TRANSIENT_MARKERS) IN THE SAME SESSION — the
        request is valid, so an immediate re-dispatch just lands a different engine
        node. The limit is per-turn (`transient_retries`); a session therefore absorbs
        any number of flakes across turns, and only escalates to a worker-level
        session restart if a SINGLE turn flakes `transient_retries` times in a row.
        A short fixed delay (not exponential backoff) — this is instant routing churn,
        not rate-limiting, so waiting longer buys nothing."""
        for attempt in range(1, self.transient_retries + 1):
            try:
                return await self.client.responses.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised unless transient
                if attempt >= self.transient_retries or not _is_transient(exc):
                    raise
                tracer.event(
                    "retry", turn=turn, attempt=attempt, error=str(exc)[:200]
                )
                await asyncio.sleep(_TRANSIENT_RETRY_DELAY_SECS)

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
        tracer = tracer or NullTracer()
        reasoning = {"effort": self.reasoning_effort} if self.reasoning_effort else None
        extra = {"reasoning": reasoning} if reasoning else {}
        # GPT-5.5 caches only when retention is requested AND a stable prompt_cache_key
        # is sent: on mantle the in-memory cache is engine-node-local, and the flaky
        # routing scatters requests across nodes — the key pins a session's turns to the
        # same cache (without it cache_read stayed 0 in a live run). The key is derived
        # from the task so it is constant across a session's turns yet unique per
        # trade/env. The cacheable prefix is the task preamble + already-stubbed history
        # (a stub is byte-stable forever); pruning is untouched (it just bounds the
        # uncached real-image tail).
        if self.prompt_cache_retention:
            extra["prompt_cache_retention"] = self.prompt_cache_retention
            extra["prompt_cache_key"] = "murex-" + hashlib.sha1(
                task.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:16]

        tracer.event(
            "session_start",
            model=self.model,
            environment=environment,
            tool="computer(custom)",
            max_turns=max_turns,
            task_chars=len(task),
            effort=self.reasoning_effort,
        )

        first_shot = await computer.screenshot()
        input_items: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task},
                    {"type": "input_image", "image_url": _image_data_url(first_shot)},
                ],
            }
        ]

        for turn in range(1, max_turns + 1):
            response = await self._create(
                tracer,
                turn,
                model=self.model,
                tools=[_TOOL],
                tool_choice="auto",
                input=input_items,
                max_output_tokens=self.max_output_tokens,
                **extra,
            )
            _emit_usage(tracer, turn, response)
            for rtext in _reasoning_texts(response):
                tracer.event("reasoning", turn=turn, text=rtext)

            calls = _function_calls(response)
            if not calls:
                final = _final_text(response)
                tracer.event("done", turns=turn, completed=True, text=final)
                return LoopResult(final_text=final, turns=turn, completed=True)

            # Echo the model's items (reasoning + function_call) back verbatim.
            input_items.extend(_output_as_input(response))

            for call in calls:
                try:
                    raw = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    raw = {}
                action = _to_canonical(raw)
                if action.get("type") == "wait":
                    tracer.event("action", turn=turn, action=action, wait_ms=wait_duration_ms(action))
                else:
                    tracer.event("action", turn=turn, action=action)
                if action.get("type") != "screenshot":
                    await dispatch(computer, action)
                # Text-only ack; the screenshot rides a separate user message below.
                input_items.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": "ok"}
                )

            shot = await computer.screenshot()
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": _image_data_url(shot)},
                    ],
                }
            )
            _prune_screenshots(input_items, self.keep_last_screenshots)

        final = _final_text(response)
        tracer.event("timeout", turns=max_turns, completed=False, text=final)
        return LoopResult(final_text=final, turns=max_turns, completed=False)
