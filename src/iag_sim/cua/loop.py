"""The OpenAI computer-use agent loop (Responses API).

Contract (per https://developers.openai.com/api/docs/guides/tools-computer-use):
  - tool entry: {"type": "computer", "display_width", "display_height", "environment"}
  - model returns output items of type "computer_call" with `actions[]`, `call_id`
  - we execute the actions, screenshot, and reply with a "computer_call_output"
    item: {type, call_id, output: {type: "computer_screenshot", image_url, detail}}
  - carry `previous_response_id` each turn; loop until no computer_call is returned

Safety checks (`pending_safety_checks`) are acknowledged via an injected callback
so the policy lives with the caller, not buried here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI

from .actions import dispatch, wait_duration_ms
from .base import Computer
from .trace import NullTracer, Tracer

# Given a pending safety check dict, return True to acknowledge and proceed.
SafetyHandler = Callable[[dict], Awaitable[bool]]


@dataclass
class LoopResult:
    final_text: str
    turns: int
    completed: bool  # False if we hit max_turns


def _image_data_url(b64_png: str) -> str:
    return f"data:image/png;base64,{b64_png}"


def _computer_calls(response) -> list:
    return [item for item in response.output if getattr(item, "type", None) == "computer_call"]


def _actions_of(call) -> list[dict]:
    """GA returns `actions[]`; preview returned a single `action`. Normalize."""
    actions = getattr(call, "actions", None)
    if actions:
        return [a if isinstance(a, dict) else a.model_dump() for a in actions]
    action = getattr(call, "action", None)
    if action is not None:
        return [action if isinstance(action, dict) else action.model_dump()]
    return []


def _final_text(response) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _reasoning_texts(response) -> list[str]:
    """Model 'reasoning' summary text, when the model emits it (may be empty for
    encrypted/omitted reasoning). Surfaces the agent's intent in the trace."""
    out: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for s in getattr(item, "summary", []) or []:
            t = getattr(s, "text", None)
            if t is None and isinstance(s, dict):
                t = s.get("text")
            if t:
                out.append(t)
    return out


async def _deny(_check: dict) -> bool:  # default: never auto-acknowledge
    return False


async def run_cua_loop(
    *,
    client: AsyncOpenAI,
    computer: Computer,
    model: str,
    task: str,
    display_width: int,
    display_height: int,
    environment: str = "browser",
    max_turns: int = 60,
    on_safety_check: SafetyHandler = _deny,
    reasoning_effort: str | None = None,
    tracer: Tracer | NullTracer | None = None,
) -> LoopResult:
    """Drive `computer` to accomplish `task` using the computer-use model.

    Every model turn, action, reasoning summary, and safety check is emitted to
    `tracer` in real time (flushed per event) for a live, channel-agnostic
    action trace. Pass a `Tracer` to record; defaults to a no-op.
    """
    tracer = tracer or NullTracer()
    # Tool shape depends on the model generation:
    #   - GA computer tool (gpt-5.x and later): `{"type": "computer"}` ONLY. The
    #     API rejects display_width/display_height/environment ("Unknown
    #     parameter: tools[0].display_width"); it sizes from the screenshots.
    #   - Legacy `computer-use-preview` model: the old `computer_use_preview`
    #     tool that DOES take the display dims + environment.
    if "computer-use-preview" in model:
        tool = {
            "type": "computer_use_preview",
            "display_width": display_width,
            "display_height": display_height,
            "environment": environment,
        }
    else:
        tool = {"type": "computer"}

    # Reasoning effort -> Responses API reasoning.effort. Unset = provider default
    # (nothing sent). Applied to every create() call below.
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None

    tracer.event(
        "session_start", model=model, environment=environment,
        tool=tool["type"], max_turns=max_turns, task_chars=len(task),
        effort=reasoning_effort,
    )

    first_shot = await computer.screenshot()
    response = await client.responses.create(
        model=model,
        tools=[tool],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(first_shot),
                        "detail": "original",
                    },
                ],
            }
        ],
        **({"reasoning": reasoning} if reasoning else {}),
    )

    for turn in range(1, max_turns + 1):
        for rtext in _reasoning_texts(response):
            tracer.event("reasoning", turn=turn, text=rtext)

        calls = _computer_calls(response)
        if not calls:
            final = _final_text(response)
            tracer.event("done", turns=turn - 1, completed=True, text=final)
            return LoopResult(final_text=final, turns=turn - 1, completed=True)

        call = calls[0]
        for action in _actions_of(call):
            if action.get("type") == "wait":
                # surface how long the agent chose to pause
                tracer.event("action", turn=turn, action=action,
                             wait_ms=wait_duration_ms(action))
            else:
                tracer.event("action", turn=turn, action=action)
            if action.get("type") != "screenshot":
                await dispatch(computer, action)

        acknowledged: list[dict] = []
        for check in getattr(call, "pending_safety_checks", []) or []:
            check_dict = check if isinstance(check, dict) else check.model_dump()
            ack = await on_safety_check(check_dict)
            tracer.event("safety_check", turn=turn, acknowledged=ack, check=check_dict)
            if ack:
                acknowledged.append(check_dict)

        shot = await computer.screenshot()
        output_item: dict = {
            "type": "computer_call_output",
            "call_id": call.call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": _image_data_url(shot),
                "detail": "original",
            },
        }
        if acknowledged:
            output_item["acknowledged_safety_checks"] = acknowledged

        response = await client.responses.create(
            model=model,
            tools=[tool],
            input=[output_item],
            previous_response_id=response.id,
            **({"reasoning": reasoning} if reasoning else {}),
        )

    final = _final_text(response)
    tracer.event("timeout", turns=max_turns, completed=False, text=final)
    return LoopResult(final_text=final, turns=max_turns, completed=False)
