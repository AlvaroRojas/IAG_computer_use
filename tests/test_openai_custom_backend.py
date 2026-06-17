"""Custom-tool computer-use backend (cua/openai_custom_backend.py), driven by a fake
Responses client. No network: client.responses.create is hand-scripted.
"""

from __future__ import annotations

import json

import pytest

from iag_sim.cua.openai_custom_backend import (
    OpenAICustomToolBackend,
    _is_transient,
    _prune_screenshots,
    _to_canonical,
)


# --- fake Responses client + item objects (mirror the SDK shapes the loop reads) ---


class _Item:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        d = dict(self.__dict__)
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


def _fc(arguments: dict, *, call_id="c1", name="computer") -> _Item:
    return _Item(
        type="function_call", name=name, call_id=call_id, id="fc_" + call_id,
        arguments=json.dumps(arguments),
    )


def _msg(text: str) -> _Item:
    return _Item(type="message", content=[_Item(type="output_text", text=text)])


class _Resp:
    def __init__(self, output, usage=None):
        self.output = output
        self.usage = usage


class _FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)


class _FakeComputer:
    def __init__(self):
        self.calls: list[tuple] = []

    async def screenshot(self):
        self.calls.append(("screenshot",))
        return "B64"

    async def click(self, x, y, button="left", keys=None):
        self.calls.append(("click", x, y, button, keys))

    async def type(self, text):
        self.calls.append(("type", text))

    async def keypress(self, keys):
        self.calls.append(("keypress", keys))


class _CapTracer:
    def __init__(self):
        self.events: list[tuple] = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def close(self):
        pass


def _backend(client, **kw) -> OpenAICustomToolBackend:
    return OpenAICustomToolBackend(client=client, model="openai.gpt-5.5", **kw)


# --- pure helpers ---


def test_to_canonical_renames_action_to_type_and_keeps_fields():
    assert _to_canonical({"action": "click", "x": 5, "y": 6, "button": "right"}) == {
        "type": "click", "x": 5, "y": 6, "button": "right"
    }
    assert _to_canonical({"action": "type", "text": "hi"}) == {"type": "type", "text": "hi"}
    assert _to_canonical({"action": "keypress", "keys": ["ctrl", "a"]}) == {
        "type": "keypress", "keys": ["ctrl", "a"]
    }
    assert _to_canonical({"action": "scroll", "x": 1, "y": 2, "scroll_y": -3}) == {
        "type": "scroll", "x": 1, "y": 2, "scroll_y": -3
    }


def _user_img(stub=False):
    block = (
        {"type": "input_text", "text": "[screenshot pruned to save tokens]"}
        if stub
        else {"type": "input_image", "image_url": "data:image/png;base64,X"}
    )
    return {"role": "user", "content": [block]}


def test_prune_keeps_last_n_images():
    items = [
        {"role": "user", "content": [
            {"type": "input_text", "text": "task"},
            {"type": "input_image", "image_url": "d0"}]},
        {"type": "function_call_output", "call_id": "c", "output": "ok"},
        _user_img(),
        _user_img(),
    ]
    _prune_screenshots(items, keep_last=2)
    # 3 images total, keep 2 -> the FIRST (in the task message) is stubbed.
    assert items[0]["content"][1] == {
        "type": "input_text", "text": "[screenshot pruned to save tokens]"
    }
    assert items[2]["content"][0]["type"] == "input_image"
    assert items[3]["content"][0]["type"] == "input_image"


def test_prune_zero_keeps_all():
    items = [_user_img(), _user_img()]
    _prune_screenshots(items, keep_last=0)
    assert all(i["content"][0]["type"] == "input_image" for i in items)


# --- the loop ---


async def test_click_dispatched_and_screenshot_fed_back():
    client = _FakeClient([
        _Resp([_fc({"action": "click", "x": 5, "y": 6}, call_id="c1")]),
        _Resp([_msg("DONE")]),
    ])
    comp = _FakeComputer()
    result = await _backend(client).run(
        computer=comp, task="do it", display_width=1024, display_height=768, max_turns=5,
    )

    assert result.completed is True
    assert result.final_text == "DONE"
    assert result.turns == 2
    assert ("click", 5, 6, "left", None) in comp.calls

    # first create carried the custom tool + auto choice + token budget
    first = client.responses.calls[0]
    assert first["tools"][0]["name"] == "computer"
    assert first["tools"][0]["type"] == "function"
    assert first["tool_choice"] == "auto"

    # the SECOND create's input echoes the function_call, then a text ack, then a
    # fresh user image (function_call_output is text-only, so the shot rides a user msg)
    second_input = client.responses.calls[1]["input"]
    acks = [b for b in second_input if isinstance(b, dict) and b.get("type") == "function_call_output"]
    assert acks and acks[0]["call_id"] == "c1" and acks[0]["output"] == "ok"
    user_imgs = [
        b for m in second_input if isinstance(m, dict) and m.get("role") == "user"
        for b in m.get("content", []) if b.get("type") == "input_image"
    ]
    assert len(user_imgs) >= 2  # the task image + the post-action image


async def test_type_and_keypress_round_trip():
    client = _FakeClient([
        _Resp([_fc({"action": "type", "text": "MUREXBO"}, call_id="t1")]),
        _Resp([_fc({"action": "keypress", "keys": ["enter"]}, call_id="k1")]),
        _Resp([_msg("ok")]),
    ])
    comp = _FakeComputer()
    await _backend(client).run(
        computer=comp, task="login", display_width=800, display_height=600, max_turns=5,
    )
    assert ("type", "MUREXBO") in comp.calls
    assert ("keypress", ["enter"]) in comp.calls


async def test_reasoning_effort_wired_into_create():
    client = _FakeClient([_Resp([_msg("done")])])
    await _backend(client, reasoning_effort="high", max_output_tokens=2048).run(
        computer=_FakeComputer(), task="t", display_width=800, display_height=600, max_turns=3,
    )
    call = client.responses.calls[0]
    assert call["reasoning"] == {"effort": "high"}
    assert call["max_output_tokens"] == 2048


async def test_no_reasoning_effort_omits_reasoning():
    client = _FakeClient([_Resp([_msg("done")])])
    await _backend(client).run(
        computer=_FakeComputer(), task="t", display_width=800, display_height=600, max_turns=3,
    )
    assert "reasoning" not in client.responses.calls[0]


async def test_prompt_cache_retention_sent_when_set():
    client = _FakeClient([_Resp([_msg("done")])])
    await _backend(client, prompt_cache_retention="in_memory").run(
        computer=_FakeComputer(), task="t", display_width=800, display_height=600, max_turns=3,
    )
    assert client.responses.calls[0]["prompt_cache_retention"] == "in_memory"


async def test_prompt_cache_retention_omitted_when_unset():
    client = _FakeClient([_Resp([_msg("done")])])
    await _backend(client).run(
        computer=_FakeComputer(), task="t", display_width=800, display_height=600, max_turns=3,
    )
    assert "prompt_cache_retention" not in client.responses.calls[0]


async def test_usage_event_emitted():
    usage = _Item(input_tokens=900, output_tokens=40,
                  input_tokens_details=_Item(cached_tokens=512))
    client = _FakeClient([
        _Resp([_fc({"action": "click", "x": 1, "y": 1}, call_id="c1")], usage=usage),
        _Resp([_msg("DONE")]),
    ])
    cap = _CapTracer()
    await _backend(client).run(
        computer=_FakeComputer(), task="go",
        display_width=800, display_height=600, max_turns=5, tracer=cap,
    )
    usages = [f for (k, f) in cap.events if k == "usage"]
    assert usages[0]["input"] == 900
    assert usages[0]["output"] == 40
    assert usages[0]["cache_read"] == 512


async def test_max_turns_timeout_not_completed():
    # never returns a no-call response -> loops until max_turns
    client = _FakeClient([
        _Resp([_fc({"action": "click", "x": 1, "y": 1}, call_id=f"c{i}")]) for i in range(3)
    ])
    result = await _backend(client).run(
        computer=_FakeComputer(), task="loop",
        display_width=800, display_height=600, max_turns=3,
    )
    assert result.completed is False
    assert result.turns == 3


# --- transient mantle-flake retry (the smoke-run blocker) ---

_FLAKE = ("BadRequestError: 400 - JSON-RPC error -32602: Job registration failed: "
          "Engine bad request: Task submission failed with status 404 Not Found")


def test_is_transient_matches_mantle_flake_only():
    assert _is_transient(RuntimeError(_FLAKE)) is True
    assert _is_transient(RuntimeError("400 - invalid 'tools[0]': unknown field")) is False


class _FlakyResponses:
    """Raises `fail_times` transient errors, then serves scripted responses."""

    def __init__(self, fail_times, then, *, exc=None):
        self.fail_times = fail_times
        self.then = list(then)
        self.exc = exc or RuntimeError(_FLAKE)
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self.exc
        return self.then.pop(0)


async def _noop_sleep(*_a, **_k):
    pass


async def test_transient_flake_retried_without_losing_session(monkeypatch):
    monkeypatch.setattr("iag_sim.cua.openai_custom_backend.asyncio.sleep", _noop_sleep)
    client = _FakeClient([])
    client.responses = _FlakyResponses(fail_times=2, then=[_Resp([_msg("DONE")])])
    cap = _CapTracer()
    result = await OpenAICustomToolBackend(
        client=client, model="openai.gpt-5.5", transient_retries=5
    ).run(
        computer=_FakeComputer(), task="t",
        display_width=800, display_height=600, max_turns=3, tracer=cap,
    )
    assert result.final_text == "DONE"
    assert client.responses.attempts == 3  # 2 flakes + 1 success, same turn
    assert sum(1 for k, _ in cap.events if k == "retry") == 2


async def test_non_transient_error_propagates(monkeypatch):
    monkeypatch.setattr("iag_sim.cua.openai_custom_backend.asyncio.sleep", _noop_sleep)
    client = _FakeClient([])
    client.responses = _FlakyResponses(
        fail_times=99, then=[], exc=RuntimeError("400 - bad field 'reasoning'")
    )
    with pytest.raises(RuntimeError, match="bad field"):
        await OpenAICustomToolBackend(client=client, model="m", transient_retries=5).run(
            computer=_FakeComputer(), task="t",
            display_width=800, display_height=600, max_turns=3,
        )


async def test_transient_retries_exhausted_reraises(monkeypatch):
    monkeypatch.setattr("iag_sim.cua.openai_custom_backend.asyncio.sleep", _noop_sleep)
    client = _FakeClient([])
    client.responses = _FlakyResponses(fail_times=99, then=[])
    with pytest.raises(RuntimeError, match="Job registration failed"):
        await OpenAICustomToolBackend(client=client, model="m", transient_retries=3).run(
            computer=_FakeComputer(), task="t",
            display_width=800, display_height=600, max_turns=3,
        )
    assert client.responses.attempts == 3
