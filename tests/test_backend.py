"""Backend selection + the Anthropic Messages loop, exercised with a fake client.

No network: clients are constructed (offline-safe) for the factory tests, and the
loop test drives a hand-scripted fake `client.beta.messages.stream`.
"""

from __future__ import annotations

from iag_sim.config import Settings
from iag_sim.cua.anthropic_backend import (
    AnthropicBackend,
    _apply_cache_breakpoints,
    _cache_anchor,
)
from iag_sim.cua.backend import build_backend
from iag_sim.cua.openai_backend import OpenAIBackend

_BASE = {
    "MUREX_BEFORE_URL": "https://before",
    "MUREX_AFTER_URL": "https://after",
    "MUREX_USER": "u",
    "MUREX_PASS": "p",
}


def _settings(monkeypatch, **extra) -> Settings:
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_REGION",
              "AWS_BEARER_TOKEN_BEDROCK", "CUA_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**_BASE, **extra}.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


# --- factory selects the right backend per provider ---


def test_build_backend_openai(monkeypatch):
    s = _settings(monkeypatch, CUA_PROVIDER="openai", OPENAI_API_KEY="sk-x")
    assert isinstance(build_backend(s), OpenAIBackend)


def test_build_backend_anthropic(monkeypatch):
    s = _settings(monkeypatch, CUA_PROVIDER="anthropic", ANTHROPIC_API_KEY="ak-x")
    b = build_backend(s)
    assert isinstance(b, AnthropicBackend)
    assert b.tool_version == "computer_20251124"
    assert b.beta_flag == "computer-use-2025-11-24"


def test_build_backend_bedrock(monkeypatch):
    s = _settings(
        monkeypatch, CUA_PROVIDER="bedrock",
        AWS_REGION="eu-west-1", AWS_BEARER_TOKEN_BEDROCK="ABSK-token",
        CUA_MODEL="eu.anthropic.claude-opus-4-8",
    )
    b = build_backend(s)
    assert isinstance(b, AnthropicBackend)
    assert b.model == "eu.anthropic.claude-opus-4-8"


# --- the Anthropic loop, driven by a fake client ---


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeStream:
    """Async context manager mirroring the SDK's stream(...) return: `async with`
    yields self, and get_final_message() returns the scripted assembled Message."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._response


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        # stream() is a SYNC call returning an async CM (matches the real SDK).
        self.calls.append(kwargs)
        return _FakeStream(self._responses.pop(0))


class _FakeBeta:
    def __init__(self, messages):
        self.messages = messages


class _FakeClient:
    def __init__(self, responses):
        self.beta = _FakeBeta(_FakeMessages(responses))


class _FakeComputer:
    def __init__(self):
        self.calls: list[tuple] = []

    async def screenshot(self):
        self.calls.append(("screenshot",))
        return "B64"

    async def click(self, x, y, button="left", keys=None):
        self.calls.append(("click", x, y, button, keys))

    async def double_click(self, x, y, button="left", keys=None):
        self.calls.append(("double_click", x, y, button, keys))

    async def move(self, x, y):
        self.calls.append(("move", x, y))

    async def drag(self, path, keys=None):
        self.calls.append(("drag", path, keys))

    async def scroll(self, x, y, scroll_x=0, scroll_y=0):
        self.calls.append(("scroll", x, y, scroll_x, scroll_y))

    async def type(self, text):
        self.calls.append(("type", text))

    async def keypress(self, keys):
        self.calls.append(("keypress", keys))

    async def wait(self, ms=1000):
        self.calls.append(("wait", ms))


def _backend(client) -> AnthropicBackend:
    return AnthropicBackend(
        client=client, model="claude-x",
        tool_version="computer_20251124", beta_flag="computer-use-2025-11-24",
    )


async def test_loop_screenshot_then_done():
    client = _FakeClient([
        _Resp(
            content=[_Block(type="tool_use", id="toolu_1", name="computer",
                            input={"action": "screenshot"})],
            stop_reason="tool_use",
        ),
        _Resp(content=[_Block(type="text", text="DONE")], stop_reason="end_turn"),
    ])
    comp = _FakeComputer()
    result = await _backend(client).run(
        computer=comp, task="do it", display_width=1024, display_height=768, max_turns=5,
    )

    assert result.completed is True
    assert result.final_text == "DONE"
    assert result.turns == 2
    assert ("screenshot",) in comp.calls

    msgs = client.beta.messages
    # first call carried the tool def + beta flag
    first = msgs.calls[0]
    assert first["betas"] == ["computer-use-2025-11-24"]
    tool = first["tools"][0]
    assert tool["type"] == "computer_20251124"
    assert tool["display_width_px"] == 1024 and tool["display_height_px"] == 768

    # a user tool_result image referencing the tool_use id was fed back. (The loop
    # reuses ONE messages list, so inspect the final accumulated conversation.)
    all_msgs = msgs.calls[-1]["messages"]
    tool_results = [
        b
        for m in all_msgs
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    block = tool_results[0]
    assert block["tool_use_id"] == "toolu_1"
    assert block["content"][0]["type"] == "image"
    assert block["content"][0]["source"]["data"] == "B64"


async def test_loop_click_action_dispatched():
    client = _FakeClient([
        _Resp(
            content=[_Block(type="tool_use", id="toolu_a", name="computer",
                            input={"action": "left_click", "coordinate": [3, 4]})],
            stop_reason="tool_use",
        ),
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    comp = _FakeComputer()
    await _backend(client).run(
        computer=comp, task="click", display_width=800, display_height=600, max_turns=5,
    )
    assert ("click", 3, 4, "left", None) in comp.calls
    assert ("screenshot",) in comp.calls


async def test_loop_bedrock_fallback_uses_create_stream(monkeypatch):
    """The Bedrock beta resource lacks the high-level .stream() helper, so the loop
    must fall back to create(stream=True) wrapped in the SDK accumulator manager."""
    import iag_sim.cua.anthropic_backend as ab

    final = _Resp(content=[_Block(type="text", text="DONE")], stop_reason="end_turn")

    class _BedrockMessages:  # like AsyncAnthropicBedrock.beta.messages: create only
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):  # returns an awaitable (the raw-stream stand-in)
            self.calls.append(kwargs)

            async def _raw():
                return final

            return _raw()

    class _BedrockClient:
        def __init__(self):
            self.beta = _FakeBeta(_BedrockMessages())

    class _FakeMgr:  # stands in for BetaAsyncMessageStreamManager
        def __init__(self, api_request, *, output_format=None):
            self._req = api_request

        async def __aenter__(self):
            self._final = await self._req  # exercises create()'s coroutine
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return self._final

    monkeypatch.setattr(ab, "BetaAsyncMessageStreamManager", _FakeMgr)

    client = _BedrockClient()
    assert not hasattr(client.beta.messages, "stream")  # forces the fallback branch
    result = await _backend(client).run(
        computer=_FakeComputer(), task="go",
        display_width=800, display_height=600, max_turns=3,
    )
    assert result.final_text == "DONE"
    call = client.beta.messages.calls[0]
    assert call["stream"] is True
    assert call["betas"] == ["computer-use-2025-11-24"]


# --- reasoning effort -> adaptive thinking + output_config.effort wiring ---


def _effort_backend(client, *, tool_version="computer_20251124", **kw):
    return AnthropicBackend(
        client=client, model="claude-x", tool_version=tool_version,
        beta_flag="computer-use-2025-11-24", max_tokens=4096, **kw,
    )


async def _run_one(backend):
    await backend.run(
        computer=_FakeComputer(), task="t",
        display_width=800, display_height=600, max_turns=3,
    )


async def test_effort_newest_uses_adaptive_thinking_and_output_config():
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(client, reasoning_effort="high"))
    call = client.beta.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    # high effort floors max_tokens at 16000 to leave thinking room
    assert call["max_tokens"] == 16000


async def test_effort_max_passes_through_on_newest():
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(client, reasoning_effort="max"))
    call = client.beta.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "max"}
    assert call["max_tokens"] == 32000


async def test_effort_budget_mode_for_old_tool_version():
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(
        client, tool_version="computer_20250124", reasoning_effort="high",
    ))
    call = client.beta.messages.calls[0]
    # old gen: manual budget_tokens, no output_config (Sonnet 4.5 has no effort param)
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "output_config" not in call
    assert call["max_tokens"] == 16384 + 4096


async def test_no_reasoning_effort_omits_thinking_and_effort():
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(client))
    call = client.beta.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call
    assert call["max_tokens"] == 4096


async def test_effort_none_disables_thinking_on_adaptive():
    # `none` is an OpenAI-only tier; the adaptive API 400s on it. The loop must
    # treat it as reasoning-OFF and send no thinking/output_config (regression:
    # effort=none previously forwarded output_config={"effort":"none"} -> 400).
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(client, reasoning_effort="none"))
    call = client.beta.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call
    assert call["max_tokens"] == 4096


async def test_effort_none_disables_thinking_on_old_tool_version():
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(
        client, tool_version="computer_20250124", reasoning_effort="none",
    ))
    call = client.beta.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call


async def test_effort_minimal_not_forwarded_on_adaptive():
    # `minimal` (sub-low, OpenAI-only) is not an adaptive effort -> no thinking sent.
    client = _FakeClient([
        _Resp(content=[_Block(type="text", text="ok")], stop_reason="end_turn"),
    ])
    await _run_one(_effort_backend(client, reasoning_effort="minimal"))
    call = client.beta.messages.calls[0]
    assert "thinking" not in call
    assert "output_config" not in call


# --- prompt caching (cache_control breakpoints) ---


def _user_img(tid="t", *, stub=False):
    inner = (
        {"type": "text", "text": "[screenshot pruned to save tokens]"}
        if stub
        else {"type": "image",
              "source": {"type": "base64", "media_type": "image/png", "data": "B"}}
    )
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": [inner]}]}


def _assistant(text="x"):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_cache_anchor_below_capacity_targets_newest():
    # 2 real screenshots, keep=3 -> nothing stubs next turn -> anchor the newest.
    msgs = [{"role": "user", "content": "task"},
            _assistant(), _user_img("a"),
            _assistant(), _user_img("b")]
    assert _cache_anchor(msgs, keep=3) == 4  # the newest user-image message


def test_cache_anchor_pruning_off_targets_newest():
    msgs = [{"role": "user", "content": "task"},
            _assistant(), _user_img("a"),
            _assistant(), _user_img("b")]
    assert _cache_anchor(msgs, keep=0) == 4


def test_cache_anchor_at_capacity_targets_before_oldest_real():
    # steady state: 2 stubbed + 3 real, keep=3. Oldest real screenshot is idx 6;
    # it stubs next turn, so the anchor is the last user msg strictly before it.
    msgs = [
        {"role": "user", "content": "task"},        # 0
        _assistant(), _user_img("s0", stub=True),   # 1, 2
        _assistant(), _user_img("s1", stub=True),   # 3, 4
        _assistant(), _user_img("r0"),              # 5, 6  <- oldest real
        _assistant(), _user_img("r1"),              # 7, 8
        _assistant(), _user_img("r2"),              # 9, 10
    ]
    assert _cache_anchor(msgs, keep=3) == 4


def test_apply_breakpoints_single_rolling_anchor():
    msgs = [
        {"role": "user", "content": "task"},
        _assistant(), _user_img("s0", stub=True),
        _assistant(), _user_img("s1", stub=True),
        _assistant(), _user_img("r0"),
        _assistant(), _user_img("r1"),
        _assistant(), _user_img("r2"),
    ]
    _apply_cache_breakpoints(msgs, keep=3)
    assert msgs[4]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    marked = [b for m in msgs if isinstance(m.get("content"), list)
              for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
    assert len(marked) == 1                      # exactly one rolling breakpoint
    # re-applying clears the prior one (never accumulates past the 4-bp cap)
    _apply_cache_breakpoints(msgs, keep=3)
    marked2 = [b for m in msgs if isinstance(m.get("content"), list)
               for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
    assert len(marked2) == 1


async def test_loop_sets_cache_control_when_enabled():
    client = _FakeClient([
        _Resp(content=[_Block(type="tool_use", id="t1", name="computer",
                              input={"action": "screenshot"})], stop_reason="tool_use"),
        _Resp(content=[_Block(type="text", text="DONE")], stop_reason="end_turn"),
    ])
    await _backend(client).run(
        computer=_FakeComputer(), task="go",
        display_width=800, display_height=600, max_turns=5,
    )
    calls = client.beta.messages.calls
    # static breakpoint on the tool def
    assert calls[0]["tools"][0]["cache_control"] == {"type": "ephemeral"}
    # one rolling conversation breakpoint, on a user tool_result block
    conv_bp = [b for m in calls[-1]["messages"] if isinstance(m.get("content"), list)
               for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
    assert len(conv_bp) == 1
    assert conv_bp[0]["type"] == "tool_result"


async def test_loop_no_cache_control_when_disabled():
    client = _FakeClient([
        _Resp(content=[_Block(type="tool_use", id="t1", name="computer",
                              input={"action": "screenshot"})], stop_reason="tool_use"),
        _Resp(content=[_Block(type="text", text="DONE")], stop_reason="end_turn"),
    ])
    b = AnthropicBackend(
        client=client, model="claude-x", tool_version="computer_20251124",
        beta_flag="computer-use-2025-11-24", prompt_cache=False,
    )
    await b.run(computer=_FakeComputer(), task="go",
                display_width=800, display_height=600, max_turns=5)
    for call in client.beta.messages.calls:
        assert "cache_control" not in call["tools"][0]
        for m in call["messages"]:
            if isinstance(m.get("content"), list):
                assert all(not (isinstance(blk, dict) and "cache_control" in blk)
                           for blk in m["content"])


class _Usage:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _CapTracer:
    def __init__(self):
        self.events: list[tuple] = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def close(self):
        pass


async def test_usage_event_emitted_with_cache_tokens():
    r1 = _Resp(content=[_Block(type="tool_use", id="t1", name="computer",
                               input={"action": "screenshot"})], stop_reason="tool_use")
    r1.usage = _Usage(input_tokens=1000, output_tokens=50,
                      cache_creation_input_tokens=900, cache_read_input_tokens=0)
    r2 = _Resp(content=[_Block(type="text", text="DONE")], stop_reason="end_turn")
    r2.usage = _Usage(input_tokens=1100, output_tokens=20,
                      cache_creation_input_tokens=0, cache_read_input_tokens=900)
    cap = _CapTracer()
    await _backend(_FakeClient([r1, r2])).run(
        computer=_FakeComputer(), task="go",
        display_width=800, display_height=600, max_turns=5, tracer=cap,
    )
    usage = [f for (k, f) in cap.events if k == "usage"]
    assert len(usage) == 2
    assert usage[0]["cache_write"] == 900 and usage[0]["cache_read"] == 0
    assert usage[1]["cache_read"] == 900 and usage[1]["input"] == 1100
