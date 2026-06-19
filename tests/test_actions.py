"""Unit tests for the computer-use action dispatcher. No browser required."""

from __future__ import annotations

import pytest

from iag_sim.cua.actions import _MAX_WAIT_MS, UnknownActionError, dispatch, wait_duration_ms


class FakeComputer:
    """Records calls so we can assert the dispatcher mapped correctly."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def screenshot(self) -> str:
        self.calls.append(("screenshot",))
        return "BASE64PNG"

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


async def test_screenshot_returns_b64():
    c = FakeComputer()
    out = await dispatch(c, {"type": "screenshot"})
    assert out == "BASE64PNG"
    assert c.calls == [("screenshot",)]


async def test_click_with_modifiers():
    c = FakeComputer()
    out = await dispatch(
        c, {"type": "click", "x": 10, "y": 20, "button": "left", "keys": ["CTRL"]}
    )
    assert out is None
    assert c.calls == [("click", 10, 20, "left", ["CTRL"])]


async def test_type_and_keypress():
    c = FakeComputer()
    await dispatch(c, {"type": "type", "text": "hello"})
    await dispatch(c, {"type": "keypress", "keys": ["ENTER"]})
    assert c.calls == [("type", "hello"), ("keypress", ["ENTER"])]


async def test_keypress_single_key_fallback():
    c = FakeComputer()
    await dispatch(c, {"type": "keypress", "key": "ENTER"})
    assert c.calls == [("keypress", ["ENTER"])]


async def test_scroll_camel_and_snake():
    c = FakeComputer()
    await dispatch(c, {"type": "scroll", "x": 1, "y": 2, "scrollY": 5})
    assert c.calls == [("scroll", 1, 2, 0, 5)]


async def test_drag_path():
    c = FakeComputer()
    await dispatch(c, {"type": "drag", "path": [[0, 0], [5, 5]]})
    assert c.calls == [("drag", [[0, 0], [5, 5]], None)]


async def test_drag_path_ga_dict_form():
    # GA computer-use sends path points as {"x":..,"y":..} dicts. The dispatcher
    # must normalize to [x, y] int pairs — regression for the scrollbar drag that
    # crashed with `int() ... 'x'` (a dict unpacked to its keys).
    c = FakeComputer()
    await dispatch(
        c,
        {"type": "drag", "path": [{"x": 1142, "y": 112}, {"x": 1142, "y": 706}], "keys": None},
    )
    assert c.calls == [("drag", [[1142, 112], [1142, 706]], None)]


async def test_unknown_action_raises():
    c = FakeComputer()
    with pytest.raises(UnknownActionError):
        await dispatch(c, {"type": "teleport"})


def test_wait_duration_default_when_unspecified():
    assert wait_duration_ms({"type": "wait"}) == 1000


def test_wait_duration_passthrough_under_cap():
    assert wait_duration_ms({"type": "wait", "duration_ms": 4000}) == 4000
    assert wait_duration_ms({"type": "wait", "ms": 4000}) == 4000


def test_wait_duration_clamped_to_max():
    # Model can request an arbitrarily long sleep; harness caps it so one wait
    # can't burn a whole turn waiting forever for sessions to reap.
    assert wait_duration_ms({"type": "wait", "duration_ms": 60_000}) == _MAX_WAIT_MS
    assert _MAX_WAIT_MS == 15_000


def test_wait_duration_at_cap_unchanged():
    assert wait_duration_ms({"type": "wait", "duration_ms": 15_000}) == 15_000


async def test_dispatch_wait_clamps_before_calling_computer():
    c = FakeComputer()
    await dispatch(c, {"type": "wait", "duration_ms": 30_000})
    assert c.calls == [("wait", _MAX_WAIT_MS)]
