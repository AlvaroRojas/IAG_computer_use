"""Unit tests for the Anthropic -> canonical action translator. No network."""

from __future__ import annotations

import pytest

from iag_sim.cua.actions import dispatch
from iag_sim.cua.anthropic_actions import UnknownAnthropicActionError, to_canonical


def test_screenshot():
    assert to_canonical({"action": "screenshot"}) == [{"type": "screenshot"}]


def test_left_click_coordinate():
    assert to_canonical({"action": "left_click", "coordinate": [10, 20]}) == [
        {"type": "click", "x": 10, "y": 20, "button": "left"}
    ]


def test_right_and_middle_click():
    assert to_canonical({"action": "right_click", "coordinate": [1, 2]}) == [
        {"type": "click", "x": 1, "y": 2, "button": "right"}
    ]
    assert to_canonical({"action": "middle_click", "coordinate": [3, 4]}) == [
        {"type": "click", "x": 3, "y": 4, "button": "middle"}
    ]


def test_click_with_text_modifiers():
    assert to_canonical(
        {"action": "left_click", "coordinate": [5, 6], "text": "ctrl+shift"}
    ) == [{"type": "click", "x": 5, "y": 6, "button": "left", "keys": ["CTRL", "SHIFT"]}]


def test_double_click_no_modifiers_has_no_keys():
    assert to_canonical({"action": "double_click", "coordinate": [7, 8]}) == [
        {"type": "double_click", "x": 7, "y": 8}
    ]


def test_triple_click_expands_to_three_clicks():
    assert to_canonical({"action": "triple_click", "coordinate": [9, 9]}) == [
        {"type": "click", "x": 9, "y": 9, "button": "left"},
        {"type": "click", "x": 9, "y": 9, "button": "left"},
        {"type": "click", "x": 9, "y": 9, "button": "left"},
    ]


def test_mouse_move():
    assert to_canonical({"action": "mouse_move", "coordinate": [11, 12]}) == [
        {"type": "move", "x": 11, "y": 12}
    ]


def test_left_click_drag():
    assert to_canonical(
        {"action": "left_click_drag", "start_coordinate": [0, 0], "coordinate": [5, 5]}
    ) == [{"type": "drag", "path": [[0, 0], [5, 5]]}]


def test_scroll_down_default_one_per_click():
    assert to_canonical(
        {"action": "scroll", "coordinate": [1, 2], "scroll_direction": "down", "scroll_amount": 3}
    ) == [{"type": "scroll", "x": 1, "y": 2, "scroll_x": 0, "scroll_y": 3}]


def test_scroll_up_is_negative():
    assert to_canonical(
        {"action": "scroll", "coordinate": [1, 2], "scroll_direction": "up", "scroll_amount": 2}
    ) == [{"type": "scroll", "x": 1, "y": 2, "scroll_x": 0, "scroll_y": -2}]


def test_scroll_right_passes_notches_through():
    # scroll_amount is already in wheel notches -> canonical delta 1:1, no scaling.
    assert to_canonical(
        {"action": "scroll", "coordinate": [1, 2], "scroll_direction": "right", "scroll_amount": 2}
    ) == [{"type": "scroll", "x": 1, "y": 2, "scroll_x": 2, "scroll_y": 0}]


def test_key_combo_normalized_letters_pass_through():
    assert to_canonical({"action": "key", "text": "ctrl+s"}) == [
        {"type": "keypress", "keys": ["CTRL", "s"]}
    ]


def test_key_keysyms_normalized():
    assert to_canonical({"action": "key", "text": "Return"}) == [
        {"type": "keypress", "keys": ["ENTER"]}
    ]
    assert to_canonical({"action": "key", "text": "Page_Down"}) == [
        {"type": "keypress", "keys": ["PAGEDOWN"]}
    ]
    assert to_canonical({"action": "key", "text": "Up"}) == [
        {"type": "keypress", "keys": ["UP"]}
    ]


def test_type():
    assert to_canonical({"action": "type", "text": "hello"}) == [
        {"type": "type", "text": "hello"}
    ]


def test_wait_seconds_to_ms():
    assert to_canonical({"action": "wait", "duration": 2}) == [
        {"type": "wait", "duration_ms": 2000}
    ]


def test_hold_key_approximated_as_keypress():
    assert to_canonical({"action": "hold_key", "text": "shift", "duration": 1.5}) == [
        {"type": "keypress", "keys": ["SHIFT"]}
    ]


def test_noop_actions_collapse_to_empty():
    for a in ("cursor_position", "left_mouse_down", "left_mouse_up"):
        assert to_canonical({"action": a}) == []


def test_unknown_action_raises():
    with pytest.raises(UnknownAnthropicActionError):
        to_canonical({"action": "teleport"})


def test_missing_action_raises():
    with pytest.raises(UnknownAnthropicActionError):
        to_canonical({"coordinate": [1, 2]})


# --- Integration: the canonical output must be dispatchable to a Computer. ---


class FakeComputer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def screenshot(self) -> str:
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


async def test_translated_click_dispatches():
    c = FakeComputer()
    for ca in to_canonical({"action": "left_click", "coordinate": [10, 20], "text": "ctrl"}):
        await dispatch(c, ca)
    assert c.calls == [("click", 10, 20, "left", ["CTRL"])]


async def test_translated_scroll_dispatches():
    c = FakeComputer()
    for ca in to_canonical(
        {"action": "scroll", "coordinate": [1, 2], "scroll_direction": "down", "scroll_amount": 5}
    ):
        await dispatch(c, ca)
    assert c.calls == [("scroll", 1, 2, 0, 5)]
