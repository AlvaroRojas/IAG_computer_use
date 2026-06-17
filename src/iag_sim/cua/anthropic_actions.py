"""Translate Anthropic computer-use actions into the canonical action dicts that
`cua/actions.py:dispatch` already understands, so ONE execution layer serves both
the OpenAI and the Anthropic loops.

Anthropic (`computer_20250124` / `computer_20251124`) returns one action dict per
`tool_use`, e.g. `{"action": "left_click", "coordinate": [x, y]}`. Each maps to
ZERO OR MORE canonical dicts (the OpenAI-shaped vocabulary in `cua/actions.py`):
most are 1:1, `triple_click` expands to three clicks, and query/low-level events
(`cursor_position`, `left_mouse_down/up`) collapse to nothing.

Key handling: Anthropic emits xdotool keysyms (`Return`, `ctrl`, `Page_Down`,
`Up`, `super`). The Computer layer maps keys via `_KEY_MAP`/`_XKEY` which uppercase
the token and fall back to it unchanged, so we normalize the well-known keysyms to
that uppercase vocabulary and leave anything else (e.g. a literal `s`) untouched.
"""

from __future__ import annotations

from .base import Action


class UnknownAnthropicActionError(ValueError):
    """Raised for an Anthropic action type we don't know how to translate."""


# xdotool keysym (lower-cased) -> canonical key name accepted by the Computer layer.
_KEYSYM_TO_CANONICAL = {
    "return": "ENTER",
    "enter": "ENTER",
    "tab": "TAB",
    "space": "SPACE",
    "backspace": "BACKSPACE",
    "delete": "DELETE",
    "escape": "ESCAPE",
    "esc": "ESCAPE",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "page_up": "PAGEUP",
    "prior": "PAGEUP",
    "page_down": "PAGEDOWN",
    "next": "PAGEDOWN",
    "home": "HOME",
    "end": "END",
    "ctrl": "CTRL",
    "control": "CTRL",
    "alt": "ALT",
    "shift": "SHIFT",
    "super": "META",
    "cmd": "META",
    "meta": "META",
}

_CLICK_BUTTON = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
}

# scroll_direction -> (x_sign, y_sign) applied to the magnitude.
_SCROLL_SIGN = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _norm_key(token: str) -> str:
    return _KEYSYM_TO_CANONICAL.get(token.strip().lower(), token)


def _split_keys(text: str) -> list[str]:
    """Split a key combo like 'ctrl+s' into normalized canonical key names."""
    return [_norm_key(t) for t in text.split("+") if t]


def _xy(coordinate) -> tuple[int, int]:
    return int(coordinate[0]), int(coordinate[1])


def _modifiers(action: Action) -> list[str]:
    """Optional held-modifier keys passed via the action's `text` field
    (shift/ctrl/alt/super on click & scroll actions)."""
    text = action.get("text")
    return _split_keys(text) if text else []


def to_canonical(action: Action) -> list[Action]:
    """Map one Anthropic action dict to a list of canonical action dicts.

    Anthropic's `scroll_amount` is already in wheel NOTCHES, which is exactly the
    canonical scroll unit: the thick channel consumes notches 1:1 (xdotool click
    repeats) and the browser Computer scales notches -> pixels. So it passes
    straight through — no per-channel knob needed.
    """
    atype = action.get("action")
    if atype is None:
        raise UnknownAnthropicActionError(f"missing 'action': {action!r}")

    if atype == "screenshot":
        return [{"type": "screenshot"}]

    if atype in _CLICK_BUTTON:
        x, y = _xy(action["coordinate"])
        out: Action = {"type": "click", "x": x, "y": y, "button": _CLICK_BUTTON[atype]}
        mods = _modifiers(action)
        if mods:
            out["keys"] = mods
        return [out]

    if atype == "double_click":
        x, y = _xy(action["coordinate"])
        out = {"type": "double_click", "x": x, "y": y}
        mods = _modifiers(action)
        if mods:
            out["keys"] = mods
        return [out]

    if atype == "triple_click":
        # Computer has no native triple-click; emulate as three left clicks.
        x, y = _xy(action["coordinate"])
        click: Action = {"type": "click", "x": x, "y": y, "button": "left"}
        mods = _modifiers(action)
        if mods:
            click["keys"] = mods
        return [dict(click), dict(click), dict(click)]

    if atype == "mouse_move":
        x, y = _xy(action["coordinate"])
        return [{"type": "move", "x": x, "y": y}]

    if atype == "left_click_drag":
        sx, sy = _xy(action["start_coordinate"])
        ex, ey = _xy(action["coordinate"])
        out = {"type": "drag", "path": [[sx, sy], [ex, ey]]}
        mods = _modifiers(action)
        if mods:
            out["keys"] = mods
        return [out]

    if atype == "scroll":
        x, y = _xy(action["coordinate"])
        direction = action.get("scroll_direction", "down")
        amount = int(action.get("scroll_amount", 0))
        sx_sign, sy_sign = _SCROLL_SIGN.get(direction, (0, 0))
        return [{
            "type": "scroll", "x": x, "y": y,
            "scroll_x": sx_sign * amount, "scroll_y": sy_sign * amount,
        }]

    if atype == "key":
        return [{"type": "keypress", "keys": _split_keys(action.get("text", ""))}]

    if atype == "hold_key":
        # No timed hold in the Computer protocol; approximate as a keypress.
        return [{"type": "keypress", "keys": _split_keys(action.get("text", ""))}]

    if atype == "type":
        return [{"type": "type", "text": action.get("text", "")}]

    if atype == "wait":
        duration = float(action.get("duration", 1.0))
        return [{"type": "wait", "duration_ms": int(duration * 1000)}]

    if atype in ("cursor_position", "left_mouse_down", "left_mouse_up"):
        # Query / low-level button events the Computer protocol does not model.
        return []

    raise UnknownAnthropicActionError(f"unknown Anthropic action: {atype!r}")
