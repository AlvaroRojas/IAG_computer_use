"""Dispatch a single computer-use action to a `Computer` implementation.

Maps the model's action dict (per the Responses API computer-use contract) onto
harness method calls. Pure routing — testable with a fake Computer.
"""

from __future__ import annotations

from .base import Action, Computer


class UnknownActionError(ValueError):
    """Raised for an action type we don't know how to execute."""


_DEFAULT_WAIT_MS = 1000


def wait_duration_ms(action: Action) -> int:
    """Resolve how long a `wait` action pauses (ms). The model may send
    `duration_ms` or `ms`; absent both it's the default. Single source of truth
    so the executed wait and the traced wait can never disagree."""
    return int(action.get("duration_ms", action.get("ms", _DEFAULT_WAIT_MS)))


def _coord_path(raw) -> list[list[int]]:
    """Normalize a drag path to integer [x, y] pairs. Accepts the GA dict form
    [{"x":..,"y":..}, ...] and the bare-pair form [[x, y], ...]."""
    path: list[list[int]] = []
    for p in raw or []:
        if isinstance(p, dict):
            path.append([int(p["x"]), int(p["y"])])
        else:
            path.append([int(p[0]), int(p[1])])
    return path


async def dispatch(computer: Computer, action: Action) -> str | None:
    """Execute one action. Returns a base64 screenshot for `screenshot`
    actions, otherwise None. Unknown types raise UnknownActionError."""
    atype = action.get("type")

    if atype == "screenshot":
        return await computer.screenshot()

    if atype == "click":
        await computer.click(
            int(action["x"]),
            int(action["y"]),
            action.get("button", "left"),
            action.get("keys"),
        )
        return None

    if atype == "double_click":
        await computer.double_click(
            int(action["x"]),
            int(action["y"]),
            action.get("button", "left"),
            action.get("keys"),
        )
        return None

    if atype == "move":
        await computer.move(int(action["x"]), int(action["y"]))
        return None

    if atype == "drag":
        # GA computer-use sends path as a list of {"x":..,"y":..} dicts; the
        # Computer.drag contract is list[list[int]] (x,y pairs). Normalize so a
        # dict point isn't unpacked as its keys ('x','y') -> int('x') ValueError.
        await computer.drag(_coord_path(action.get("path")), action.get("keys"))
        return None

    if atype == "scroll":
        await computer.scroll(
            int(action["x"]),
            int(action["y"]),
            int(action.get("scrollX", action.get("scroll_x", 0))),
            int(action.get("scrollY", action.get("scroll_y", 0))),
        )
        return None

    if atype == "type":
        await computer.type(action["text"])
        return None

    if atype == "keypress":
        # GA returns keys[]; preview sometimes a single key.
        keys = action.get("keys") or ([action["key"]] if "key" in action else [])
        await computer.keypress(keys)
        return None

    if atype == "wait":
        await computer.wait(wait_duration_ms(action))
        return None

    raise UnknownActionError(f"unknown action type: {atype!r}")
