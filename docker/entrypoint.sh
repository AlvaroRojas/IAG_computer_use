#!/bin/bash
# Container entrypoint for the Murex thick channel.
# Starts a headless X server + window manager, then auto-launches the Mx.3
# client so the container comes up "ready to drive" for the computer-use loop.
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
APP_DIR="${MUREX_APP_DIR:-/opt/murex}"
EXPORT_DIR="${MUREX_CONTAINER_EXPORT_DIR:-/exports}"
mkdir -p "$EXPORT_DIR"

# Xvfb needs WxHxDEPTH. The harness passes SCREEN_GEOMETRY as WxH only
# (config.py: "{width}x{height}") -> append a colour depth if absent.
GEO="${SCREEN_GEOMETRY:-1280x800}"
case "$GEO" in
  *x*x*) ;;                 # already has depth
  *)     GEO="${GEO}x24" ;;
esac

echo "[entrypoint] starting Xvfb on $DISPLAY @ $GEO"
Xvfb "$DISPLAY" -screen 0 "$GEO" -ac -nolisten tcp +extension RANDR \
  >/var/log/xvfb.log 2>&1 &

# Wait until the display answers before launching GUI processes.
for _ in $(seq 1 50); do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then break; fi
  sleep 0.2
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  echo "[entrypoint] FATAL: Xvfb did not come up on $DISPLAY" >&2
  cat /var/log/xvfb.log >&2 || true
  exit 1
fi

# Headless desktop background. fluxbox's default style paints a wallpaper via
# `fbsetbg` (ubuntu-light.png); in this minimal image that fails and pops an
# xmessage error dialog OVER the UI — poison for screenshots / the vision model.
# Kill it at the source: `background: none` in the overlay disables fluxbox's
# wallpaper handling entirely, and we paint a flat colour ourselves via fbsetroot.
mkdir -p "$HOME/.fluxbox"
printf 'background: none\n' > "$HOME/.fluxbox/overlay"
printf 'session.screen0.rootCommand: fbsetroot -solid #1b1d23\n' > "$HOME/.fluxbox/init"
fbsetroot -solid '#1b1d23' 2>/dev/null || true

# Lightweight WM so Swing frames/dialogs get focus + sane stacking (Murex opens
# many windows); also keeps xdotool focus behaviour predictable.
fluxbox >/var/log/fluxbox.log 2>&1 &

# Belt-and-suspenders: after fluxbox settles, re-assert the flat root and close
# any fbsetbg error dialog that slipped through before the overlay applied.
( sleep 3; fbsetroot -solid '#1b1d23' 2>/dev/null
  xdotool search --name xmessage windowkill 2>/dev/null
  xdotool search --class xmessage windowkill 2>/dev/null ) >/dev/null 2>&1 &

# Optional live view (debug): x11vnc on 5900 + noVNC web on 6080. Off by default;
# the computer-use harness drives via import/xdotool and does not need it.
if [ "${ENABLE_VNC:-false}" = "true" ]; then
  echo "[entrypoint] starting x11vnc (:5900) + noVNC web (:6080)"
  x11vnc -display "$DISPLAY" -forever -shared -nopw -rfbport 5900 \
    -bg -o /var/log/x11vnc.log >/dev/null 2>&1 || true
  NOVNC_DIR=/usr/share/novnc
  if [ -f "$NOVNC_DIR/vnc.html" ] && [ ! -e "$NOVNC_DIR/index.html" ]; then
    ln -s vnc.html "$NOVNC_DIR/index.html" 2>/dev/null || true
  fi
  websockify --web="$NOVNC_DIR" 6080 localhost:5900 >/var/log/websockify.log 2>&1 &
fi

# Thick login CANNOT be scripted -> the client always boots to its login screen
# and the computer-use model authenticates + selects the group itself. Any
# MUREX_USER/MUREX_PASS are deliberately NOT consumed here (and the harness does
# not even pass them in the default MUREX_LLM_LOGIN=true mode).
if [ "${MUREX_LLM_LOGIN:-true}" != "true" ]; then
  echo "[entrypoint] WARNING: thick login cannot be automated by the entrypoint;" \
       "booting to the login screen for the model. Set MUREX_LLM_LOGIN=true to silence." >&2
fi

echo "[entrypoint] launching Murex client (target=${MUREX_ENV_TARGET:-<vendor default>})"
cd "$APP_DIR"

# Run in the background (not exec) so the VNC server keeps serving and we can
# hold the container open on error for inspection.
/usr/local/bin/launch-client.sh &
CLIENT_PID=$!
trap 'kill "$CLIENT_PID" 2>/dev/null' TERM INT
wait "$CLIENT_PID"
RC=$?
echo "[entrypoint] client exited rc=$RC"

# Debug convenience: when viewing via VNC, keep the container up after a client
# crash so the error dialog / logs can be inspected. Never holds open in normal
# (non-VNC) harness runs.
if [ "${ENABLE_VNC:-false}" = "true" ] && [ "${KEEP_ALIVE_ON_EXIT:-true}" = "true" ]; then
  echo "[entrypoint] holding container open for inspection (KEEP_ALIVE_ON_EXIT=true); docker stop to end."
  tail -f /dev/null
fi
exit "$RC"
