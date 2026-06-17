# Murex thick-client container

Headless Linux image that runs the **Mx.3 Java Swing client** (from `../murex/`)
under Xvfb and **auto-launches it on container start**, so the computer-use loop
(`src/iag_sim/harness/docker.py`) can screenshot it with ImageMagick `import` and
drive it with `xdotool`. One container per trade.

## Files

| File | Role |
|------|------|
| `Dockerfile.thick` | Ubuntu 22.04 + Xvfb/fluxbox + ImageMagick/xdotool + Java 8 + Murex payload |
| `entrypoint.sh` | Starts Xvfb on `$DISPLAY` @ `$SCREEN_GEOMETRY`, starts the WM, launches the client |
| `launch-client.sh` | Parameterized clone of `murex/client.sh`; reads the fileserver target from the env |
| `java.policy` | Permissive policy (vendor scripts reference it; not shipped in `murex/`) |

## Build

Context **must** be the repo root (the image needs `murex/` and `docker/`):

```powershell
docker build -f docker/Dockerfile.thick -t murex-thick:latest .
```

The Java 8 runtime is pulled from `murex/jdk-8u202-linux-x64.tar.gz` (auto-extracted
by `ADD`). The bulky pre-extracted JDK dirs and the Windows JDK are excluded via
`.dockerignore`. First build is large (~800 MB+: 412 MB of jars + the JDK).

## How the app uses it

Set in `.env`:

```ini
MUREX_CHANNEL=thick
MUREX_DOCKER_IMAGE=murex-thick:latest
MUREX_LLM_LOGIN=true            # required: thick login cannot be scripted
MUREX_BEFORE_URL=10.10.0.93:20001   # fileserver target -> $MUREX_ENV_TARGET
MUREX_AFTER_URL=10.10.0.94:20001    # the "after" backend (set to the real host)
```

`docker.py` then runs, per trade:

```
docker run -d --rm \
  -e MUREX_ENV_TARGET=<url_for(env)> \
  -e MUREX_TRADE_ID=<id> \
  -e DISPLAY=:99 -e SCREEN_GEOMETRY=1280x800 \
  -v <host_export>:/exports \
  murex-thick:latest
```

`MUREX_USER`/`MUREX_PASS` are **not** passed in `MUREX_LLM_LOGIN=true` mode — the
client boots to its login screen and the model authenticates + picks the group.

### Fileserver target

`launch-client.sh` resolves the Murex fileserver in this order:

1. explicit `MXJ_FILESERVER_HOST` / `MXJ_FILESERVER_PORT`
2. `$MUREX_ENV_TARGET` parsed as `host`, `host:port`, or `scheme://host:port/path`
3. vendor default `10.10.0.93:20001`

So a before/after diff across two backends = two different `MUREX_*_URL` values.

## Networking

The client streams most of its jars from the fileserver at runtime
(`java.rmi.server.codebase=http://<host>:<port>/...`) and connects to the Murex
backend. **The container must be able to reach that host** (10.10.0.93:20001 for
jar streaming, plus the RMI session ports in the vendor config). On Docker Desktop / Windows that usually means a VPN
on the host plus, if needed, extra run args:

```ini
MUREX_DOCKER_RUN_EXTRA=--shm-size=2g
```

## Smoke test (manual)

```powershell
# Build, then run one container against a reachable backend:
docker run --rm -e MUREX_ENV_TARGET=10.10.0.93:20001 -e DISPLAY=:99 `
  -e SCREEN_GEOMETRY=1280x800 -v ${PWD}\data\out\smoke:/exports murex-thick:latest

# In another shell, screenshot what the model would see:
$cid = docker ps -q --filter ancestor=murex-thick:latest
docker exec $cid sh -c "export DISPLAY=:99 && import -window root png:-" > screen.png
```

`screen.png` should show the Murex login screen once the client has connected.
Tune `MUREX_CONTAINER_READY_SECS` if the client needs longer to settle.
