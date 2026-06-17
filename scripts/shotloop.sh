#!/usr/bin/env bash
# Per-session screenshot loop. Maps each running murex container to its
# env/trade via the bind-mount source path (run_dir/<env>/<trade> -> /exports),
# so each PNG is named <env>_<trade>_<HHMMSS>.png — identifiable when a session
# gets stuck. Captures every INTERVAL seconds for ITERS iterations.
set -u
RUN="${1:?run-dir name required}"
INTERVAL="${2:-15}"
ITERS="${3:-100}"
DIR="data/out/${RUN}/shots"
mkdir -p "$DIR"
echo "shotloop: RUN=$RUN every ${INTERVAL}s x${ITERS} -> $DIR"
for i in $(seq 1 "$ITERS"); do
  ts=$(date -u +%H%M%S)
  for id in $(docker ps -q --filter ancestor=murex-thick:latest); do
    src=$(docker inspect -f '{{range .Mounts}}{{.Source}}{{end}}' "$id" 2>/dev/null)
    label=$(printf '%s' "$src" | tr '\\' '/' | awk -F/ '{print $(NF-1)"_"$NF}')
    [ -z "$label" ] && label="$id"
    docker exec "$id" sh -c "export DISPLAY=:99 && import -window root png:-" \
      > "$DIR/${label}_${ts}.png" 2>/dev/null
  done
  sleep "$INTERVAL"
done
