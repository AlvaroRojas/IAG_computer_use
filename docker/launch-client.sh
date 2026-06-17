#!/bin/bash
# Parameterized Mx.3 client launcher.
#
# Mirrors the vendor murex/client.sh EXCEPT:
#   - the fileserver target is read from the environment (one image, many
#     before/after backends);
#   - the JVM is version-aware: the vendor Java-8-only flags
#     (-Xbootclasspath/p:, -XX:MaxPermSize) are emitted ONLY on Java 8. The real
#     payload is Java 17, where those flags are removed/illegal.
# The vendor file is left untouched.
#
# Target resolution (first match wins):
#   1. explicit MXJ_FILESERVER_HOST / MXJ_FILESERVER_PORT
#   2. $MUREX_ENV_TARGET  (host | host:port | scheme://host:port[/path])
#   3. vendor defaults (10.10.0.93 : 20001)
set -euo pipefail

APP_DIR="${MUREX_APP_DIR:-/opt/murex}"
cd "$APP_DIR"

export JAVAHOME="${JAVAHOME:-${JAVA_HOME:-/opt/java/openjdk}}"

# Parse MUREX_ENV_TARGET into host[:port].
TARGET="${MUREX_ENV_TARGET:-}"
T_HOST=""
T_PORT=""
if [ -n "$TARGET" ]; then
  TARGET="${TARGET#*://}"     # strip scheme://
  TARGET="${TARGET%%/*}"      # strip /path
  T_HOST="${TARGET%%:*}"
  case "$TARGET" in *:*) T_PORT="${TARGET##*:}" ;; esac
fi

export MXJ_FILESERVER_HOST="${MXJ_FILESERVER_HOST:-${T_HOST:-10.10.0.93}}"
export MXJ_FILESERVER_PORT="${MXJ_FILESERVER_PORT:-${T_PORT:-20001}}"

# Live env answers on site1 with NO destination-site (proven by the vendor's
# working launcher); site=default + destination=site1 makes the client report
# "MX.3 temporarily unavailable / services not started". Still env-overridable so
# a before/after diff can target different sites.
export MXJ_SITE_NAME="${MXJ_SITE_NAME:-site1}"
export MXJ_PLATFORM_NAME="${MXJ_PLATFORM_NAME:-MX}"
export MXJ_PROCESS_NICK_NAME="${MXJ_PROCESS_NICK_NAME:-MX}"
export MXJ_PING_POP_GUI_DOCUMENT="${MXJ_PING_POP_GUI_DOCUMENT:-1}"
export MXJ_POP_CONNECTION_TIMEOUT="${MXJ_POP_CONNECTION_TIMEOUT:-60000}"

export PATH="$JAVAHOME/bin:$PATH"
# Native middleware libs (libmiddleware_system_linux_x86_64*.so) live in bin/.
export LD_LIBRARY_PATH="$APP_DIR/bin:${LD_LIBRARY_PATH:-}"

MXJ_JAR_FILELIST="murex.download.guiclient.download"
MXJ_POLICY="${MXJ_POLICY:-$APP_DIR/java.policy}"
MXJ_BOOT="mxjboot.jar"
MXJ_CONFIG_FILE="client.xml"

# Embedded JxBrowser (used by some Mx.3 GUI panels) logs/crash-dumps here.
mkdir -p "$APP_DIR/logs/JxBrowser" 2>/dev/null || true

# Refresh the boot jar from jar/ (vendor behaviour) — note jar/mxjboot.jar is the
# Java-17 build that drives the rest of the client.
[ -f "jar/$MXJ_BOOT" ] && cp -f "jar/$MXJ_BOOT" .

# Detect the JVM major version (8, 11, 17, ...).
JV="$("$JAVAHOME/bin/java" -version 2>&1 | awk -F'"' '/version/{print $2; exit}')"
case "$JV" in
  1.8*) JMAJOR=8 ;;
  "")   JMAJOR=0 ;;
  *)    JMAJOR="${JV%%.*}" ;;
esac

# Common args.
JAVA_ARGS=(
  -Xmx256M
  -Dsun.java2d.noddraw=true
  -Dsun.java2d.uiScale.enabled=false
  -DJINTEGRA_NATIVE_MODE
  -Djxbrowser.logging.file=logs/jxbrowser.log
  -Djxbrowser.crash.dump.dir=logs/JxBrowser
  -Djava.security.policy="$MXJ_POLICY"
  -Djava.rmi.server.codebase="http://$MXJ_FILESERVER_HOST:$MXJ_FILESERVER_PORT/$MXJ_JAR_FILELIST"
)

if [ "$JMAJOR" -eq 8 ]; then
  # Vendor Java-8 path: prepend the bundled XML impls + PermGen sizing.
  JAVA_ARGS=(
    -Xbootclasspath/p:jar/xercesImpl-2.9.1.jar:jar/xml-apis-1.3.04.jar:jar/xalan-2.7.1m1.jar:jar/serializer-2.7.1m.jar
    -XX:MaxPermSize=100M
    "${JAVA_ARGS[@]}"
  )
else
  # Java 9+ strong encapsulation: the Mx.3 Swing UI reaches into JDK-internal
  # APIs (sun.swing, sun.awt, ...) and reflects into java.base. Without these
  # the client dies with IllegalAccessError/InaccessibleObjectException as soon
  # as it shows a dialog. This is the standard Murex-on-JDK17 flag set.
  JAVA_ARGS+=(
    --add-exports=java.desktop/sun.swing=ALL-UNNAMED
    --add-exports=java.desktop/sun.awt=ALL-UNNAMED
    --add-exports=java.desktop/sun.awt.image=ALL-UNNAMED
    --add-exports=java.desktop/com.sun.java.swing.plaf.windows=ALL-UNNAMED
    --add-exports=java.desktop/com.sun.java.swing.plaf.motif=ALL-UNNAMED
    --add-exports=java.base/sun.security.action=ALL-UNNAMED
    --add-opens=java.base/java.lang=ALL-UNNAMED
    --add-opens=java.base/java.lang.reflect=ALL-UNNAMED
    --add-opens=java.base/java.util=ALL-UNNAMED
    --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
    --add-opens=java.base/java.text=ALL-UNNAMED
    --add-opens=java.base/java.io=ALL-UNNAMED
    --add-opens=java.base/java.nio=ALL-UNNAMED
    --add-opens=java.base/java.net=ALL-UNNAMED
    --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
    --add-opens=java.base/java.security=ALL-UNNAMED
    --add-opens=java.desktop/java.awt=ALL-UNNAMED
    --add-opens=java.desktop/java.awt.event=ALL-UNNAMED
    --add-opens=java.desktop/sun.swing=ALL-UNNAMED
    --add-opens=java.desktop/sun.awt=ALL-UNNAMED
    --add-opens=java.desktop/javax.swing=ALL-UNNAMED
    --add-opens=java.desktop/javax.swing.text=ALL-UNNAMED
    --add-opens=java.desktop/javax.swing.plaf.basic=ALL-UNNAMED
    --add-opens=java.desktop/com.sun.java.swing.plaf.windows=ALL-UNNAMED
  )
fi

echo "[launch] java=$JV (major=$JMAJOR) fileserver=$MXJ_FILESERVER_HOST:$MXJ_FILESERVER_PORT"

exec "$JAVAHOME/bin/java" \
  "${JAVA_ARGS[@]}" \
  -jar "$MXJ_BOOT" \
  /MXJ_MLC_SERVICE:MXMLC.SESSION \
  /MXJ_SITE_NAME:"$MXJ_SITE_NAME" \
  /MXJ_CLASS_NAME:murex.gui.xml.XmlGuiClientBoot \
  /MXJ_PLATFORM_NAME:"$MXJ_PLATFORM_NAME" \
  /MXJ_PROCESS_NICK_NAME:"$MXJ_PROCESS_NICK_NAME" \
  /MXJ_CONFIG_FILE:"$MXJ_CONFIG_FILE" \
  /MXJ_PING_POP_GUI_DOCUMENT:"$MXJ_PING_POP_GUI_DOCUMENT" \
  /MXJ_POP_CONNECTION_TIMEOUT:"$MXJ_POP_CONNECTION_TIMEOUT" \
  "$@"
