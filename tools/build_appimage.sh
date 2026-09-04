#!/usr/bin/env bash
# Build a single-file Linux AppImage from a PyInstaller onedir bundle.
# Prereq: dist/<APP_NAME>/ exists (run PyInstaller with the matching .spec first).
# Reusable in CI (the GitHub Actions Linux jobs call this after PyInstaller).
#
# Parameterized so both the full app and the standalone crop tool can share it.
# Defaults reproduce the full app's original behavior exactly:
#   APP_NAME     PyInstaller onedir + executable name (default DollarDetective)
#   APP_DISPLAY  .desktop display name              (default "Dollar Detective")
#   APP_ICON     source PNG for the AppImage icon   (default assets/icon.png)
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

APP_NAME="${APP_NAME:-DollarDetective}"
APP_DISPLAY="${APP_DISPLAY:-Dollar Detective}"
APP_ICON="${APP_ICON:-assets/icon.png}"

ONEDIR="dist/$APP_NAME"
APPDIR="build/$APP_NAME.AppDir"
TOOL="build/appimagetool-x86_64.AppImage"

[ -d "$ONEDIR" ] || { echo "ERROR: $ONEDIR not found. Build with PyInstaller first."; exit 1; }

echo "==> Preparing AppDir for $APP_NAME"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -a "$ONEDIR" "$APPDIR/usr/bin/$APP_NAME"

# AppRun: launch the bundled executable. $APP_NAME expands now; the runtime
# shell vars ($0/$@/$(...)) are escaped so they stay literal in the script.
cat > "$APPDIR/AppRun" <<EOF
#!/bin/bash
HERE="\$(dirname "\$(readlink -f "\$0")")"
exec "\$HERE/usr/bin/$APP_NAME/$APP_NAME" "\$@"
EOF
chmod +x "$APPDIR/AppRun"

# .desktop entry
cat > "$APPDIR/$APP_NAME.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_DISPLAY
Exec=$APP_NAME
Icon=$APP_NAME
Categories=Utility;Graphics;
Terminal=false
EOF

# Icon: use the provided app icon; fall back to a generated $ if it's missing.
if [ -f "$APP_ICON" ]; then
    cp "$APP_ICON" "$APPDIR/$APP_NAME.png"
elif command -v convert >/dev/null 2>&1; then
    convert -size 256x256 xc:'#1F6E56' -gravity center -fill white \
        -pointsize 170 -annotate 0 '$' "$APPDIR/$APP_NAME.png"
else
    # minimal 1x1 fallback so packaging still succeeds
    printf '\x89PNG\r\n\x1a\n' > "$APPDIR/$APP_NAME.png"
fi

echo "==> Fetching appimagetool (if needed)"
if [ ! -x "$TOOL" ]; then
    curl -fsSL -o "$TOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi

echo "==> Building AppImage"
OUT="dist/$APP_NAME-x86_64.AppImage"
# --appimage-extract-and-run avoids needing FUSE for the *tool* itself
ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUT"
echo "==> Done: $OUT"
ls -lh "$OUT"
