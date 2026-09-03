#!/usr/bin/env bash
# Build a single-file Linux AppImage from the PyInstaller onedir bundle.
# Prereq: dist/DollarDetective/ exists (run: pyinstaller DollarDetective.spec).
# Reusable in CI (the GitHub Actions Linux job calls this after PyInstaller).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
ONEDIR="dist/DollarDetective"
APPDIR="build/DollarDetective.AppDir"
TOOL="build/appimagetool-x86_64.AppImage"

[ -d "$ONEDIR" ] || { echo "ERROR: $ONEDIR not found. Build with PyInstaller first."; exit 1; }

echo "==> Preparing AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -a "$ONEDIR" "$APPDIR/usr/bin/DollarDetective"

# AppRun: launch the bundled executable
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/DollarDetective/DollarDetective" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# .desktop entry
cat > "$APPDIR/DollarDetective.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Dollar Detective
Exec=DollarDetective
Icon=DollarDetective
Categories=Utility;Graphics;
Terminal=false
EOF

# Icon: use the bundled app icon; fall back to a generated $ if it's missing.
if [ -f assets/icon.png ]; then
    cp assets/icon.png "$APPDIR/DollarDetective.png"
elif command -v convert >/dev/null 2>&1; then
    convert -size 256x256 xc:'#1F6E56' -gravity center -fill white \
        -pointsize 170 -annotate 0 '$' "$APPDIR/DollarDetective.png"
else
    # minimal 1x1 fallback so packaging still succeeds
    printf '\x89PNG\r\n\x1a\n' > "$APPDIR/DollarDetective.png"
fi

echo "==> Fetching appimagetool (if needed)"
if [ ! -x "$TOOL" ]; then
    curl -fsSL -o "$TOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi

echo "==> Building AppImage"
OUT="dist/DollarDetective-x86_64.AppImage"
# --appimage-extract-and-run avoids needing FUSE for the *tool* itself
ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUT"
echo "==> Done: $OUT"
ls -lh "$OUT"
