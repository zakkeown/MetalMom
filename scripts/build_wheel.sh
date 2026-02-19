#!/bin/bash
# build_wheel.sh - Build a Python wheel for MetalMom.
#
# This script:
#   1. Builds the native dylib via build_dylib.sh
#   2. Builds a Python wheel that bundles the dylib + header
#
# Usage:
#   ./scripts/build_wheel.sh            # build wheel into dist/
#   ./scripts/build_wheel.sh --no-swift # skip Swift build (dylib must already exist)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_SWIFT=0
for arg in "$@"; do
    case "$arg" in
        --no-swift) SKIP_SWIFT=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Build the native dylib (unless --no-swift)
# ---------------------------------------------------------------------------
if [ "$SKIP_SWIFT" -eq 0 ]; then
    echo "==> Building native dylib..."
    bash "$SCRIPT_DIR/build_dylib.sh"
else
    echo "==> Skipping Swift build (--no-swift)"
fi

# Verify that the dylib and header exist before proceeding
DYLIB="python/metalmom/_lib/libmetalmom.dylib"
HEADER="python/metalmom/_lib/metalmom.h"

if [ ! -f "$DYLIB" ]; then
    echo "ERROR: Dylib not found at $DYLIB"
    echo "Run ./scripts/build_dylib.sh first, or omit --no-swift."
    exit 1
fi
if [ ! -f "$HEADER" ]; then
    echo "ERROR: Header not found at $HEADER"
    echo "Run ./scripts/build_dylib.sh first, or omit --no-swift."
    exit 1
fi

echo "==> Native artifacts present:"
echo "    $DYLIB  ($(du -h "$DYLIB" | cut -f1))"
echo "    $HEADER"

# ---------------------------------------------------------------------------
# Step 2: Build the Python wheel
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"

# Ensure build dependencies are available
for pkg in build hatchling; do
    if ! "$PYTHON" -c "import $pkg" >/dev/null 2>&1; then
        echo "==> Installing '$pkg'..."
        "$PYTHON" -m pip install --quiet "$pkg"
    fi
done

# Use --no-isolation so hatchling's force-include can see the .gitignored
# _lib/ directory.  (An isolated build copies only VCS-tracked sources into
# a temp tree, which excludes the dylib.)
echo "==> Building wheel..."
"$PYTHON" -m build --wheel --no-isolation --outdir dist/

# ---------------------------------------------------------------------------
# Step 3: Report & validate the result
# ---------------------------------------------------------------------------
WHEEL=$(ls -t dist/metalmom-*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found in dist/"
    exit 1
fi

echo ""
echo "==> Wheel built successfully:"
echo "    $WHEEL  ($(du -h "$WHEEL" | cut -f1))"

# Quick sanity check: verify the _lib files are inside the wheel
if unzip -l "$WHEEL" | grep -q "_lib/libmetalmom.dylib"; then
    echo "    [ok] dylib included in wheel"
else
    echo "    [WARN] dylib NOT found inside wheel!"
    exit 1
fi

if unzip -l "$WHEEL" | grep -q "_lib/metalmom.h"; then
    echo "    [ok] header included in wheel"
else
    echo "    [WARN] header NOT found inside wheel!"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: Fix platform tag (macOS arm64 only, stable ABI from Python 3.11)
#
# The wheel is built as py3-none-any but bundles a macOS arm64 dylib.
# We use `wheel tags` to properly re-tag both the filename AND the internal
# .dist-info/WHEEL metadata, avoiding mismatches that break PyPI/pip.
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import wheel" >/dev/null 2>&1; then
    echo "==> Installing 'wheel'..."
    "$PYTHON" -m pip install --quiet wheel
fi

if echo "$WHEEL" | grep -q "py3-none-any"; then
    echo "==> Re-tagging wheel for macOS arm64 stable ABI..."
    "$PYTHON" -m wheel tags --remove \
        --python-tag cp311 \
        --abi-tag abi3 \
        --platform-tag macosx_14_0_arm64 \
        "$WHEEL"
    # wheel tags renames the file in place; find the new one
    WHEEL=$(ls -t dist/metalmom-*.whl 2>/dev/null | head -1)
    echo "    [ok] re-tagged: $(basename "$WHEEL")"
fi

echo ""
echo "Done. Install with:  pip install $WHEEL"
