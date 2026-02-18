#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building libmetalmom.dylib..."
cd "$PROJECT_DIR"

swift build -c release

# Get the build directory (avoids find issues with symlinks)
BIN_PATH=$(swift build -c release --show-bin-path)
DYLIB_PATH="$BIN_PATH/libMetalMomBridge.dylib"

if [ ! -f "$DYLIB_PATH" ]; then
    echo "ERROR: Could not find built dylib at $DYLIB_PATH"
    echo "Looking with find -L as fallback..."
    DYLIB_PATH=$(find -L .build/release -name "libMetalMomBridge.dylib" -type f | head -1)
    if [ -z "$DYLIB_PATH" ]; then
        echo "ERROR: Could not find built dylib"
        exit 1
    fi
fi

# Copy to python package
mkdir -p python/metalmom/_lib
cp "$DYLIB_PATH" python/metalmom/_lib/libmetalmom.dylib
cp Sources/MetalMomCBridge/include/metalmom.h python/metalmom/_lib/metalmom.h

echo "Copied dylib and header to python/metalmom/_lib/"
echo "Done."
