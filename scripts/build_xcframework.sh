#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/xcframework"
DERIVED_DATA="$BUILD_DIR/DerivedData"

echo "Building MetalMom XCFramework..."
cd "$PROJECT_DIR"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------------------
# Strategy:
#   SPM static library products built via xcodebuild produce .o object files
#   (not .framework bundles), and dynamic products hit a known SPM/xcodebuild
#   linking bug with C target dependencies.
#
#   Workaround: build static, then manually assemble each platform's static
#   library (.a) + headers + Swift module, and pass them to
#   xcodebuild -create-xcframework -library ... -headers ...
# ---------------------------------------------------------------------------

build_platform() {
    local LABEL="$1"      # e.g. "iOS device"
    local DEST="$2"       # e.g. "generic/platform=iOS"
    local DD_NAME="$3"    # e.g. "ios"
    local DD_PATH="$DERIVED_DATA/$DD_NAME"

    echo ""
    echo "==> Building for $LABEL..."
    xcodebuild build \
        -scheme MetalMomCore \
        -destination "$DEST" \
        -derivedDataPath "$DD_PATH" \
        -skipPackagePluginValidation \
        -configuration Release \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        ONLY_ACTIVE_ARCH=NO

    # Find the merged .o in Build/Products/Release*/ (sdk suffix varies)
    local PRODUCTS_DIR
    PRODUCTS_DIR=$(find -L "$DD_PATH/Build/Products" -maxdepth 1 -name "Release*" -type d | head -1)
    if [ -z "$PRODUCTS_DIR" ]; then
        echo "ERROR: Could not find Release products dir in $DD_PATH" >&2
        exit 1
    fi

    local OBJ="$PRODUCTS_DIR/MetalMomCore.o"
    if [ ! -f "$OBJ" ]; then
        echo "ERROR: MetalMomCore.o not found in $PRODUCTS_DIR" >&2
        find -L "$PRODUCTS_DIR" -type f >&2
        exit 1
    fi

    # Create static library from object file
    local STAGE="$BUILD_DIR/stage/$DD_NAME"
    mkdir -p "$STAGE/lib" "$STAGE/headers"
    libtool -static "$OBJ" -o "$STAGE/lib/libMetalMomCore.a"

    # Copy Swift module (contains .swiftinterface, .swiftdoc, .swiftmodule)
    local SWIFTMOD="$PRODUCTS_DIR/MetalMomCore.swiftmodule"
    if [ -d "$SWIFTMOD" ]; then
        cp -R "$SWIFTMOD" "$STAGE/headers/"
    fi

    # Copy C bridge public headers
    local C_HEADERS="$PROJECT_DIR/Sources/MetalMomCBridge/include"
    if [ -d "$C_HEADERS" ]; then
        cp -R "$C_HEADERS"/*.h "$STAGE/headers/" 2>/dev/null || true
    fi

    # Create a module.modulemap that exposes both the Swift and C modules
    cat > "$STAGE/headers/module.modulemap" <<MMAP
framework module MetalMomCore {
    header "metalmom.h"
    export *
}
MMAP

    echo "    [ok] $LABEL -> $STAGE/lib/libMetalMomCore.a"
}

# Build for each platform
build_platform "iOS device (arm64)"  "generic/platform=iOS"           "ios"
build_platform "iOS Simulator"       "generic/platform=iOS Simulator"  "sim"
build_platform "macOS"               "generic/platform=macOS"          "mac"

# Create XCFramework from the staged static libraries + headers
echo ""
echo "==> Creating XCFramework..."
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/stage/ios/lib/libMetalMomCore.a" \
    -headers "$BUILD_DIR/stage/ios/headers" \
    -library "$BUILD_DIR/stage/sim/lib/libMetalMomCore.a" \
    -headers "$BUILD_DIR/stage/sim/headers" \
    -library "$BUILD_DIR/stage/mac/lib/libMetalMomCore.a" \
    -headers "$BUILD_DIR/stage/mac/headers" \
    -output "$BUILD_DIR/MetalMomCore.xcframework"

echo ""
echo "XCFramework created at: $BUILD_DIR/MetalMomCore.xcframework"
echo "Done."
