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
# SPM packages built via `xcodebuild archive` produce .o object files, not
# .framework bundles.  Using `xcodebuild build` instead generates proper
# PackageFrameworks/<Name>.framework output that xcodebuild -create-xcframework
# can consume.
# ---------------------------------------------------------------------------

# Build for iOS device
echo "Building for iOS device (arm64)..."
xcodebuild build \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS" \
    -derivedDataPath "$DERIVED_DATA/ios" \
    -skipPackagePluginValidation \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    ONLY_ACTIVE_ARCH=NO

# Build for iOS Simulator
echo "Building for iOS Simulator..."
xcodebuild build \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS Simulator" \
    -derivedDataPath "$DERIVED_DATA/sim" \
    -skipPackagePluginValidation \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    ONLY_ACTIVE_ARCH=NO

# Build for macOS
echo "Building for macOS..."
xcodebuild build \
    -scheme MetalMomCore \
    -destination "generic/platform=macOS" \
    -derivedDataPath "$DERIVED_DATA/mac" \
    -skipPackagePluginValidation \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    ONLY_ACTIVE_ARCH=NO

# ---------------------------------------------------------------------------
# Locate the .framework bundles in DerivedData.
# SPM puts them under Build/Products/<Config>-<sdk>/PackageFrameworks/
# ---------------------------------------------------------------------------
find_framework() {
    local DD="$1"
    local NAME="$2"

    local FW
    FW=$(find -L "$DD" -path "*/PackageFrameworks/${NAME}.framework" -type d 2>/dev/null | head -1)
    if [ -n "$FW" ]; then
        echo "$FW"
        return
    fi

    # Fallback: search for any .framework with the right name
    FW=$(find -L "$DD" -name "${NAME}.framework" -type d 2>/dev/null | head -1)
    if [ -n "$FW" ]; then
        echo "$FW"
        return
    fi

    echo "ERROR: Could not find ${NAME}.framework in ${DD}" >&2
    echo "Contents of DerivedData:" >&2
    find -L "$DD" -name "*.framework" -type d 2>/dev/null >&2 || true
    exit 1
}

IOS_FW=$(find_framework "$DERIVED_DATA/ios" "MetalMomCore")
SIM_FW=$(find_framework "$DERIVED_DATA/sim" "MetalMomCore")
MAC_FW=$(find_framework "$DERIVED_DATA/mac" "MetalMomCore")

echo ""
echo "Found frameworks:"
echo "  iOS:       $IOS_FW"
echo "  Simulator: $SIM_FW"
echo "  macOS:     $MAC_FW"

# Create XCFramework
echo ""
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
    -framework "$IOS_FW" \
    -framework "$SIM_FW" \
    -framework "$MAC_FW" \
    -output "$BUILD_DIR/MetalMomCore.xcframework"

echo ""
echo "XCFramework created at: $BUILD_DIR/MetalMomCore.xcframework"
echo "Done."
