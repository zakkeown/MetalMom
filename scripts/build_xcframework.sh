#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/xcframework"

echo "Building MetalMom XCFramework..."
cd "$PROJECT_DIR"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Archive for iOS device
echo "Archiving for iOS device (arm64)..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS" \
    -archivePath "$BUILD_DIR/MetalMomCore-iOS.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Archive for iOS Simulator
echo "Archiving for iOS Simulator..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=iOS Simulator" \
    -archivePath "$BUILD_DIR/MetalMomCore-iOSSimulator.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Archive for macOS
echo "Archiving for macOS..."
xcodebuild archive \
    -scheme MetalMomCore \
    -destination "generic/platform=macOS" \
    -archivePath "$BUILD_DIR/MetalMomCore-macOS.xcarchive" \
    -skipPackagePluginValidation \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

# Find the framework or library in each archive.
# SPM packages produce PackageFrameworks/<Name>.framework inside the archive,
# not the standard Products/Library/Frameworks/ path.
find_framework_or_library() {
    local ARCHIVE="$1"
    local NAME="$2"

    # Try standard framework path
    local FW="$ARCHIVE/Products/Library/Frameworks/${NAME}.framework"
    if [ -d "$FW" ]; then
        echo "-framework" "$FW"
        return
    fi

    # Try SPM PackageFrameworks path (inside Products)
    FW=$(find "$ARCHIVE" -path "*/PackageFrameworks/${NAME}.framework" -type d | head -1)
    if [ -n "$FW" ]; then
        echo "-framework" "$FW"
        return
    fi

    # Try static library path
    local LIB=$(find "$ARCHIVE" -name "lib${NAME}.a" -type f | head -1)
    if [ -n "$LIB" ]; then
        echo "-library" "$LIB"
        return
    fi

    echo "ERROR: Could not find framework or library for ${NAME} in ${ARCHIVE}" >&2
    exit 1
}

# Create XCFramework
echo "Creating XCFramework..."

IOS_ARGS=$(find_framework_or_library "$BUILD_DIR/MetalMomCore-iOS.xcarchive" "MetalMomCore")
SIM_ARGS=$(find_framework_or_library "$BUILD_DIR/MetalMomCore-iOSSimulator.xcarchive" "MetalMomCore")
MAC_ARGS=$(find_framework_or_library "$BUILD_DIR/MetalMomCore-macOS.xcarchive" "MetalMomCore")

echo "  iOS:       $IOS_ARGS"
echo "  Simulator: $SIM_ARGS"
echo "  macOS:     $MAC_ARGS"

xcodebuild -create-xcframework \
    $IOS_ARGS \
    $SIM_ARGS \
    $MAC_ARGS \
    -output "$BUILD_DIR/MetalMomCore.xcframework"

echo ""
echo "XCFramework created at: $BUILD_DIR/MetalMomCore.xcframework"
echo "Done."
