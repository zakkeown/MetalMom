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

# Create XCFramework
echo "Creating XCFramework..."
xcodebuild -create-xcframework \
    -archive "$BUILD_DIR/MetalMomCore-iOS.xcarchive" -framework MetalMomCore.framework \
    -archive "$BUILD_DIR/MetalMomCore-iOSSimulator.xcarchive" -framework MetalMomCore.framework \
    -archive "$BUILD_DIR/MetalMomCore-macOS.xcarchive" -framework MetalMomCore.framework \
    -output "$BUILD_DIR/MetalMomCore.xcframework"

echo ""
echo "XCFramework created at: $BUILD_DIR/MetalMomCore.xcframework"
echo "Done."
