#!/bin/bash
# bump_version.sh - Bump version across all source files, generate changelog, commit and tag.
#
# Usage:
#   ./scripts/bump_version.sh 0.2.0           # release
#   ./scripts/bump_version.sh 0.2.0 --rc 1    # release candidate (tags v0.2.0-rc1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VERSION=""
RC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rc)
            RC="$2"
            shift 2
            ;;
        *)
            if [ -z "$VERSION" ]; then
                VERSION="$1"
            else
                echo "Unknown argument: $1"; exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [--rc <number>]"
    echo "  e.g. $0 0.2.0"
    echo "  e.g. $0 0.2.0 --rc 1"
    exit 1
fi

# Validate semver format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "ERROR: Version must be semver (e.g. 0.2.0), got: $VERSION"
    exit 1
fi

if [ -n "$RC" ]; then
    TAG="v${VERSION}-rc${RC}"
else
    TAG="v${VERSION}"
fi

echo "==> Bumping to $VERSION (tag: $TAG)"

# ---------------------------------------------------------------------------
# Check for clean working tree
# ---------------------------------------------------------------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree has uncommitted changes. Commit or stash first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Update version in all source files
# ---------------------------------------------------------------------------
echo "==> Updating pyproject.toml..."
sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

echo "==> Updating python/metalmom/__init__.py..."
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" python/metalmom/__init__.py

echo "==> Updating docs/api/conf.py..."
sed -i '' "s/^release = \".*\"/release = \"$VERSION\"/" docs/api/conf.py

# ---------------------------------------------------------------------------
# Generate changelog (if git-cliff is installed)
# ---------------------------------------------------------------------------
if command -v git-cliff &>/dev/null; then
    echo "==> Generating changelog with git-cliff..."
    git-cliff --tag "$TAG" --output CHANGELOG.md
else
    echo "==> WARN: git-cliff not found, skipping changelog generation"
    echo "    Install with: brew install git-cliff"
fi

# ---------------------------------------------------------------------------
# Commit and tag
# ---------------------------------------------------------------------------
echo "==> Committing version bump..."
git add pyproject.toml python/metalmom/__init__.py docs/api/conf.py CHANGELOG.md
git commit -m "release: $TAG"

echo "==> Creating tag: $TAG"
git tag -a "$TAG" -m "Release $TAG"

echo ""
echo "==> Done! To publish:"
echo "    git push origin main --tags"
