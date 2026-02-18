# Deployment & Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship MetalMom v0.1.0 with professional release engineering for both PyPI and SPM/XCFramework, including RC workflow, signed releases, auto-changelog, HF model hosting, and CI notifications.

**Architecture:** Fix the existing release pipeline (platform tags, XCFramework), add release tooling (version bump, git-cliff, sigstore), create a `metalmom.models` module for HF model downloads, and extend CI with parity failure notifications. Single version number across Python and Swift.

**Tech Stack:** hatchling (wheel build), git-cliff (changelog), sigstore (signing), huggingface_hub (model download), gh CLI (issue filing)

---

### Task 1: Fix Wheel Platform Tag

**Files:**
- Modify: `scripts/build_wheel.sh`
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml to declare the stable ABI tag**

Add a custom wheel tag section. Since cffi uses the stable ABI and the bundled dylib is Python-version-independent, we use `abi3` to cover Python 3.11+.

In `pyproject.toml`, add after the existing `[tool.hatch.build.targets.wheel.force-include]` section:

```toml
[tool.hatch.build.targets.wheel.shared-data]

[tool.hatch.build.hooks.custom]
```

Actually, hatchling does not natively support overriding platform tags. The simplest approach: rename the wheel file after building.

In `scripts/build_wheel.sh`, after the wheel is built and validated, add a renaming step:

```bash
# ---------------------------------------------------------------------------
# Step 3: Fix platform tag (macOS arm64 only, stable ABI from Python 3.11)
# ---------------------------------------------------------------------------
ORIGINAL_WHEEL="$WHEEL"
FIXED_WHEEL=$(echo "$WHEEL" | sed 's/py3-none-any/cp311-abi3-macosx_14_0_arm64/')

if [ "$ORIGINAL_WHEEL" != "$FIXED_WHEEL" ]; then
    mv "$ORIGINAL_WHEEL" "$FIXED_WHEEL"
    WHEEL="$FIXED_WHEEL"
    echo "    [ok] renamed to platform-specific tag: $(basename "$FIXED_WHEEL")"
fi
```

**Step 2: Run the build to verify**

Run: `swift build -c release && ./scripts/build_dylib.sh && ./scripts/build_wheel.sh --no-swift`
Expected: Wheel file named `metalmom-0.1.0-cp311-abi3-macosx_14_0_arm64.whl`

**Step 3: Verify the wheel installs correctly**

Run: `.venv/bin/pip install dist/metalmom-0.1.0-cp311-abi3-macosx_14_0_arm64.whl --force-reinstall && .venv/bin/python -c "import metalmom; print(metalmom.__version__)"`
Expected: Prints `0.1.0`

**Step 4: Commit**

```bash
git add scripts/build_wheel.sh pyproject.toml
git commit -m "fix: wheel platform tag to cp311-abi3-macosx_14_0_arm64"
```

---

### Task 2: Version Bump Script

**Files:**
- Create: `scripts/bump_version.sh`
- Modify: `python/metalmom/__init__.py` (version updated by script)
- Modify: `pyproject.toml` (version updated by script)
- Modify: `docs/api/conf.py` (version updated by script)

**Step 1: Create the bump script**

Create `scripts/bump_version.sh`:

```bash
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
```

**Step 2: Make it executable and test dry-run behavior**

Run: `chmod +x scripts/bump_version.sh && scripts/bump_version.sh`
Expected: Prints usage message and exits 1

Run: `scripts/bump_version.sh not-semver`
Expected: Prints "ERROR: Version must be semver" and exits 1

**Step 3: Commit**

```bash
git add scripts/bump_version.sh
git commit -m "feat: add version bump script"
```

---

### Task 3: git-cliff Configuration

**Files:**
- Create: `cliff.toml`

**Step 1: Install git-cliff locally**

Run: `brew install git-cliff`
Expected: `git-cliff --version` prints a version

**Step 2: Create cliff.toml**

Create `cliff.toml` at the repo root:

```toml
[changelog]
header = """# Changelog

All notable changes to MetalMom will be documented in this file.\n
"""
body = """
{%- macro remote_url() -%}
  https://github.com/zakkeown/MetalMom
{%- endmacro -%}

{% if version -%}
    ## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else -%}
    ## [Unreleased]
{% endif -%}

{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | striptags | trim | upper_first }}
    {% for commit in commits %}
        - {{ commit.message | split(pat="\n") | first | trim }}\
    {% endfor %}
{% endfor %}
"""
footer = ""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
split_commits = false
commit_parsers = [
    { message = "^feat", group = "Added" },
    { message = "^fix", group = "Fixed" },
    { message = "^perf", group = "Performance" },
    { message = "^refactor", group = "Changed" },
    { message = "^docs", group = "Documentation" },
    { message = "^test", group = "Testing" },
    { message = "^ci", group = "CI" },
    { message = "^release", skip = true },
]
filter_commits = false
tag_pattern = "v[0-9].*"
sort_commits = "newest"
```

**Step 3: Test changelog generation**

Run: `git-cliff --unreleased`
Expected: Prints grouped changelog entries from recent commits

**Step 4: Commit**

```bash
git add cliff.toml
git commit -m "ci: add git-cliff changelog configuration"
```

---

### Task 4: Parity Test Failure Notification Script

**Files:**
- Create: `scripts/ci/file_parity_issues.py`
- Modify: `.github/workflows/parity.yml`

**Step 1: Create the parity issue filing script**

Create `scripts/ci/file_parity_issues.py` following the same pattern as `scripts/ci/file_api_drift_issues.py`:

```python
#!/usr/bin/env python3
"""File GitHub issues for parity test failures.

Reads a JUnit XML test result file and creates a GitHub issue (via `gh` CLI)
if any tests failed.

Usage:
    python scripts/ci/file_parity_issues.py [--report parity-results.xml] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def parse_junit_xml(path: Path) -> dict:
    """Parse JUnit XML and extract failure information."""
    tree = ET.parse(path)
    root = tree.getroot()

    failures = []
    total = 0
    failed = 0

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            total += 1
            failure = testcase.find("failure")
            if failure is not None:
                failed += 1
                failures.append({
                    "name": testcase.get("name", "unknown"),
                    "classname": testcase.get("classname", ""),
                    "message": failure.get("message", ""),
                })

    return {"total": total, "failed": failed, "failures": failures}


def build_issue_body(report: dict, run_url: str) -> str:
    """Build a markdown issue body from the parsed test results."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    failures = report["failures"]

    lines = [
        "## Parity Test Failure",
        "",
        f"**{report['failed']}** of **{report['total']}** parity/compat tests failed.",
        "",
        f"- **Date:** {today}",
        f"- **Workflow run:** {run_url}" if run_url else "",
        "",
        "---",
        "",
        "### Failed Tests",
        "",
    ]

    for f in failures[:50]:  # cap at 50 to avoid huge issues
        classname = f["classname"]
        name = f["name"]
        message = f["message"][:200] if f["message"] else ""
        lines.append(f"- [ ] `{classname}::{name}`")
        if message:
            lines.append(f"  > {message}")

    if len(failures) > 50:
        lines.append(f"\n... and {len(failures) - 50} more failures (see workflow run)")

    lines.extend([
        "",
        "---",
        "",
        "*This issue was auto-generated by the parity test workflow.*",
    ])

    return "\n".join(lines)


def build_issue_title(report: dict) -> str:
    """Build a concise issue title."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"Parity test failure: {report['failed']} test(s) failed ({today})"


def check_duplicate_issues(label: str) -> dict | None:
    """Check if an open issue with the given label already exists. Returns issue number if found."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--label", label,
             "--json", "number,title", "--limit", "1"],
            capture_output=True, text=True, check=True,
        )
        issues = json.loads(result.stdout)
        if issues:
            return issues[0]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def file_issue(title: str, body: str, labels: list[str], dry_run: bool) -> bool:
    """Create a GitHub issue using the gh CLI."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    for label in labels:
        cmd.extend(["--label", label])

    if dry_run:
        print("DRY RUN -- would create issue:")
        print(f"  Title: {title}")
        print(f"\nBody:\n{body}")
        return True

    print(f"Creating issue: {title}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Issue created: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gh issue create failed: {e.stderr}")
        return False


def add_comment(issue_number: int, body: str, dry_run: bool) -> bool:
    """Add a comment to an existing issue."""
    cmd = ["gh", "issue", "comment", str(issue_number), "--body", body]

    if dry_run:
        print(f"DRY RUN -- would comment on issue #{issue_number}")
        return True

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Comment added to issue #{issue_number}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gh issue comment failed: {e.stderr}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="File GitHub issues for parity test failures")
    parser.add_argument(
        "--report", type=Path, default=Path("parity-results.xml"),
        help="Path to the JUnit XML test report (default: parity-results.xml)",
    )
    parser.add_argument(
        "--run-url", type=str, default="",
        help="URL to the GitHub Actions workflow run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be filed without actually creating an issue",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"ERROR: Report file not found: {args.report}")
        return 1

    report = parse_junit_xml(args.report)

    if report["failed"] == 0:
        print("All parity tests passed -- nothing to file.")
        return 0

    title = build_issue_title(report)
    body = build_issue_body(report, args.run_url)
    labels = ["parity-failure", "automated"]

    # Deduplicate: comment on existing issue instead of creating a new one
    if not args.dry_run:
        existing = check_duplicate_issues("parity-failure")
        if existing:
            print(f"Existing open parity issue found: #{existing['number']} - {existing['title']}")
            return 0 if add_comment(existing["number"], body, args.dry_run) else 1

    success = file_issue(title, body, labels, args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Add failure notification step to parity.yml**

In `.github/workflows/parity.yml`, add `permissions` and a new step after the upload artifact step:

```yaml
    permissions:
      issues: write
      contents: read
```

And add this step at the end of the job:

```yaml
      - name: File GitHub issue on failure
        if: failure()
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          python scripts/ci/file_parity_issues.py \
            --report parity-results.xml \
            --run-url "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
```

**Step 3: Test the script locally with a dummy XML**

Run: `python3 scripts/ci/file_parity_issues.py --dry-run --report /dev/null` (should error cleanly on parse)

Create a test XML: write a minimal JUnit file with one failure, then:
Run: `python3 scripts/ci/file_parity_issues.py --dry-run --report test-parity.xml`
Expected: Prints "DRY RUN" with formatted issue body. Delete the test file after.

**Step 4: Commit**

```bash
git add scripts/ci/file_parity_issues.py .github/workflows/parity.yml
git commit -m "ci: auto-file GitHub issues on parity test failure"
```

---

### Task 5: XCFramework in Release Workflow

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add XCFramework build, zip, and checksum steps**

In `.github/workflows/release.yml`, add these steps after the "Build dylib" step and before the "Set up Python" step:

```yaml
      # ── XCFramework ─────────────────────────────────────────
      - name: Build XCFramework
        run: ./scripts/build_xcframework.sh

      - name: Package XCFramework
        run: |
          cd .build/xcframework
          zip -r MetalMomCore.xcframework.zip MetalMomCore.xcframework
          shasum -a 256 MetalMomCore.xcframework.zip | cut -d ' ' -f1 > MetalMomCore.xcframework.sha256
          echo "XCFramework checksum: $(cat MetalMomCore.xcframework.sha256)"
```

**Step 2: Update the GitHub Release step to attach XCFramework**

Replace the existing release creation step with:

```yaml
      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          TAG="${{ steps.version.outputs.tag }}"
          IS_RC=false
          if echo "$TAG" | grep -q "\-rc"; then
            IS_RC=true
          fi

          PRERELEASE_FLAG=""
          if [ "$IS_RC" = true ]; then
            PRERELEASE_FLAG="--prerelease"
          fi

          NOTES_FILE=""
          if command -v git-cliff &>/dev/null; then
            git-cliff --latest --strip header > release-notes.md
            NOTES_FILE="--notes-file release-notes.md"
          else
            NOTES_FILE="--generate-notes"
          fi

          gh release create "$TAG" \
            --title "MetalMom $TAG" \
            $PRERELEASE_FLAG \
            $NOTES_FILE \
            dist/metalmom-*.whl \
            .build/xcframework/MetalMomCore.xcframework.zip \
            .build/xcframework/MetalMomCore.xcframework.sha256
```

**Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add XCFramework build and RC support to release workflow"
```

---

### Task 6: Sigstore Signing

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add sigstore signing steps after wheel build**

Add these steps in `release.yml` after the "Verify wheel" step and before the GitHub Release step:

```yaml
      # ── Signing ─────────────────────────────────────────────
      - name: Install sigstore
        run: pip install sigstore

      - name: Sign wheel
        run: python -m sigstore sign dist/metalmom-*.whl

      - name: Sign XCFramework
        run: python -m sigstore sign .build/xcframework/MetalMomCore.xcframework.zip
```

**Step 2: Add id-token permission for OIDC**

Update the permissions block:

```yaml
    permissions:
      contents: write
      id-token: write   # required for sigstore OIDC
```

**Step 3: Attach .sigstore bundles to the release**

Update the `gh release create` command to also attach the sigstore bundles:

```yaml
          gh release create "$TAG" \
            --title "MetalMom $TAG" \
            $PRERELEASE_FLAG \
            $NOTES_FILE \
            dist/metalmom-*.whl \
            dist/*.sigstore \
            .build/xcframework/MetalMomCore.xcframework.zip \
            .build/xcframework/MetalMomCore.xcframework.sha256 \
            .build/xcframework/*.sigstore
```

**Step 4: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add sigstore signing for wheel and XCFramework"
```

---

### Task 7: TestPyPI for Release Candidates

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add TestPyPI publish step for RC tags**

Replace the existing PyPI publish step with both TestPyPI (for RCs) and real PyPI (for releases):

```yaml
      # ── TestPyPI (for release candidates) ────────────────────
      - name: Publish to TestPyPI
        if: contains(steps.version.outputs.tag, '-rc') && env.HAS_TEST_PYPI_TOKEN == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.TEST_PYPI_TOKEN }}
          repository-url: https://test.pypi.org/legacy/

      # ── PyPI (for final releases only) ───────────────────────
      - name: Publish to PyPI
        if: "!contains(steps.version.outputs.tag, '-rc') && env.HAS_PYPI_TOKEN == 'true'"
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}
```

**Step 2: Add the TEST_PYPI_TOKEN env check**

In the `env:` block at the top of the job, add:

```yaml
      HAS_TEST_PYPI_TOKEN: ${{ secrets.TEST_PYPI_TOKEN != '' }}
```

**Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add TestPyPI publish for release candidates"
```

---

### Task 8: Hugging Face Model Upload

**Files:**
- Create: `scripts/upload_models_hf.py`

**Step 1: Create the upload script**

Create `scripts/upload_models_hf.py`:

```python
#!/usr/bin/env python3
"""Upload CoreML models to Hugging Face Hub.

Usage:
    python scripts/upload_models_hf.py [--repo-id zkeown/metalmom-coreml-models] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


MODELS_DIR = Path(__file__).parent.parent / "models" / "converted"

MODEL_CARD = """\
---
license: cc-by-nc-sa-4.0
tags:
  - audio
  - music-information-retrieval
  - coreml
  - metalmom
---

# MetalMom CoreML Models

67 CoreML models converted from [madmom](https://github.com/CPJKU/madmom) for use with [MetalMom](https://github.com/zakkeown/MetalMom).

## Model Families

| Family | Count | Description |
|--------|-------|-------------|
| beats | 16 | RNN beat tracking (LSTM + BLSTM) |
| chords | 1 | Deep chroma chord recognition |
| chroma | 1 | DNN chroma extraction |
| downbeats | 16 | RNN downbeat tracking (BLSTM + BGRU) |
| key | 1 | CNN key recognition |
| notes | 14 | RNN note/onset detection |
| onsets | 16 | RNN onset detection |

## Usage

```python
from metalmom.models import download_models, list_models

# List available models
list_models()

# Download all models
download_models()

# Download a specific family
download_models("beats")
```

## Provenance

Converted from madmom's pickled NumPy weights using `coremltools` NeuralNetworkBuilder.
Peephole LSTM connections are preserved (not available via PyTorch conversion path).

## License

CC-BY-NC-SA 4.0 (same as original madmom model weights).
"""


def build_config(models_dir: Path) -> dict:
    """Build a config.json mapping model paths to metadata."""
    config = {"models": {}}
    for family_dir in sorted(models_dir.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("."):
            continue
        for model_file in sorted(family_dir.glob("*.mlmodel")):
            key = f"{family_dir.name}/{model_file.stem}"
            config["models"][key] = {
                "file": f"{family_dir.name}/{model_file.name}",
                "family": family_dir.name,
                "size_bytes": model_file.stat().st_size,
            }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload CoreML models to HF Hub")
    parser.add_argument(
        "--repo-id", default="zkeown/metalmom-coreml-models",
        help="Hugging Face repo ID",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found: {MODELS_DIR}")
        return 1

    model_files = list(MODELS_DIR.rglob("*.mlmodel"))
    print(f"Found {len(model_files)} models in {MODELS_DIR}")

    config = build_config(MODELS_DIR)
    print(f"Config has {len(config['models'])} entries")

    if args.dry_run:
        print(f"\nDRY RUN -- would upload to {args.repo_id}:")
        for key in config["models"]:
            print(f"  {key}")
        return 0

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(args.repo_id, repo_type="model", exist_ok=True)

    # Upload README
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )

    # Upload config.json
    config_bytes = json.dumps(config, indent=2).encode()
    api.upload_file(
        path_or_fileobj=config_bytes,
        path_in_repo="config.json",
        repo_id=args.repo_id,
    )

    # Upload model files
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=args.repo_id,
        path_in_repo=".",
        ignore_patterns=["*.mlmodelc/*", ".gitignore"],
    )

    print(f"\nDone! Models uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Run dry-run to verify**

Run: `pip install huggingface_hub && python3 scripts/upload_models_hf.py --dry-run`
Expected: Lists 65 models it would upload

**Step 3: Commit (do NOT upload yet — that happens after full plan execution)**

```bash
git add scripts/upload_models_hf.py
git commit -m "feat: add HF Hub model upload script"
```

---

### Task 9: Python `metalmom.models` Module

**Files:**
- Create: `python/metalmom/models.py`
- Modify: `pyproject.toml` (add `[models]` optional dependency)
- Create: `Tests/test_models_module.py`

**Step 1: Write the failing test**

Create `Tests/test_models_module.py`:

```python
"""Tests for the metalmom.models module."""

import pytest


def test_list_models_returns_list():
    from metalmom.models import list_models
    result = list_models()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(m, str) for m in result)


def test_list_models_contains_known_families():
    from metalmom.models import list_models
    models = list_models()
    families = {m.split("/")[0] for m in models}
    assert "beats" in families
    assert "chords" in families


def test_list_models_family_filter():
    from metalmom.models import list_models
    beats = list_models("beats")
    assert all(m.startswith("beats/") for m in beats)
    assert len(beats) > 0


def test_model_path_returns_path_when_cached(tmp_path, monkeypatch):
    """Test model_path returns a Path when the model is in the cache."""
    from metalmom import models as mod

    # Create a fake cached model
    fake_model = tmp_path / "beats" / "beats_lstm_1.mlmodel"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_text("fake")

    monkeypatch.setattr(mod, "_get_cache_dir", lambda: tmp_path)

    from metalmom.models import model_path
    result = model_path("beats/beats_lstm_1")
    assert result is not None
    assert result.exists()


def test_model_path_returns_none_when_not_cached(tmp_path, monkeypatch):
    from metalmom import models as mod
    monkeypatch.setattr(mod, "_get_cache_dir", lambda: tmp_path)

    from metalmom.models import model_path
    result = model_path("beats/nonexistent")
    assert result is None


def test_download_models_requires_huggingface_hub():
    """Ensure a clear error when huggingface_hub is not available."""
    # This test just verifies the import guard exists
    from metalmom.models import download_models
    assert callable(download_models)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/test_models_module.py -v`
Expected: FAIL (module not found)

**Step 3: Implement metalmom.models**

Create `python/metalmom/models.py`:

```python
"""Model download and management for MetalMom CoreML models.

Models are hosted on Hugging Face Hub at zkeown/metalmom-coreml-models.
Install the optional dependency: pip install metalmom[models]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO_ID = "zkeown/metalmom-coreml-models"

# Canonical model list (matches models/converted/ structure)
_MODEL_FAMILIES = {
    "beats": [
        "beats_lstm_1", "beats_lstm_2", "beats_lstm_3", "beats_lstm_4",
        "beats_lstm_5", "beats_lstm_6", "beats_lstm_7", "beats_lstm_8",
        "beats_blstm_1", "beats_blstm_2", "beats_blstm_3", "beats_blstm_4",
        "beats_blstm_5", "beats_blstm_6", "beats_blstm_7", "beats_blstm_8",
    ],
    "chords": ["chords_dnn"],
    "chroma": ["chroma_dnn"],
    "downbeats": [],  # populated from HF config at runtime
    "key": ["key_cnn"],
    "notes": [],  # populated from HF config at runtime
    "onsets": [],  # populated from HF config at runtime
}


def _get_cache_dir() -> Path:
    """Return the local model cache directory."""
    return Path.home() / ".cache" / "metalmom" / "models"


def list_models(family: Optional[str] = None) -> list[str]:
    """List available model identifiers.

    Parameters
    ----------
    family : str, optional
        Filter to a specific family (e.g. "beats", "key").
        If None, returns all models.

    Returns
    -------
    list of str
        Model identifiers in "family/name" format.
    """
    # Try to read from HF config if cached, otherwise use hardcoded list
    cache_dir = _get_cache_dir()
    config_path = cache_dir / "config.json"

    if config_path.exists():
        config = json.loads(config_path.read_text())
        models = sorted(config.get("models", {}).keys())
    else:
        # Fallback: enumerate from cache directory
        models = []
        if cache_dir.exists():
            for family_dir in sorted(cache_dir.iterdir()):
                if family_dir.is_dir() and not family_dir.name.startswith("."):
                    for f in sorted(family_dir.glob("*.mlmodel")):
                        models.append(f"{family_dir.name}/{f.stem}")

        # If cache is empty, use hardcoded list
        if not models:
            for fam, names in sorted(_MODEL_FAMILIES.items()):
                for name in names:
                    models.append(f"{fam}/{name}")

    if family:
        models = [m for m in models if m.startswith(f"{family}/")]

    return models


def model_path(model_id: str) -> Optional[Path]:
    """Get the local path for a cached model.

    Parameters
    ----------
    model_id : str
        Model identifier in "family/name" format (e.g. "beats/beats_lstm_1").

    Returns
    -------
    Path or None
        Path to the .mlmodel file if cached, None otherwise.
    """
    cache_dir = _get_cache_dir()
    path = cache_dir / f"{model_id}.mlmodel"
    return path if path.exists() else None


def download_models(family: Optional[str] = None, cache_dir: Optional[Path] = None) -> Path:
    """Download models from Hugging Face Hub.

    Parameters
    ----------
    family : str, optional
        Download only a specific family (e.g. "beats").
        If None, downloads all models.
    cache_dir : Path, optional
        Override the default cache directory (~/.cache/metalmom/models/).

    Returns
    -------
    Path
        Path to the local cache directory containing downloaded models.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install metalmom[models]"
        )

    dest = cache_dir or _get_cache_dir()

    if family:
        allow_patterns = [f"{family}/*.mlmodel", "config.json"]
    else:
        allow_patterns = ["*.mlmodel", "config.json"]

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=["*.mlmodelc/*"],
    )

    return dest
```

**Step 4: Add optional dependency to pyproject.toml**

In `pyproject.toml` under `[project.optional-dependencies]`, add:

```toml
models = ["huggingface_hub>=0.20"]
```

**Step 5: Run tests**

Run: `.venv/bin/pytest Tests/test_models_module.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add python/metalmom/models.py Tests/test_models_module.py pyproject.toml
git commit -m "feat: add metalmom.models module for HF model downloads"
```

---

### Task 10: Update Release Workflow (Full Assembly)

**Files:**
- Modify: `.github/workflows/release.yml`

This task assembles all the release workflow changes from Tasks 5, 6, 7 into a single coherent file. Write the complete updated `release.yml`:

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  release:
    name: Build & Release
    runs-on: macos-14

    permissions:
      contents: write
      id-token: write   # sigstore OIDC

    env:
      HAS_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN != '' }}
      HAS_TEST_PYPI_TOKEN: ${{ secrets.TEST_PYPI_TOKEN != '' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # needed for git-cliff

      # ── Extract version from tag ──────────────────────────────
      - name: Extract version
        id: version
        run: |
          TAG="${GITHUB_REF#refs/tags/}"
          VERSION="${TAG#v}"
          echo "tag=$TAG"         >> "$GITHUB_OUTPUT"
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"

      # ── Swift + Dylib ─────────────────────────────────────────
      - name: Build Swift package (release)
        run: swift build -c release

      - name: Build dylib
        run: ./scripts/build_dylib.sh

      # ── XCFramework ──────────────────────────────────────────
      - name: Build XCFramework
        run: ./scripts/build_xcframework.sh

      - name: Package XCFramework
        run: |
          cd .build/xcframework
          zip -r MetalMomCore.xcframework.zip MetalMomCore.xcframework
          shasum -a 256 MetalMomCore.xcframework.zip | cut -d ' ' -f1 > MetalMomCore.xcframework.sha256
          echo "XCFramework checksum: $(cat MetalMomCore.xcframework.sha256)"

      # ── Python wheel ──────────────────────────────────────────
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install build dependencies
        run: pip install build hatchling sigstore

      - name: Build wheel
        run: ./scripts/build_wheel.sh --no-swift

      - name: Verify wheel
        run: |
          WHEEL=$(ls dist/metalmom-*.whl | head -1)
          echo "Built: $WHEEL"
          unzip -l "$WHEEL" | grep -E "(dylib|\.h|METADATA)"

      # ── Signing ───────────────────────────────────────────────
      - name: Sign wheel
        run: python -m sigstore sign dist/metalmom-*.whl

      - name: Sign XCFramework
        run: python -m sigstore sign .build/xcframework/MetalMomCore.xcframework.zip

      # ── Changelog ─────────────────────────────────────────────
      - name: Install git-cliff
        run: brew install git-cliff

      - name: Generate release notes
        run: git-cliff --latest --strip header > release-notes.md

      # ── GitHub Release ────────────────────────────────────────
      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          TAG="${{ steps.version.outputs.tag }}"

          PRERELEASE_FLAG=""
          if echo "$TAG" | grep -q "\-rc"; then
            PRERELEASE_FLAG="--prerelease"
          fi

          gh release create "$TAG" \
            --title "MetalMom $TAG" \
            $PRERELEASE_FLAG \
            --notes-file release-notes.md \
            dist/metalmom-*.whl \
            dist/*.sigstore \
            .build/xcframework/MetalMomCore.xcframework.zip \
            .build/xcframework/MetalMomCore.xcframework.sha256 \
            .build/xcframework/*.sigstore

      # ── TestPyPI (for release candidates) ─────────────────────
      - name: Publish to TestPyPI
        if: contains(steps.version.outputs.tag, '-rc') && env.HAS_TEST_PYPI_TOKEN == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.TEST_PYPI_TOKEN }}
          repository-url: https://test.pypi.org/legacy/

      # ── PyPI (for final releases only) ────────────────────────
      - name: Publish to PyPI
        if: "!contains(steps.version.outputs.tag, '-rc') && env.HAS_PYPI_TOKEN == 'true'"
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}
```

**Step 1: Write the complete file**

Replace `.github/workflows/release.yml` with the content above.

**Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"`
(Install pyyaml if needed: `pip install pyyaml`)

**Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: full release pipeline with XCFramework, sigstore, git-cliff, RC/TestPyPI"
```

---

### Task 11: RELEASING.md

**Files:**
- Create: `RELEASING.md`

**Step 1: Write the release checklist**

Create `RELEASING.md`:

```markdown
# Releasing MetalMom

## Prerequisites

- `git-cliff` installed (`brew install git-cliff`)
- GitHub repo secrets configured:
  - `PYPI_TOKEN` — PyPI API token
  - `TEST_PYPI_TOKEN` — TestPyPI API token
- Clean working tree on `main` with CI green

## Release Candidate

```bash
# 1. Bump version and tag RC
./scripts/bump_version.sh 0.2.0 --rc 1

# 2. Push to trigger release workflow
git push origin main --tags

# 3. Verify
#    - GitHub pre-release created with wheel + XCFramework + .sigstore bundles
#    - TestPyPI: pip install -i https://test.pypi.org/simple/ metalmom==0.2.0rc1
#    - XCFramework zip + sha256 attached
```

## Final Release

```bash
# 1. Bump version and tag
./scripts/bump_version.sh 0.2.0

# 2. Push to trigger release workflow
git push origin main --tags

# 3. Verify
#    - GitHub Release created (not pre-release)
#    - PyPI: pip install metalmom==0.2.0
#    - XCFramework zip + sha256 attached
#    - Swift Package Index updated (may take up to 1 hour)

# 4. Verify sigstore signatures
pip install sigstore
python -m sigstore verify identity dist/metalmom-*.whl \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  --cert-identity "https://github.com/zakkeown/MetalMom/.github/workflows/release.yml@refs/tags/v0.2.0"
```

## What the Release Workflow Does

1. Builds Swift package in release mode
2. Builds dylib and copies to Python package
3. Builds XCFramework (iOS device + iOS Simulator + macOS)
4. Zips XCFramework and computes SHA256 checksum
5. Builds Python wheel with correct macOS arm64 platform tag
6. Signs wheel and XCFramework with sigstore (keyless OIDC)
7. Generates release notes with git-cliff
8. Creates GitHub Release with all artifacts
9. Publishes to TestPyPI (RC tags) or PyPI (release tags)

## Version Sources

The bump script updates all of these:
- `pyproject.toml` — `version = "X.Y.Z"`
- `python/metalmom/__init__.py` — `__version__ = "X.Y.Z"`
- `docs/api/conf.py` — `release = "X.Y.Z"`
- `CHANGELOG.md` — generated by git-cliff

SPM version comes from the git tag (no file to update).

## Model Updates

Models are hosted separately on Hugging Face Hub:

```bash
# Upload or update models
python scripts/upload_models_hf.py --repo-id zkeown/metalmom-coreml-models
```
```

**Step 2: Commit**

```bash
git add RELEASING.md
git commit -m "docs: add release checklist"
```

---

### Task 12: Swift Package Index Submission

**Manual step — no code changes required.**

After the first tagged release is pushed:

1. Go to https://github.com/SwiftPackageIndex/PackageList
2. Fork the repo
3. Add `https://github.com/zakkeown/MetalMom` to `packages.json`
4. Submit a PR

This is a one-time action. After merged, every future tagged release auto-appears on swiftpackageindex.com.

**Step 1: Document the intent**

This step is performed after the first release. No commit needed.

---

### Task 13: Upload Models to Hugging Face Hub

**Manual step — run after all code changes are committed.**

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/upload_models_hf.py --repo-id zkeown/metalmom-coreml-models
```

Verify at https://huggingface.co/zkeown/metalmom-coreml-models that README, config.json, and all 65 `.mlmodel` files are present.

---

### Task 14: Cut v0.1.0-rc1

**The culminating task — run after all previous tasks are committed.**

```bash
# Verify everything is clean and green
swift build && swift test --skip MetalSTFTTests --skip MetalMatmulTests --skip MetalElementwiseTests --skip MetalReductionTests --skip MetalConvolutionTests --skip MetalBackendTests --skip GPUParityTests --skip ThresholdCalibrationTests
./scripts/build_dylib.sh
.venv/bin/pytest Tests/ -v

# Tag the release candidate
./scripts/bump_version.sh 0.1.0 --rc 1
git push origin main --tags

# Watch the release workflow at:
# https://github.com/zakkeown/MetalMom/actions/workflows/release.yml
```

Verify:
- GitHub pre-release created with all artifacts
- TestPyPI install works: `pip install -i https://test.pypi.org/simple/ metalmom==0.1.0rc1`
- XCFramework zip and sha256 attached
- Sigstore bundles attached
