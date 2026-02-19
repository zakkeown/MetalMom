# MetalMom Deployment & Infrastructure Design

**Date:** 2026-02-18
**Status:** Approved

## Goals

Ship MetalMom v0.1.0 with professional release engineering for both Python MIR researchers (PyPI) and Swift/iOS developers (SPM/XCFramework). Single version number across both platforms.

## 1. Wheel Platform Tag Fix

The current wheel is `py3-none-any`, which is incorrect for a macOS-only binary package.

**Fix:** Tag wheels as `cp311-abi3-macosx_14_0_arm64` using hatchling's wheel build hook or post-build rename in `build_wheel.sh`. The `abi3` stable ABI tag means one wheel covers Python 3.11+, since cffi uses the stable ABI and the bundled dylib is Python-version-independent.

**Result:** PyPI correctly restricts installation to macOS arm64. Linux/Windows users see "not available for your platform" instead of a broken install.

## 2. Multi-Python Wheel Strategy

Single ABI3 wheel covers all supported Python versions (3.11, 3.12, 3.13, future 3.14+). No per-version matrix needed.

Release workflow matrix structured for future expansion:
```yaml
strategy:
  matrix:
    include:
      - os: macos-14
        platform-tag: macosx_14_0_arm64
```

Can add Intel (`macosx_14_0_x86_64`) later if demand warrants cross-compilation or a second runner.

## 3. Version Bump Automation

`scripts/bump_version.sh` takes a semver argument and updates all version sources in one shot:

1. `pyproject.toml` — `version = "X.Y.Z"`
2. `docs/api/conf.py` — `version` and `release` fields
3. Runs git-cliff to update `CHANGELOG.md`
4. Creates git commit: `release: vX.Y.Z`
5. Creates git tag: `vX.Y.Z`

`Package.swift` has no version field — SPM derives version from git tags, which is correct.

**Release flow:**
```bash
./scripts/bump_version.sh 0.2.0
git push origin main --tags
# release.yml fires automatically
```

## 4. Release Candidate Workflow

Tags matching `v*-rc*` (e.g. `v0.1.0-rc1`) trigger an RC path in the release workflow:

- Publishes to **TestPyPI** (requires `TEST_PYPI_TOKEN` secret)
- Creates a GitHub Release marked as **pre-release**
- XCFramework attached as zip to the pre-release
- Full build pipeline (Swift, dylib, wheel, XCFramework, signing)

`bump_version.sh` supports `--rc`: `./scripts/bump_version.sh 0.1.0 --rc 1` tags `v0.1.0-rc1`.

**Promotion flow:**
```bash
./scripts/bump_version.sh 0.1.0 --rc 1   # test
# verify TestPyPI install, check GitHub pre-release
./scripts/bump_version.sh 0.1.0           # ship
```

**Secrets required:** `PYPI_TOKEN` (real PyPI), `TEST_PYPI_TOKEN` (TestPyPI).

## 5. XCFramework in Release Pipeline

Add XCFramework build to `release.yml`:

1. Run `scripts/build_xcframework.sh` (archives iOS device, iOS Simulator, macOS)
2. Zip: `MetalMomCore.xcframework.zip`
3. Compute SHA256 checksum → `MetalMomCore.xcframework.sha256`
4. Attach both to GitHub Release alongside the wheel

Release notes include a SPM binary target snippet:
```swift
.binaryTarget(
    name: "MetalMomCore",
    url: "https://github.com/zakkeown/MetalMom/releases/download/vX.Y.Z/MetalMomCore.xcframework.zip",
    checksum: "<sha256>"
)
```

**Swift Package Index:** One-time PR to SwiftPackageIndex/PackageList to add the repo. Every tagged release auto-appears in SPM search.

## 6. Hugging Face Model Distribution

67 CoreML models hosted on Hugging Face Hub at `zkeown/metalmom-coreml-models`.

**HF repo structure:**
```
beats/          # RNN beat processor models
chords/         # chord recognition models
chroma/         # chroma models
downbeats/      # downbeat models
key/            # key detection models
notes/          # note/onset models
README.md       # model card: families, sizes, provenance, license
config.json     # model metadata: input/output shapes, sample rate, hop size
```

**Python integration:** `metalmom.models` module:
```python
from metalmom.models import download_models, list_models

list_models()              # → ['beats/rnn_1', 'beats/rnn_2', ...]
download_models()          # downloads all to ~/.cache/metalmom/models/
download_models("beats")   # downloads just the beats family
```

- `huggingface_hub` as optional dependency: `pip install metalmom[models]`
- No auto-download on first use — explicit `download_models()` call required
- Swift side gets a configuration option to point at the HF cache path

## 7. Auto-Changelog with git-cliff

Use git-cliff (Rust binary) for changelog generation from conventional commits.

**Setup:**
- `cliff.toml` at repo root defining section groupings
- Install: `brew install git-cliff` locally, prebuilt binary in CI
- Conventional commit prefixes: `feat:`, `fix:`, `perf:`, `docs:`, `ci:`, `refactor:`, `test:`

**Integration:** `bump_version.sh` calls `git cliff --bump --output CHANGELOG.md` before committing the version bump. Release workflow uses generated notes via `gh release create --notes-file`.

No enforcement hook on commit messages — prefixes are a convention, not a gate.

## 8. CI Failure Notifications

Extend parity test workflow to auto-file GitHub Issues on failure:

- `scripts/ci/file_parity_issues.py` (follows same pattern as existing `file_api_drift_issues.py`)
- Runs `if: failure()` at end of `parity.yml`
- Creates issue titled `Parity test failure: <date>` with failed test details and workflow run link
- Label: `parity-failure`
- Deduplication: if open issue with `parity-failure` label exists, adds comment instead

GitHub Issues as single notification channel — repo watch triggers email automatically.

## 9. Signed Releases with Sigstore

Keyless signing using GitHub Actions' OIDC identity via the Python `sigstore` package.

- Sign wheel: `python -m sigstore sign dist/metalmom-*.whl` → produces `.sigstore` bundle
- Sign XCFramework zip the same way
- Attach `.sigstore` bundles to GitHub Release

**Verification:**
```bash
python -m sigstore verify identity dist/metalmom-*.whl \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  --cert-identity <workflow-url>
```

No GPG, no key management, no Apple code signing (XCFramework is a build dependency compiled into consumer's app with their identity; dylib loaded by Python process, not Gatekeeper-gated).

## 10. RELEASING.md

Human-readable release checklist:

1. Ensure `main` is green (CI passing)
2. `./scripts/bump_version.sh X.Y.Z --rc 1` for release candidate
3. Push tag, verify TestPyPI install and GitHub pre-release assets
4. `./scripts/bump_version.sh X.Y.Z` for real release
5. Push tag, verify PyPI install, GitHub Release assets (wheel + sigstore bundle + XCFramework zip + sha256)
6. Verify Swift Package Index picks up the new tag

## Summary

| Component | Approach |
|---|---|
| Wheel platform tag | `cp311-abi3-macosx_14_0_arm64` (stable ABI) |
| Multi-Python | Single ABI3 wheel covers 3.11+ |
| Version bumping | `scripts/bump_version.sh` updates pyproject.toml + conf.py, tags |
| Release candidates | `v*-rc*` tags → TestPyPI + GitHub pre-release |
| XCFramework | Built in release workflow, zipped + sha256, attached to release |
| Models | HF Hub `zkeown/metalmom-coreml-models`, explicit `download_models()` |
| Changelog | git-cliff with `cliff.toml` |
| CI notifications | Parity failures auto-file GitHub Issues |
| Signing | Sigstore keyless (OIDC from GitHub Actions) |
| Docs | RELEASING.md checklist + Swift Package Index submission |
