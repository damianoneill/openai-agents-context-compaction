# Maintainers Guide

This document provides guidelines for maintaining the `openai-agents-context-compaction` package, including the release process, CI/CD workflows, and ongoing maintenance responsibilities.

## Table of Contents

- [Release Process](#release-process)
- [CI/CD Workflows](#cicd-workflows)
- [Maintenance Responsibilities](#maintenance-responsibilities)
- [Dependency Management](#dependency-management)
- [Security](#security)
- [Support](#support)

## Release Process

This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/) to automate versioning and changelog generation based on [Conventional Commits](https://www.conventionalcommits.org/).

### Quick Release

For a standard release (recommended):

```bash
# Run all checks and bump version based on commit history
make release
```

This will:
1. Run all checks (format, lint, typecheck, tests with coverage)
2. Analyze commits since the last tag to determine version bump
3. Update version in `pyproject.toml` and `_version.py`
4. Update `CHANGELOG.md`
5. Create a commit and git tag

### Manual Version Control

If you need more control over the version bump:

```bash
# Auto-detect bump type from commits (feat -> minor, fix -> patch)
make bump

# Force a minor version bump (0.1.0 -> 0.2.0)
make bump-minor

# Force a major version bump (0.1.0 -> 1.0.0)
make bump-major

# Only regenerate the changelog
make changelog
```

### How Commitizen Determines Version Bumps

Commitizen analyzes commit messages since the last tag:

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `feat:` | Minor (0.1.0 → 0.2.0) | `feat: add sliding window compaction` |
| `fix:` | Patch (0.1.0 → 0.1.1) | `fix: resolve token counting issue` |
| `perf:` | Patch | `perf: optimize summary generation` |
| `BREAKING CHANGE:` | Major (0.1.0 → 1.0.0) | `feat!: redesign CompactionConfig API` |
| `docs:`, `style:`, `refactor:`, `test:`, `chore:` | No bump | `docs: update README` |

### Step-by-Step Release

1. **Ensure all changes are committed** with conventional commit messages:
   ```bash
   git status  # Should be clean
   ```

2. **Run all checks**:
   ```bash
   make check
   ```

3. **Preview what will happen** (dry run):
   ```bash
   uv run cz bump --dry-run
   ```

4. **Perform the release**:
   ```bash
   make release
   # Or for more control:
   make bump
   ```

5. **Push the commit and tag**:
   ```bash
   git push origin main --tags
   ```

6. **Create GitHub Release**:
   - Go to the repository's **Releases** page
   - Click **Draft a new release**
   - Select the tag that was just created (e.g., `v0.2.0`)
   - Copy the relevant section from `CHANGELOG.md` into the release notes
   - Click **Publish release**

7. **Automated Publishing**:
   - The `publish.yml` workflow triggers automatically
   - Builds the package with `uv build`
   - Publishes to PyPI with `uv publish`

### Post-Release Verification

- Verify the package on PyPI: https://pypi.org/project/openai-agents-context-compaction/
- Test installation: `pip install openai-agents-context-compaction==X.Y.Z`
- Announce the release if appropriate

### PyPI Publishing Prerequisites

The publish workflow uses `uv publish` with token authentication:

1. **Create a PyPI API token:**
   - Go to https://pypi.org/manage/account/token/
   - Create a token scoped to the `openai-agents-context-compaction` project (or account-wide for first publish)

2. **Add the token to GitHub secrets:**
   - Go to your repository's Settings → Secrets and variables → Actions
   - Create a new secret named `PYPI_API_TOKEN`
   - Paste the token value (including the `pypi-` prefix)

The workflow uses `UV_PUBLISH_TOKEN` environment variable which `uv publish` reads automatically.

## CI/CD Workflows

### CI Workflow (`ci.yml`)

**Triggers:** Push to `main`, Pull Requests to `main`

**Jobs:**
- Tests across Python 3.11 and 3.13
- Pre-commit hooks (on Python 3.11 only)
- Linting with ruff
- Type checking with mypy (on Python 3.11 only)
- Tests with 90% coverage enforcement

All PRs must pass CI before merging.

### Compatibility Workflow (`compatibility.yml`)

**Triggers:** Weekly (Mondays at 6am UTC), Manual dispatch

**Purpose:** Ensures compatibility with the latest OpenAI Agents SDK

**Behavior:**
- Installs the latest `openai-agents` package
- Runs the full test suite
- On success: Updates `.last-tested-sdk-version` and commits
- On failure: Creates a GitHub issue with the `compatibility` and `bug` labels

**Action Required:** When compatibility issues arise:
1. Review the created issue and linked workflow run
2. Identify breaking changes in the SDK
3. Update the package code to maintain compatibility
4. Test locally and submit a PR

### Publish Workflow (`publish.yml`)

**Triggers:** Release publication (when you publish a GitHub Release)

**Purpose:** Automatically builds and publishes to PyPI using `uv`

**Steps:**
1. Checks out the code
2. Installs `uv`
3. Builds the package with `uv build`
4. Publishes to PyPI with `uv publish`

See [PyPI Publishing Prerequisites](#pypi-publishing-prerequisites) for setup.

## Maintenance Responsibilities

### Code Quality

- Enforce [Conventional Commits](https://www.conventionalcommits.org/) for all PRs
- Review PRs for:
  - Code quality and readability
  - Test coverage (maintain 90% minimum)
  - Type annotations
  - Documentation for public APIs
- Run `make check` before merging any PR

### Pre-commit Hooks

The repository uses pre-commit hooks for:
- `ruff` - formatting and linting
- `mypy` - type checking
- `trailing-whitespace` - whitespace cleanup
- `end-of-file-fixer` - ensures files end with newline
- `check-yaml` - YAML syntax validation
- `check-merge-conflict` - prevents merge conflict markers
- `conventional-pre-commit` - enforces conventional commit messages

Contributors should install hooks with:
```bash
make pre-commit-install
```

### Conventional Commits

All commits must follow the conventional commits format. Use commitizen for interactive commit creation:

```bash
uv run cz commit
# Or use the shorthand:
uv run cz c
```

This provides an interactive prompt to build a properly formatted commit message.

## Dependency Management

### Core Dependencies

| Package | Purpose | Update Frequency |
|---------|---------|------------------|
| `openai-agents` | SDK being extended | Monitor weekly |

### Updating Dependencies

1. Update version constraints in `pyproject.toml`
2. Run `uv sync --all-extras` to update `uv.lock`
3. Run `make check` to verify compatibility
4. Test manually if possible

### Version Constraints

- Use minimum version constraints (`>=X.Y.Z`) for flexibility
- Avoid upper bounds unless necessary for known incompatibilities
- The weekly compatibility workflow will catch SDK breaking changes

## Security

### Vulnerability Response

1. Monitor security advisories for dependencies
2. Address vulnerabilities promptly:
   - Critical/High: Patch within 24-48 hours
   - Medium: Patch within 1 week
   - Low: Address in next regular release

## Support

### Issue Triage

Label issues appropriately:
- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `compatibility` - SDK compatibility issues (auto-created by CI)
- `question` - General questions

### Response Times

- **Compatibility issues**: Address within 1 week (blocks users on new SDK versions)
- **Bugs**: Acknowledge within 48 hours, fix based on severity
- **Enhancements**: Review and prioritize quarterly

### Questions

For maintenance questions:
- Open an issue or discussion in the repository
- Reference this document and `CONTRIBUTING.md`

---

## Quick Reference

| Task | Command |
|------|---------|
| Run all checks | `make check` |
| Run tests only | `make test` |
| Run tests with coverage | `make coverage` |
| Format code | `make format` |
| Lint code | `make lint` |
| Type check | `make typecheck` |
| Install pre-commit hooks | `make pre-commit-install` |
| Clean build artifacts | `make clean` |
| **Release (recommended)** | `make release` |
| Auto-bump version | `make bump` |
| Bump minor version | `make bump-minor` |
| Bump major version | `make bump-major` |
| Update changelog only | `make changelog` |
| Interactive commit | `uv run cz commit` |
| Preview version bump | `uv run cz bump --dry-run` |
