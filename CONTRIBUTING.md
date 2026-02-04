# Contributing to openai-agents-context-compaction

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Getting Started

1. Fork the repository on GitHub

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/openai-agents-context-compaction.git
   cd openai-agents-context-compaction
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/damianoneill/openai-agents-context-compaction.git
   ```

4. Install dependencies:
   ```bash
   make sync
   ```

5. Install pre-commit hooks:
   ```bash
   make pre-commit-install
   ```

## Development Workflow

### Running Checks

Run all checks (format, lint, typecheck, tests with coverage):
```bash
make check
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:
```bash
make pre-commit
```

## Code Standards

- **Formatting**: We use [ruff](https://github.com/astral-sh/ruff) for formatting (line length: 100)
- **Linting**: Ruff is also used for linting
- **Type Hints**: All code must have type hints and pass mypy
- **Test Coverage**: Minimum 90% coverage is enforced
- **Docstrings**: Use Google-style docstrings for public APIs

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) and use [Commitizen](https://commitizen-tools.github.io/commitizen/) for interactive commit creation and automated versioning.

### Interactive Commits (Recommended)

Use commitizen to create properly formatted commits:

```bash
uv run cz commit
# Or the shorthand:
uv run cz c
```

This provides an interactive prompt that guides you through creating a well-formatted commit message.

### Manual Commits

If you prefer to write commits manually, use this structure:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature (triggers minor version bump)
- **fix**: A bug fix (triggers patch version bump)
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (formatting)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance (triggers patch version bump)
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

### Examples

```bash
# Using commitizen (recommended)
uv run cz commit

# Or manual commits
git commit -m "feat: add sliding window compaction strategy"
git commit -m "fix: resolve token counting edge case"
git commit -m "docs: update configuration examples"
git commit -m "test: add coverage for summarization"
```

## Pull Request Process

1. Sync your fork with upstream:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

3. Make your changes and ensure all checks pass:
   ```bash
   make check
   ```

4. Commit with a descriptive message following Conventional Commits

5. Push and create a Pull Request against `main`

6. Ensure CI passes and address any review feedback

## Compatibility

This package maintains compatibility with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). When the SDK releases new versions:

- The weekly CI job tests against the latest SDK
- If tests fail, an issue is automatically created
- Please help fix compatibility issues if you can!

## Questions?

Open an issue for questions, bug reports, or feature requests.
