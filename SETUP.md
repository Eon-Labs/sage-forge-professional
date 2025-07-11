# Workspace Setup Guide

This workspace is configured for robust NautilusTrader development with minimal linter errors and IDE issues.

## Quick Setup

1. **Open in VS Code**: Open the workspace root (`/workspaces/nt/`) in VS Code
2. **Run setup script**:
   ```bash
   cd nautilus_test
   ./scripts/setup_dev_env.sh
   ```
3. **Reload VS Code**: `Ctrl+Shift+P` → "Developer: Reload Window"
4. **Verify setup**: Check that import errors are resolved

## Configuration Structure

```
/workspaces/nt/                    # 🏠 Workspace root
├── .vscode/settings.json         # 🔧 VS Code workspace settings
├── .claude/                      # 🤖 Claude Code configuration
├── pyrightconfig.json            # 🐍 Python type checking config
├── nautilus_test/                # 📦 Main development project
│   ├── .venv/                    # 🐍 Python virtual environment
│   ├── .python-version           # 🐍 Python version pinning
│   ├── pyproject.toml            # 📦 Project configuration
│   └── scripts/setup_dev_env.sh  # 🚀 Environment setup
└── nt_reference/                 # 📚 Reference NautilusTrader repo
```

## Key Features

### 🔍 **Robust Import Resolution**
- Direct paths to `.venv/lib/python3.12/site-packages`
- Explicit `nautilus_trader` package indexing depth
- Workspace-level `pyrightconfig.json` for consistent type checking

### 🐍 **Python Environment**
- Python 3.12 for consistency with environment
- UV package manager for fast dependency resolution
- Automatic virtual environment activation

### 🧪 **Testing Integration**
- Pytest configured to run with `uv run pytest`
- Test discovery from `nautilus_test/tests/`
- Proper working directory configuration

### 🎨 **Code Quality**
- Ruff for linting and formatting
- Black code formatting (100 char line length)
- Mypy type checking integration

## Troubleshooting Import Errors

If you still see import errors:

1. **Check Python interpreter**: Bottom-left of VS Code should show `3.12.x` from `.venv`
2. **Manually select interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv/bin/python`
3. **Clear VS Code cache**: `Ctrl+Shift+P` → "Developer: Reload Window"
4. **Re-run setup**: `./scripts/setup_dev_env.sh`

## Development Workflow

```bash
# Start development
cd nautilus_test

# Install/update dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run specific script
uv run python examples/sandbox/simple_bars_test.py

# Check code quality
make dev-workflow
```

This configuration eliminates common VS Code Python linting issues by providing explicit paths and proper environment detection.