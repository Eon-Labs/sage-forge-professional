# Nautilus Test Development Container

This project provides a simple development container setup for the nautilus-test project using `uv` for Python dependency management.

## Quick Start

### Option 1: Using Dev Container (Recommended)

1. Open this project in VS Code
2. Install the "Dev Containers" extension if you haven't already
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
4. Wait for the container to build and start
5. The dependencies will be automatically installed via `uv sync`

### Option 2: Local Development

1. Make sure you have `uv` installed:
   ```bash
   pip install uv
   ```

2. Navigate to the nautilus_test directory and sync dependencies:
   ```bash
   cd nautilus_test
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   ```

### Option 3: Using the Run Script

Make the run script executable and run it:
```bash
chmod +x run.sh
./run.sh
```

## Project Structure

```
nt/
├── .devcontainer/
│   ├── devcontainer.json    # Dev container configuration
│   └── Dockerfile          # Container image definition
├── nautilus_test/
│   ├── main.py             # Main application
│   ├── pyproject.toml      # Project dependencies
│   └── uv.lock            # Locked dependencies
├── run.sh                  # Simple run script
└── README.md              # This file
```

## Features

- **Python 3.12**: Latest Python version
- **uv**: Fast Python package manager
- **Dev Container**: Consistent development environment
- **VS Code Integration**: Pre-configured extensions and settings
- **Non-root User**: Secure container setup

## Dependencies

The project uses:
- `nautilus-trader>=1.219.0`: Main trading framework dependency 