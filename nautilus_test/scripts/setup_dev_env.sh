#!/bin/bash
# Development environment setup script for NautilusTrader testing

set -e

echo "🚀 Setting up NautilusTrader development environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the nautilus_test directory"
    exit 1
fi

# Ensure Python 3.12 is available
echo "🐍 Checking Python version..."
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 not found, using default python..."
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3.12"
fi

echo "   Using: $($PYTHON_CMD --version)"

# Install/update uv if needed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📦 Syncing dependencies with uv..."
uv sync --all-extras

echo "🔍 Validating installation..."
uv run python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import pandas
    print('✅ pandas imported successfully')
except ImportError as e:
    print(f'❌ pandas import failed: {e}')

try:
    import nautilus_trader
    print('✅ nautilus_trader imported successfully')
except ImportError as e:
    print(f'❌ nautilus_trader import failed: {e}')

try:
    import rich
    print('✅ rich imported successfully')
except ImportError as e:
    print(f'❌ rich import failed: {e}')
"

echo "🧪 Running basic tests..."
uv run pytest tests/test_basic.py -v

echo "✅ Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Reload VS Code window (Ctrl+Shift+P → 'Developer: Reload Window')"
echo "2. Select Python interpreter: .venv/bin/python"
echo "3. Run: uv run python examples/sandbox/simple_bars_test.py"
echo ""
echo "🔧 Available commands:"
echo "   uv run python <script>     # Run Python scripts"
echo "   uv run pytest             # Run tests"
echo "   make dev-workflow          # Run full development check"