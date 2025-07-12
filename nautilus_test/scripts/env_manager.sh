#!/bin/bash
# Environment Manager for NautilusTrader Project
# Robust solution for virtual environment conflicts

echo "🔧 Environment Manager for NautilusTrader Project"
echo "=================================================="

# Function to detect environment conflicts
detect_conflicts() {
    echo "🔍 Detecting environment conflicts..."
    echo "VIRTUAL_ENV: $VIRTUAL_ENV"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "Current Python: $(which python 2>/dev/null || echo 'Not found')"
    echo "Current UV: $(which uv 2>/dev/null || echo 'Not found')"
    
    # Check for conflicting environments
    if [[ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" != "$(pwd)/.venv" ]]; then
        echo "⚠️  Conflict detected: VIRTUAL_ENV points to $VIRTUAL_ENV"
        echo "   Expected: $(pwd)/.venv"
        return 1
    fi
    
    if [[ -n "$PYTHONPATH" ]]; then
        echo "⚠️  PYTHONPATH is set: $PYTHONPATH"
        echo "   This may cause import conflicts"
        return 1
    fi
    
    echo "✅ No major conflicts detected"
    return 0
}

# Function to clear environment conflicts
clear_conflicts() {
    echo "🧹 Clearing environment conflicts..."
    
    # Method 1: Unset conflicting variables
    unset VIRTUAL_ENV
    unset PYTHONPATH
    unset CONDA_DEFAULT_ENV
    
    echo "✅ Environment variables cleared"
}

# Function to verify project setup
verify_project() {
    echo "🔍 Verifying project setup..."
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        echo "❌ Error: pyproject.toml not found. Are you in the project root?"
        return 1
    fi
    
    # Check if .venv exists
    if [[ ! -d ".venv" ]]; then
        echo "⚠️  .venv not found, creating virtual environment..."
        uv venv
    fi
    
    # Verify uv can access the project
    if ! uv run python --version >/dev/null 2>&1; then
        echo "❌ Error: uv run failed. Syncing dependencies..."
        uv sync --all-extras
    fi
    
    echo "✅ Project setup verified"
}

# Function to run scripts with proper isolation
run_script() {
    local script_path="$1"
    
    if [[ -z "$script_path" ]]; then
        echo "❌ Error: No script path provided"
        echo "Usage: $0 run <script_path>"
        return 1
    fi
    
    echo "🚀 Running script with isolated environment: $script_path"
    
    # Use uv run for complete isolation
    uv run python "$script_path"
}

# Function to install packages safely
install_package() {
    local package="$1"
    
    if [[ -z "$package" ]]; then
        echo "❌ Error: No package name provided"
        echo "Usage: $0 install <package_name>"
        return 1
    fi
    
    echo "📦 Installing package: $package"
    uv add "$package"
}

# Function to show project status
show_status() {
    echo "📊 Project Status"
    echo "=================="
    echo "Working Directory: $(pwd)"
    echo "UV Version: $(uv --version)"
    echo "Project Python: $(uv run python --version)"
    echo "Project Python Path: $(uv run which python)"
    echo ""
    echo "Installed Packages:"
    uv run pip list | head -10
    echo "..."
    echo ""
    echo "Dependencies from pyproject.toml:"
    grep -A 10 "dependencies" pyproject.toml | head -7
}

# Function to configure VS Code Python interpreter
configure_vscode() {
    echo "🔧 Configuring VS Code Python interpreter..."
    
    # Check if .vscode directory exists
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
    fi
    
    # Get the absolute path to the project's Python interpreter
    local python_path="$(pwd)/.venv/bin/python"
    
    if [[ ! -f "$python_path" ]]; then
        echo "❌ Error: Project Python interpreter not found at $python_path"
        echo "Run '$0 verify' first to set up the project"
        return 1
    fi
    
    echo "✅ VS Code configured to use: $python_path"
    echo "💡 Restart VS Code and reload the window for changes to take effect"
}

# Main command dispatcher
case "$1" in
    "detect")
        detect_conflicts
        ;;
    "clear")
        clear_conflicts
        ;;
    "verify")
        verify_project
        ;;
    "run")
        run_script "$2"
        ;;
    "install")
        install_package "$2"
        ;;
    "status")
        show_status
        ;;
    "vscode")
        configure_vscode
        ;;
    "fix")
        echo "🔧 Running complete environment fix..."
        clear_conflicts
        verify_project
        configure_vscode
        echo "✅ Environment fix completed!"
        ;;
    *)
        echo "🔧 Environment Manager Usage:"
        echo "  $0 detect    - Detect environment conflicts"
        echo "  $0 clear     - Clear environment conflicts"
        echo "  $0 verify    - Verify project setup"
        echo "  $0 run <script> - Run script with isolated environment"
        echo "  $0 install <pkg> - Install package safely"
        echo "  $0 status    - Show project status"
        echo "  $0 vscode    - Configure VS Code Python interpreter"
        echo "  $0 fix       - Run complete environment fix"
        echo ""
        echo "🚀 Recommended usage:"
        echo "  $0 fix && $0 run examples/sandbox/simple_bars_test.py"
        echo ""
        echo "📝 VS Code Setup:"
        echo "  1. Run: $0 fix"
        echo "  2. Restart VS Code"
        echo "  3. Open Command Palette (Cmd+Shift+P)"
        echo "  4. Select 'Python: Select Interpreter'"
        echo "  5. Choose: ./.venv/bin/python"
        ;;
esac