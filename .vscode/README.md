# VS Code/Cursor Configuration for uv

This workspace is configured to use `uv run` by default for Python execution.

## Configuration Files

### settings.json
- **Python interpreter**: Points to `./nautilus_test/.venv` (the uv virtual environment)
- **Formatter**: Uses Ruff for formatting on save
- **Terminal**: Defaults to `nautilus_test` directory
- **Testing**: Configured to use `uv run pytest`

### launch.json
- **Debug configurations** for running Python files with uv
- Two options:
  1. Direct `.venv/bin/python` execution
  2. `uv run python` execution

### tasks.json
- **Build task**: Run Python with uv (`Ctrl+Shift+P` → "Tasks: Run Task")
- **Test task**: Run pytest with uv
- **Format task**: Format with ruff

### keybindings.json
- **Ctrl+F5**: Run current Python file with `uv run` (using command-variable)
- **Ctrl+Shift+F5**: Run Python tests with `uv run`
- **Ctrl+Alt+F**: Format current file with ruff
- **Ctrl+Alt+L**: Lint current file with ruff

## Usage

### Running Python Files
1. **From terminal**: `uv run python filename.py` (recommended)
2. **With keyboard**: `Ctrl+F5` when editing a Python file
3. **From command palette**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Python with uv"
4. **Debug mode**: `F5` (uses launch configuration)

### Python Interpreter Selection
If VS Code doesn't automatically detect the interpreter:
1. `Ctrl+Shift+P`
2. "Python: Select Interpreter"
3. Choose "Enter interpreter path"
4. Enter: `./nautilus_test/.venv` (the folder, not the python executable)

## Key Benefits
- ✅ Consistent dependency management
- ✅ Isolated virtual environment
- ✅ Format on save with Ruff
- ✅ Proper test discovery
- ✅ No "ModuleNotFoundError" issues
- ✅ UV Toolkit integration for elegant package management
- ✅ Enhanced syntax highlighting for pyproject.toml

## Extensions & Tools
- **Command Variable**: Dynamic variable substitution for tasks and launch configs
- **Ruff**: Modern, fast Python linting and formatting (replaces Pylint + Black)
- **Key Features**: File path resolution, module path conversion, intelligent task execution

## Troubleshooting

### "Module not found" errors
- Ensure you're using `uv run python script.py`
- Or use `Ctrl+F5` keyboard shortcut
- Check that the interpreter points to `.venv` folder

### Dependencies not found
- Run `uv sync --all-extras` in the nautilus_test directory
- Restart VS Code/Cursor after sync