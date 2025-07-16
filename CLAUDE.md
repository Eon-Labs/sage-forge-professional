# Project Memory

## 🧠 Workspace 
`uv run python -c "import pathlib;g=next((x for x in [pathlib.Path.cwd()]+list(pathlib.Path.cwd().parents) if (x/'.git').exists()),pathlib.Path.cwd());print(g)"`

**Tools**: uv, black, ruff, mypy, pytest  
**Python**: 3.11+, type hints required  
**Commands**: Use `make` or `uv run` for operations

## Cache System
**Uses `platformdirs`** for platform-appropriate cache directories (not workspace dirs)

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Always verify data source authenticity before questioning calculation accuracy.