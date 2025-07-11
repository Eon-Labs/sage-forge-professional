# Claude Code Hooks Configuration

## Automatic Python Formatting Hook

This workspace is configured with a **deterministic** Claude Code hook that automatically formats Python files after any edit.

### Configuration

Located in `/workspaces/nt/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": {
          "tool_name": ["Edit", "MultiEdit", "Write"],
          "file_paths": ["*.py"]
        },
        "command": "cd /workspaces/nt/nautilus_test && echo '🎨 Auto-formatting Python files...' && uv run ruff check --fix $CLAUDE_FILE_PATHS && uv run ruff format $CLAUDE_FILE_PATHS && echo '✅ Formatting complete'"
      }
    ]
  }
}
```

### What It Does

- **Triggers**: After any `Edit`, `MultiEdit`, or `Write` operation on Python files
- **Actions**: 
  1. Runs `uv run ruff check --fix` (linting with auto-fixes)
  2. Runs `uv run ruff format` (code formatting)
- **Scope**: Only affects `*.py` files
- **Working Directory**: Always runs in `/workspaces/nt/nautilus_test/`

### Benefits

✅ **Deterministic**: Always runs, doesn't rely on AI memory  
✅ **Minimal**: Only affects Python files, high impact  
✅ **Standards Compliant**: Uses NautilusTrader's 100-char line length  
✅ **Zero Configuration**: Works automatically after setup  
✅ **Fast**: Uses uv for optimal performance  

### Security

The hook uses relative paths and only runs formatting tools - no dangerous operations.

## Verification

After any Python file edit, you should see output like:
```
🎨 Auto-formatting Python files...
✅ Formatting complete
```

This ensures your code always meets NautilusTrader's standards automatically.