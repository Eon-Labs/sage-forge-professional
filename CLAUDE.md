# Project Memory

## 🧠 Workspace

`uv run python -c "import pathlib;g=next((x for x in [pathlib.Path.cwd()]+list(pathlib.Path.cwd().parents) if (x/'.git').exists()),pathlib.Path.cwd());print(g)"`

**Tools**: uv, black, ruff, mypy, pytest  
**Python**: 3.11+, type hints required  
**Commands**: Use `make` or `uv run` for operations

## Cache System

\*\*Uses `platformdirs` for platform-appropriate cache directories (not workspace dirs)

## Project Resources

## FPPA: FinPlot Pattern Alignment

- FinPlot pattern is in the `/Users/terryli/eon/nt/repos/finplot`.
- Prefer `/Users/terryli/eon/nt/repos/finplot/finplot/examples/complicated.py` as the default template.
- Proactively conform to the native paradigm of FinPlot—including its provided classes, idiomatic patterns, and native conventions.

## NTPA: NautilusTrader Pattern Alignment

- NautilusTrader pattern is in the `/Users/terryli/eon/nt/repos/nautilus_trader`.
- Proactively conform to the native paradigm of NautilusTrader—including its provided classes, idiomatic patterns, and native conventions.

## Financial Time Series Trading Optimization

Recommend 2025 state-of-the-art, benchmark-validated, top-ranked algorithms implemented in off-the-shelf, future-proof, turnkey Python libraries that require minimal or no manual tuning—avoiding hardcoded thresholds or magic numbers. In other words, prioritize generalizability, auto-tuning capabilities, and integration-friendliness. 

Proactively research recent best practices for the host framework's native paradigm—including its provided classes, idiomatic patterns, and native conventions—and present your findings as a concise, reproducible, cookbook‑style reference.

Proactively conform to the native paradigm of NautilusTrader—including its provided classes, idiomatic patterns, and native conventions.

## Output

- Minimize the number of lines by using Rich Progress and related functions.

# important-instruction-reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (\*.md) or README files. Only create documentation files if explicitly requested by the User.
Always verify data source authenticity before questioning calculation accuracy.