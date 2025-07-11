#!/bin/bash
# Wrapper script to run Python with uv
cd "$(dirname "$0")"
exec uv run python "$@"