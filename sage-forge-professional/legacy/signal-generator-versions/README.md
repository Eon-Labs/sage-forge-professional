# Signal Generator Versions Archive

**Purpose**: Historical versions of TiRex signal generators that are no longer the primary implementation.

## Archived Files

### `tirex_signal_generator_v2_preview.py`
- **Origin**: Preview version of TiRex signal generator
- **Status**: Superseded by current implementation
- **Archive Date**: August 2025
- **Reason**: No references found in active codebase - replaced by current version

## Active Files (NOT archived)

### `tirex_signal_generator_original.py` - **DO NOT ARCHIVE**
- **Status**: **CRITICAL SAFETY MECHANISM** - Part of active rollback procedures
- **References**: Used in validation gates and rollback procedures (5+ references in planning docs)
- **Purpose**: Safety backup for disaster recovery

### `tirex_signal_generator.py` - **DO NOT ARCHIVE**
- **Status**: **PRIMARY IMPLEMENTATION** - Current active version
- **Purpose**: Main TiRex signal generation script

## Notes

Only archive signal generator versions that have NO active references in the codebase. The original version serves as a critical rollback mechanism and must remain in the root directory.