# SAGE-Forge Professional Organization Plan

## Current Issues
- Scattered test files in root directory
- Missing documentation of recent breakthroughs
- Unclear file structure and purposes
- No clear regression testing strategy

## Organization Strategy

### 1. File Structure Reorganization

**Tests Organization:**
```
tests/
├── unit/                           # Unit tests for individual components
├── integration/                    # Integration tests for system components
├── functional/                     # End-to-end functionality tests
│   ├── test_tirex_signal_generation.py
│   ├── test_tirex_backtest_integration.py
│   └── test_nt_compliance_validation.py
├── regression/                     # Regression tests for known fixes
│   ├── test_signal_threshold_fix.py
│   └── test_nt_bar_structure_fix.py
└── validation/                     # Comprehensive validation suites
    ├── comprehensive_signal_validation.py
    └── definitive_signal_proof_test.py
```

**Documentation Organization:**
```
docs/
├── README.md                       # Main documentation entry
├── breakthroughs/                  # Major breakthrough documentation
│   ├── 2025-08-03-tirex-signal-optimization.md
│   └── 2025-08-03-nt-compliance-fix.md
├── implementation/                 # Implementation guides
│   ├── tirex/
│   │   ├── signal-optimization.md
│   │   └── threshold-tuning.md
│   └── backtesting/
│       ├── nt-compliance.md
│       └── look-ahead-bias-prevention.md
├── operations/                     # Operational guides
│   └── troubleshooting/
└── reference/                      # API and reference docs
```

### 2. Files to Move/Organize

**Root Level Cleanup:**
- Move all `test_*.py` files to appropriate test directories
- Move `debug_*.py` files to `debug/` directory
- Move validation files to `tests/validation/`
- Archive old documentation files

**Documentation Updates:**
- Document TiRex signal optimization breakthrough
- Document NT compliance fixes
- Update main README with current state
- Create usage guides

### 3. Regression Prevention

**Key Tests to Maintain:**
1. Signal generation threshold fix (0.01% vs 0.1%)
2. NT Bar structure compliance
3. Look-ahead bias prevention
4. Real DSM data integration
5. TiRex model loading and inference

## Implementation Steps

1. Create proper directory structure
2. Move files to correct locations
3. Update imports and references
4. Document breakthroughs
5. Create regression test suite
6. Validate everything still works