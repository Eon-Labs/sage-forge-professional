# TiRex Implementation Evolution Archive

This directory contains the evolutionary progression of TiRex signal generation implementations, preserved as reference for understanding the development path.

## Evolution Timeline

### Phase 1: Initial Implementation
- **File**: `visualize_authentic_tirex_signals.py`
- **Characteristics**: Sequential data feeding, single model state
- **Issues**: Bias accumulation leading to unreasonable signal patterns

### Phase 2: Extended Approach  
- **File**: `visualize_authentic_tirex_signals_extended.py`
- **Characteristics**: State clearing between windows, 512-bar context windows
- **Benefits**: Resolved bias issues, diverse signal generation
- **Issues**: Architectural violations (512 bars vs native 128), computational waste

### Phase 3: Adversarial Audit
- **Files**: `adversarial_audit_report.py`, `HOSTILE_AUDIT_FINAL_REPORT.md`
- **Purpose**: Critical analysis of extended approach
- **Findings**: Extended version works by accident, not design

### Phase 4: Comparative Analysis
- **Files**: `debug_comparison_test.py`, `deep_dive_analysis.py`
- **Purpose**: Understand why extended version succeeds despite violations
- **Insights**: State clearing and diverse windowing are key benefits

### Phase 5: Evolutionary Solution
- **File**: `optimal_tirex_solution.py` (archived - promotional naming)
- **Purpose**: Combine benefits without architectural violations
- **Approach**: Native 128-bar windows with strategic state management

## Key Learnings

1. **State Clearing**: Prevents bias accumulation from trending market data
2. **Diverse Windowing**: Captures different market regimes for balanced predictions  
3. **Architecture Compliance**: Respecting native model design prevents technical debt
4. **Accidental Success**: Extended approach worked due to deque auto-truncation

## Reference Usage

These files serve as:
- Historical context for implementation decisions
- Examples of common pitfalls in model integration
- Evidence of evolutionary development process
- Reference for future architectural decisions

**Note**: The current production implementation has evolved beyond these archived versions, incorporating lessons learned while maintaining architectural integrity.