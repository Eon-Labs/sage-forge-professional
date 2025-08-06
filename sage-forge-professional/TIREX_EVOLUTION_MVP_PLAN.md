# 🎯 TiRex Evolution MVP Plan: Minimum Viable ODEB Integration

**Date**: August 5, 2025  
**Objective**: Add ONLY what ODEB needs to `tirex_signal_generator.py` - nothing more  
**Philosophy**: Fail fast, validate hard, minimum viable features only  

---

## 📋 **ODEB MINIMUM REQUIREMENTS**

### **What ODEB Actually Needs (NOTHING MORE)**
1. **Real NT Positions** with:
   - `open_time` (entry timestamp)
   - `close_time` (exit timestamp)  
   - `size_usd` (position size in USD)
   - `pnl` (actual P&L from NT)
   - `direction` (1 for LONG, -1 for SHORT)

2. **Visualization** showing:
   - Entry/exit points on existing chart
   - P&L curve (single line, nothing fancy)
   - ODEB efficiency ratio (single metric display)

### **What We DON'T Need (CUT THESE)**
❌ Multi-horizon testing (1h, 4h, 24h) - not needed for ODEB  
❌ Benchmark strategies - ODEB is the benchmark  
❌ Interactive controls - nice to have, not needed  
❌ Statistical panels - ODEB provides the statistics  
❌ Multiple modes - just add backtest capability  
❌ Report generation - console output is enough  
❌ Drawdown panels - not required for ODEB  

---

## 🔴 **PHASE 0: CRITICAL INFRASTRUCTURE FIX**
**Goal**: Fix NT→ODEB conversion first (biggest failure risk)  
**Timeline**: 1 session  

### **Step 0.1: Investigate NT Position Structure**
```bash
# Run test backtest to see actual NT result structure
python cli/sage-backtest quick-test --debug > nt_structure_debug.txt
```

### **🚪 VALIDATION GATE 0.1: NT Structure Discovered**
**MUST PASS ALL**:
- [ ] Actual NT position object attributes documented
- [ ] Know exact path to positions in backtest results (e.g., `results.portfolio.positions_closed`)
- [ ] Sample NT position object captured and saved
- [ ] Mapping from NT fields to ODEB fields verified

**FAIL → STOP**: Cannot proceed without knowing NT structure

### **Step 0.2: Implement Real Position Extraction**
Fix `src/sage_forge/backtesting/tirex_backtest_engine.py` lines 243-260

### **🚪 VALIDATION GATE 0.2: Position Extraction Works**
**MUST PASS ALL**:
- [ ] Test script extracts at least 1 real NT position
- [ ] Extracted position has all 5 required ODEB fields
- [ ] No synthetic fallback triggered
- [ ] P&L matches NT reported P&L exactly

**FAIL → STOP**: ODEB needs real positions, not synthetic

---

## 🟡 **PHASE 1: MINIMAL BACKTEST INTEGRATION**
**Goal**: Add backtest capability without breaking current script  
**Timeline**: 1 session  

### **Step 1.1: Create Safe Copy**
```bash
cp tirex_signal_generator.py tirex_signal_generator_original.py
git add tirex_signal_generator_original.py
git commit -m "backup: preserve original before evolution"
```

### **Step 1.2: Add Minimal Argument Parser**
```python
# Add to tirex_signal_generator.py header comment:
# Evolution Plan: See TIREX_EVOLUTION_MVP_PLAN.md for incremental implementation

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true', 
                       help='Run backtesting after signal generation')
    args = parser.parse_args()
    
    # Original signal generation (unchanged)
    signals = generate_tirex_signals(...)
    
    if args.backtest:
        # NEW: Run backtest
        positions = run_minimal_backtest(signals)
```

### **🚪 VALIDATION GATE 1.1: Original Mode Unbroken**
**MUST PASS ALL**:
- [ ] `python tirex_signal_generator.py` works EXACTLY as before
- [ ] Finplot visualization unchanged
- [ ] Signal generation identical
- [ ] No new dependencies required for default mode

**FAIL → ROLLBACK**: Restore from tirex_signal_generator_original.py

### **Step 1.3: Minimal Backtest Function**
```python
def run_minimal_backtest(signals):
    """Run NT backtest and extract positions for ODEB."""
    from sage_forge.backtesting import TiRexBacktestEngine
    
    engine = TiRexBacktestEngine()
    engine.setup_backtest()
    results = engine.run_backtest()
    
    # Use FIXED extraction from Phase 0
    positions = engine.extract_positions_from_backtest(results)
    return positions
```

### **🚪 VALIDATION GATE 1.2: Backtest Produces Positions**
**MUST PASS ALL**:
- [ ] `python tirex_signal_generator.py --backtest` runs without error
- [ ] Returns list of Position objects (not empty)
- [ ] Each position has all 5 ODEB fields populated
- [ ] Console shows "Extracted X positions" (X > 0)

**FAIL → DEBUG**: Check backtest configuration and strategy

---

## 🟢 **PHASE 2: ODEB ANALYSIS INTEGRATION**
**Goal**: Run ODEB on real positions  
**Timeline**: 0.5 session  

### **Step 2.1: Add ODEB Analysis**
```python
def run_odeb_analysis(positions, market_data):
    """Run ODEB analysis on backtest positions."""
    from sage_forge.reporting.performance import OmniscientDirectionalEfficiencyBenchmark
    
    odeb = OmniscientDirectionalEfficiencyBenchmark(positions)
    result = odeb.calculate_odeb_ratio(positions, market_data)
    
    # Simple console output (no fancy panels)
    print(f"ODEB Efficiency: {result.tirex_efficiency_ratio:.2%}")
    print(f"Directional Capture: {result.directional_capture_pct:.1f}%")
    return result
```

### **🚪 VALIDATION GATE 2.1: ODEB Calculates Metrics**
**MUST PASS ALL**:
- [ ] ODEB efficiency ratio between 0 and 1
- [ ] Directional capture percentage calculated
- [ ] No errors from ODEB framework
- [ ] Results object contains all expected fields

**FAIL → CHECK**: Position data format and ODEB inputs

---

## 🔵 **PHASE 3: MINIMAL VISUALIZATION ENHANCEMENT**
**Goal**: Add ONLY position markers and P&L line to existing finplot  
**Timeline**: 1 session  

### **Step 3.1: Add Position Markers**
```python
def add_backtest_overlay(ax, positions, market_data):
    """Add minimal backtest visualization to existing plot."""
    import finplot as fplt
    
    for position in positions:
        # Entry marker (green/red circle)
        entry_color = '#00ff00' if position.direction > 0 else '#ff0000'
        fplt.plot([position.open_time], [position.entry_price], 
                 ax=ax, color=entry_color, style='o', width=3)
        
        # Exit marker (X)
        fplt.plot([position.close_time], [position.exit_price],
                 ax=ax, color='#ffffff', style='x', width=2)
```

### **🚪 VALIDATION GATE 3.1: Positions Visible on Chart**
**MUST PASS ALL**:
- [ ] Entry points show as circles (green=long, red=short)
- [ ] Exit points show as X marks
- [ ] Original signals (triangles) still visible
- [ ] No overlap/confusion between signals and positions

**FAIL → ADJUST**: Position marker styling or placement

### **Step 3.2: Add P&L Curve Panel**
```python
def add_pnl_panel(market_data, positions):
    """Add simple P&L curve as new panel."""
    # Calculate cumulative P&L
    cumulative_pnl = calculate_cumulative_pnl(positions)
    
    # Add as third row panel
    ax3 = fplt.create_plot('P&L', rows=3)
    fplt.plot(timestamps, cumulative_pnl, ax=ax3, 
             color='#58a6ff', legend='TiRex P&L')
```

### **🚪 VALIDATION GATE 3.2: P&L Curve Displays**
**MUST PASS ALL**:
- [ ] Third panel shows P&L curve
- [ ] P&L starts at 0 and moves with trades
- [ ] Final P&L matches sum of position P&Ls
- [ ] Original OHLC and volume panels unchanged

**FAIL → CHECK**: Panel creation and data alignment

---

## ✅ **PHASE 4: FINAL INTEGRATION**
**Goal**: Clean integration with ODEB metric display  
**Timeline**: 0.5 session  

### **Step 4.1: Add ODEB Metric Display**
```python
# Add single line to console output
print(f"ODEB Efficiency: {odeb_result.tirex_efficiency_ratio:.2%} "
      f"(Captured {odeb_result.directional_capture_pct:.1f}% of perfect trades)")
```

### **Step 4.2: Update Header Documentation**
```python
"""
TiRex Signal Generator - Evolutionary Implementation
Now with ODEB backtesting integration (MVP)

Default: Signal generation with finplot (unchanged)
With --backtest: Adds real NT positions and ODEB analysis

See TIREX_EVOLUTION_MVP_PLAN.md for implementation details
"""
```

### **🚪 VALIDATION GATE 4.1: Complete Integration**
**MUST PASS ALL**:
- [ ] Default mode works exactly as original
- [ ] --backtest mode shows positions and P&L
- [ ] ODEB metrics displayed in console
- [ ] No performance degradation in visualization
- [ ] Script remains under 500 lines total

**SUCCESS → COMPLETE**: MVP achieved with minimum viable features

---

## 🎯 **SUCCESS CRITERIA SUMMARY**

### **Minimum Viable Product Delivered**
✅ Real NT positions extracted (not synthetic)  
✅ ODEB analysis runs on real data  
✅ Position entry/exit visible on chart  
✅ P&L curve shows actual performance  
✅ ODEB efficiency metric displayed  
✅ Original functionality preserved  

### **What We DIDN'T Add (Correctly Avoided)**
❌ Multiple modes (research, etc.)  
❌ Multi-horizon testing  
❌ Benchmark strategies  
❌ Interactive controls  
❌ Statistical panels  
❌ Report generation  
❌ Fancy overlays  

---

## 📍 **CURRENT STATUS**

**Phase**: 0 - Critical Infrastructure Fix  
**Next Gate**: 0.1 - NT Structure Discovery  
**Blocked By**: Need to run test backtest to see NT structure  

---

## 🔗 **REFERENCE LOCATIONS**

**This Plan**: `/home/tca/eon/nt/sage-forge-professional/TIREX_EVOLUTION_MVP_PLAN.md`  
**Script**: `/home/tca/eon/nt/sage-forge-professional/tirex_signal_generator.py`  
**Backup**: `/home/tca/eon/nt/sage-forge-professional/tirex_signal_generator_original.py`  
**NT Fix**: `/home/tca/eon/nt/sage-forge-professional/src/sage_forge/backtesting/tirex_backtest_engine.py`  

---

**REMEMBER**: Each gate must pass 100% before proceeding. Fail fast, fix issues, validate hard.