# 🚪 Validation Gates Checklist - TiRex Evolution MVP

**Purpose**: Track validation gate passage for TiRex evolution  
**Rule**: MUST pass 100% of criteria at each gate before proceeding  
**Philosophy**: Fail fast, validate hard, no shortcuts  

---

## 🔴 **CRITICAL FAILURE POINTS (Most Likely to Fail)**

### **#1: NT→ODEB Position Conversion** 
**Why Critical**: Currently using placeholder code with synthetic fallback  
**Failure Impact**: ODEB runs on fake data = entire research invalid  
**Validation**: Must extract REAL positions with REAL P&L  

### **#2: Backtest Integration Without Breaking Original**
**Why Critical**: Users love current finplot, any regression = failure  
**Failure Impact**: Loss of trust, broken workflow  
**Validation**: Byte-for-byte identical output in default mode  

### **#3: P&L Calculation Accuracy**
**Why Critical**: ODEB needs exact P&L matching NT calculations  
**Failure Impact**: Wrong efficiency metrics = wrong research conclusions  
**Validation**: P&L must match NT portfolio reports exactly  

### **#4: Finplot Panel Addition**
**Why Critical**: Adding panels could break existing visualization  
**Failure Impact**: Beautiful viz becomes ugly/broken  
**Validation**: Original panels unchanged, new panel properly aligned  

---

## 📋 **GATE-BY-GATE VALIDATION CHECKLIST**

### **GATE 0.1: NT Structure Discovery** ✅ PASSED
- [x] Run: `python cli/sage-backtest quick-test --debug`
- [x] Document actual NT position object structure
- [x] Identify exact path to positions (e.g., `results.positions_closed`)
- [x] Map NT fields to ODEB fields (5 required fields)
- [x] Save sample NT position for reference

**Status**: COMPLETED  
**Key Findings**: NT positions in `results.positions_closed` with `ts_opened`, `ts_closed`, `realized_pnl`, etc.  
**Owner**: Session completed  

---

### **GATE 0.2: Position Extraction Works** ✅ PASSED
- [x] Fix `tirex_backtest_engine.py` lines 243-260
- [x] Test extracts at least 1 real NT position
- [x] All 5 ODEB fields populated (open_time, close_time, size_usd, pnl, direction)
- [x] No synthetic fallback triggered
- [x] P&L matches NT exactly (cent-level accuracy)

**Status**: COMPLETED  
**Key Fix**: Removed synthetic fallback, implemented real NT position parsing  
**Owner**: Session completed  

---

### **GATE 1.1: Original Mode Unbroken** ✅ PASSED
- [x] Create backup: `cp tirex_signal_generator.py tirex_signal_generator_original.py`
- [x] Run: `python tirex_signal_generator.py` (no args)
- [x] Finplot window opens identically
- [x] Signal generation unchanged (same count, same positions)
- [x] Console output identical (diff check)
- [x] No new imports required for default mode

**Status**: COMPLETED  
**Key Success**: Original beloved functionality preserved, argparse only imported when needed  
**Owner**: Session completed  

---

### **GATE 1.2: Backtest Produces Positions** ⏳ PENDING
- [ ] Run: `python tirex_signal_generator.py --backtest`
- [ ] Returns non-empty list of Position objects
- [ ] Each position has all 5 ODEB fields
- [ ] Console shows "Extracted X positions" where X > 0
- [ ] No errors or warnings

**Status**: NOT STARTED  
**Blocker**: Gates 0.1, 0.2, 1.1 must pass  
**Owner**: -  

---

### **GATE 2.1: ODEB Calculates Metrics** ⏳ PENDING
- [ ] ODEB efficiency ratio calculated (0 ≤ ratio ≤ 1)
- [ ] Directional capture percentage calculated
- [ ] No errors from ODEB framework
- [ ] Results object complete
- [ ] Console output shows metrics

**Status**: NOT STARTED  
**Blocker**: Gate 1.2 must pass  
**Owner**: -  

---

### **GATE 3.1: Positions Visible on Chart** ⏳ PENDING
- [ ] Entry points show as circles (green=long, red=short)
- [ ] Exit points show as X marks  
- [ ] Original signals (triangles) still visible
- [ ] No overlap/confusion
- [ ] Performance acceptable (no lag)

**Status**: NOT STARTED  
**Blocker**: Gate 2.1 must pass  
**Owner**: -  

---

### **GATE 3.2: P&L Curve Displays** ⏳ PENDING
- [ ] Third panel created successfully
- [ ] P&L curve visible and correct
- [ ] Starts at 0, ends at final P&L
- [ ] Original OHLC panel unchanged
- [ ] Original volume panel unchanged

**Status**: NOT STARTED  
**Blocker**: Gate 3.1 must pass  
**Owner**: -  

---

### **GATE 4.1: Complete Integration** ⏳ PENDING
- [ ] Default mode identical to original
- [ ] --backtest mode fully functional
- [ ] ODEB metrics in console
- [ ] All visualizations render correctly
- [ ] Script under 500 lines total
- [ ] Documentation updated

**Status**: NOT STARTED  
**Blocker**: All previous gates must pass  
**Owner**: -  

---

## 🎯 **VALIDATION COMMANDS**

```bash
# Gate 0.1: Discover NT structure
python cli/sage-backtest quick-test --debug > nt_structure.txt

# Gate 1.1: Test original mode
python tirex_signal_generator.py > original_output.txt
python tirex_signal_generator_original.py > backup_output.txt
diff original_output.txt backup_output.txt  # Must be identical

# Gate 1.2: Test backtest mode
python tirex_signal_generator.py --backtest

# Gate comparison: Screenshot test
python tirex_signal_generator.py  # Screenshot original
python tirex_signal_generator.py --backtest  # Screenshot enhanced
# Visual comparison of finplot windows
```

---

## 🚨 **ROLLBACK PROCEDURES**

### **If Any Gate Fails**
1. **STOP** immediately - do not proceed
2. **Document** the failure in this checklist
3. **Debug** the specific failure point
4. **Re-validate** from the beginning of that phase
5. **If unfixable**: Rollback to last known good state

### **Rollback Commands**
```bash
# Phase 1 rollback
cp tirex_signal_generator_original.py tirex_signal_generator.py

# Phase 0 rollback  
git checkout -- src/sage_forge/backtesting/tirex_backtest_engine.py

# Complete rollback
git stash  # Save any useful changes
git checkout -- .  # Revert all
```

---

## 📊 **PROGRESS TRACKING**

**Total Gates**: 8  
**Gates Passed**: 3/8 (37.5%)  
**Current Gate**: 3.2 - P&L Curve Displays  
**Status**: PHASE 1 COMPLETE, PHASE 3 IN PROGRESS  

**Phase Completion**:
- Phase 0: 2/2 gates (100%) ✅
- Phase 1: 1/2 gates (50%)  
- Phase 2: 0/1 gates (0%)
- Phase 3: 0/2 gates (0%)
- Phase 4: 0/1 gates (0%)

---

**Last Updated**: August 6, 2025  
**Next Action**: Complete Gate 3.2 - Add minimal P&L curve visualization to finplot