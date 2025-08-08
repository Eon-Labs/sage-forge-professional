# TiRex Merit Isolation Research Implementation Plan

**Date**: August 5, 2025  
**Purpose**: Design and implement dedicated TiRex merit isolation research framework  
**Context**: Critical infrastructure gaps discovered blocking merit isolation research  

---

## 🎯 **RESEARCH OBJECTIVE**

Answer the fundamental question: **Does TiRex alone justify its computational cost and complexity?**

### **Core Research Questions** (from pending_research_topics.md)
1. **Signal Quality**: Can TiRex forecasts alone generate profitable trades vs buy-and-hold?
2. **Uncertainty Edge**: Do TiRex confidence estimates improve Sharpe ratios and drawdown control?
3. **Optimal Horizon**: What forecast length (1h, 4h, 24h) works best for crypto trading?
4. **Market Regime Performance**: Where does TiRex excel vs. fail across market conditions?
5. **Transaction Cost Break-Even**: At what accuracy does TiRex overcome trading fees and slippage?

---

## 🏗️ **EVOLUTION STRATEGY: Create Dedicated Research Script**

### **Why NOT Evolve Existing `tirex_signal_generator.py`**

**Current Purpose**: Signal generation and visualization only
- Generates TiRex signals from market data
- Shows signal statistics and finplot visualization  
- Does NOT run actual backtesting or produce real trades
- Optimized for signal preview and analysis

**Evolution Problems**:
- Would completely change script purpose (signal generation → full research framework)
- Mixing concerns (generation + backtesting + ODEB + research)
- Breaking existing users who expect signal generation
- Becoming massive monolithic script

### **RECOMMENDED APPROACH: Create `tirex_merit_isolation_research.py`**

**Benefits**:
✅ **Clear separation of concerns** - Signal generation vs research are different purposes  
✅ **Purpose-built** - Optimized specifically for merit isolation research needs  
✅ **Evolutionary development** - Builds on existing components while advancing the mission  
✅ **Focused research** - Addresses specific research questions systematically  
✅ **Infrastructure integration** - Combines TiRex + NT backtesting + ODEB analysis  

---

## 📋 **CRITICAL INFRASTRUCTURE GAPS TO RESOLVE**

### **Gap #1: NT→ODEB Position Conversion (CRITICAL BLOCKER)**
**Location**: `src/sage_forge/backtesting/tirex_backtest_engine.py` lines 243-260  
**Problem**: Placeholder code that falls back to synthetic positions  
**Impact**: ODEB analysis runs on fake data, making research invalid  
**Solution Required**:
1. Run test backtest to examine actual NT result structure
2. Implement real position extraction from NT portfolio/orders/fills
3. Convert NT positions to ODEB custom Position dataclass
4. Validate P&L calculations match NT calculations

### **Gap #2: Multi-Horizon Testing Infrastructure**
**Problem**: No framework for testing 1h, 4h, 24h forecast horizons  
**Solution Required**:
1. Multi-timeframe data loading capability
2. Strategy variants for different horizons
3. Horizon-specific performance comparison
4. Optimal horizon identification methodology

### **Gap #3: Benchmark Comparison Framework**
**Problem**: No buy-and-hold or baseline strategy comparison  
**Solution Required**:
1. Buy-and-hold strategy implementation
2. Simple moving average strategies for baselines
3. Performance comparison metrics (Sharpe, drawdown, return)
4. Statistical significance testing

### **Gap #4: Transaction Cost Analysis**
**Problem**: No real trading cost impact assessment  
**Solution Required**:
1. Realistic spread and fee modeling
2. Transaction cost impact on different time horizons
3. Break-even accuracy calculation
4. Cost-adjusted performance metrics

---

## 🔧 **IMPLEMENTATION ARCHITECTURE**

### **Core Components Integration**

```python
# tirex_merit_isolation_research.py architecture
class TiRexMeritIsolationResearch:
    def __init__(self):
        self.tirex_model = None           # From tirex_signal_generator.py
        self.backtest_engine = None       # Fixed NT→ODEB conversion
        self.benchmark_strategies = {}    # Buy-and-hold, SMA baselines
        self.odeb_analyzer = None         # From performance.py
        self.multi_horizon_config = {}    # 1h, 4h, 24h configurations
        
    def run_merit_isolation_study(self):
        """Execute complete merit isolation research study."""
        # 1. Multi-horizon signal generation
        # 2. NT backtesting with real position extraction
        # 3. ODEB analysis on real positions
        # 4. Benchmark comparison
        # 5. Transaction cost analysis
        # 6. Statistical significance testing
        # 7. Comprehensive research report
```

### **Research Workflow**

1. **Data Preparation**
   - Load multi-timeframe market data (1h, 4h, 24h bars)
   - Validate data quality and temporal consistency
   - Prepare benchmark comparison datasets

2. **Signal Generation**
   - Use existing `tirex_signal_generator.py` logic
   - Generate signals for each time horizon
   - Record confidence estimates and volatility forecasts

3. **Backtesting Execution** 
   - Fix NT→ODEB position conversion (Gap #1)
   - Run NT backtests with TiRex strategy
   - Extract real NT positions for ODEB analysis

4. **Performance Analysis**
   - ODEB directional efficiency benchmarking
   - Compare vs buy-and-hold and baseline strategies
   - Calculate transaction cost impact
   - Statistical significance testing

5. **Research Documentation**
   - Answer all 5 core research questions
   - Generate comprehensive research report
   - Create performance visualizations
   - Document findings and recommendations

---

## 📊 **SUCCESS CRITERIA**

### **Infrastructure Validation**
- [ ] NT→ODEB position conversion produces real trade data (not synthetic)
- [ ] Multi-horizon testing operational for 1h, 4h, 24h windows
- [ ] Benchmark strategies implemented and validated
- [ ] Transaction cost modeling produces realistic results

### **Research Questions Answered**
- [ ] **Signal Quality**: TiRex vs buy-and-hold performance quantified
- [ ] **Uncertainty Edge**: Confidence-based position sizing impact measured  
- [ ] **Optimal Horizon**: Best forecast length identified empirically
- [ ] **Market Regime Performance**: TiRex performance across market conditions mapped
- [ ] **Transaction Cost Break-Even**: Minimum accuracy threshold calculated

### **Scientific Validation**  
- [ ] TiRex standalone merit validated or refuted with statistical significance
- [ ] Clear evidence for/against TiRex value before ensemble complexity
- [ ] Optimal configuration identified for production deployment
- [ ] Research findings documented for future development decisions

---

## 🎯 **NEXT SESSION PRIORITIES**

### **PRIORITY 1: Fix NT→ODEB Conversion**
1. Run simple TiRex backtest using existing CLI: `sage-backtest quick-test`
2. Examine actual NT backtest result structure  
3. Implement real position extraction replacing placeholder code
4. Validate extracted positions match expected format

### **PRIORITY 2: Create Research Script Foundation**
1. Create `tirex_merit_isolation_research.py` 
2. Integrate existing TiRex signal generation logic
3. Add backtesting integration with fixed position conversion
4. Implement basic ODEB analysis on real positions

### **PRIORITY 3: Multi-Horizon Testing Framework**
1. Design multi-timeframe data handling
2. Create strategy variants for different horizons  
3. Implement horizon-specific performance comparison
4. Add benchmark strategy implementations

---

## 🔗 **COMPONENT RELATIONSHIPS**

### **Preserve Existing Functionality**
- `tirex_signal_generator.py` → **Keep as-is** for signal generation/visualization
- `src/sage_forge/backtesting/tirex_backtest_engine.py` → **Fix NT conversion**
- `src/sage_forge/reporting/performance.py` → **Use ODEB framework**
- `src/sage_forge/strategies/tirex_sage_strategy.py` → **Use for backtesting**

### **New Research Integration**
- `tirex_merit_isolation_research.py` → **NEW: Dedicated research framework**
- Uses signal generation from existing script
- Uses fixed backtesting engine  
- Uses ODEB analysis framework
- Adds benchmark comparison and multi-horizon testing

---

**Implementation Status**: Ready to begin with PRIORITY 1 - Fix NT→ODEB conversion  
**Expected Timeline**: 2-3 sessions for complete merit isolation research framework  
**Research Impact**: Provides scientific foundation for all future TiRex development decisions