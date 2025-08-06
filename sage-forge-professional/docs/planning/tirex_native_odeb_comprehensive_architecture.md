# TiRex-Native ODEB Comprehensive Architecture Plan

**Created**: 2025-08-06  
**Status**: Phase 3B Implementation Ready  
**Context**: Post-Phase 3A successful TiRex→NT integration with comprehensive ODEB enhancement  

---

## 🎯 Executive Summary

This document establishes the comprehensive architecture for **TiRex-Native ODEB Integration** based on paradigm-shifting discoveries about TiRex's multi-horizon forecasting capabilities and the resolution of critical walk-forward analysis gaps. All concepts follow **Canonicalized Naming Convention** for precise conceptual boundaries.

### **Strategic Impact**
- **Resolves**: Confidence Leak Problem Resolution (CLPR) in current ODEB framework
- **Enables**: Multi-Horizon Directional Capture Analysis (MHDCA) with trajectory fidelity
- **Preserves**: Temporal Causal Ordering Preservation (TCOP) for bias-free walk-forward validation
- **Implements**: Internal Rival Paradigm Benchmarking (IRPB) with authentic TiRex oracle modeling

---

## 🏗️ Layered Architecture with Canonicalized Concepts

### **Layer 1: TiRex Native Model (Foundational Intelligence)**

#### **Multi-Horizon Trajectory Forecasting Capability (MHTFC)**
- **Definition**: TiRex's native ability to predict 1-768+ bars with full quantile distributions
- **Current State**: Underutilized (using prediction_length=1 only)
- **Enhancement Potential**: Multi-step oracle trajectory modeling
- **Key Files**: `repos/tirex/src/tirex/models/tirex.py:137-187`
- **Acronym**: MHTFC

#### **Temporal Causal Ordering Preservation (TCOP)**
- **Definition**: All regime detection and predictions use strictly historical context (T-1) without look-ahead bias  
- **Validation**: Regime detection uses `price_buffer` (historical only), predictions use context up to T-1
- **Critical Property**: Preserves walk-forward analysis validity
- **Key Files**: `src/sage_forge/models/tirex_model.py:199-229`
- **Acronym**: TCOP

#### **Context Boundary Prediction Phases (CBPP)**
- **Definition**: Three distinct TiRex prediction phases with different reliability characteristics
- **Phases**: 
  - WARM_UP_PERIOD (bars 0-127): No predictions
  - CONTEXT_BOUNDARY (bar 128): First prediction with minimum context
  - STABLE_WINDOW (bars 129+): Rolling window predictions
- **ODEB Integration**: Only STABLE_WINDOW predictions should participate in evaluation
- **Key Files**: `src/sage_forge/models/tirex_model.py:375`, `src/sage_forge/models/prediction_boundary_validator.py`
- **Acronym**: CBPP

#### **Regime-Aware Continuity Pattern (RACP)**
- **Definition**: TiRex maintains 128-bar context continuity across regime changes without context reset
- **Behavior**: Regime detection serves as confidence threshold adjustment, not context boundary
- **Integration**: Walk-forward windows must respect context continuity requirements
- **Key Files**: `src/sage_forge/strategies/tirex_sage_strategy.py:416-452`
- **Acronym**: RACP

### **Layer 2: TiRex Translated Strategy (Operational Intelligence)**

#### **Confidence-Weighted Position Sizing Protocol (CWPSP)**
- **Definition**: Position sizing algorithm using TiRex confidence scores with regime multipliers
- **Current Implementation**: `base_size * regime_multiplier * confidence_factor`
- **Critical Issue**: Confidence information lost in ODEB position dataclass
- **Key Files**: `src/sage_forge/strategies/tirex_sage_strategy.py:469-491`
- **Acronym**: CWPSP

#### **Adaptive Confidence Threshold Management (ACTM)**
- **Definition**: Dynamic confidence thresholds (8%-20%) based on detected market regime
- **Regime Mapping**: High volatility = lower thresholds, Low volatility = higher thresholds
- **Oracle Integration**: Oracle should inherit same threshold sensitivity patterns
- **Key Files**: `src/sage_forge/strategies/tirex_sage_strategy.py:416-452`
- **Acronym**: ACTM

#### **Creative Bridge Configuration Resolution (CBCR)**
- **Definition**: Multi-path configuration handling to ensure proper TiRex→NT instrument binding
- **Achievement**: Resolved Phase 3A critical integration barriers
- **Pattern**: Fall-back configuration discovery with direct instrument_id subscription
- **Key Files**: `src/sage_forge/strategies/tirex_sage_strategy.py:298-327`
- **Acronym**: CBCR

### **Layer 3: ODEB-ized Oracle (Omniscient Benchmarking Intelligence)**

#### **Trajectory Fidelity Oracle Modeling (TFOM)**
- **Definition**: Oracle maintains TiRex's multi-step forecast confidence profile while optimizing execution timing
- **Revolutionary Approach**: N-bar perfect trajectory alignment instead of single-bar perfect timing
- **Implementation**: Oracle uses same quantile uncertainty distributions with perfect entry/exit
- **Enhancement**: Multi-horizon trajectory predictions for realistic oracle benchmarking
- **Acronym**: TFOM

#### **Internal Rival Paradigm Benchmarking (IRPB)**
- **Definition**: ODEB benchmarks TiRex Strategy against "Omniscient TiRex Oracle" rather than external benchmarks
- **Paradigm Shift**: Strategy vs Perfect-Information-Same-Model instead of Strategy vs Market-Index
- **Core Principle**: Oracle maintains TiRex's prediction regime, confidence patterns, position sizing logic
- **Measurement**: Capture percentage of oracle-level performance achievable by translated strategy
- **Key Files**: `sage-forge-professional/docs/implementation/tirex/odeb-benchmark-specification.md`
- **Acronym**: IRPB

#### **Confidence Inheritance Architecture (CIA)**
- **Definition**: Oracle inherits TiRex confidence distributions and position sizing patterns with perfect timing execution
- **Design Options**:
  - Option A: Same confidence distribution + perfect timing
  - Option B: Perfect confidence (1.0) + TiRex-like position sizing
- **Integration**: Resolves Confidence Leak Problem through end-to-end confidence preservation
- **Acronym**: CIA

### **Layer 4: ODEB Method (Capture Efficiency Quantification)**

#### **Multi-Horizon Directional Capture Analysis (MHDCA)**
- **Definition**: ODEB evaluation across multiple TiRex prediction horizons (1, 5, 15, 30 bars)
- **Methodology**: Different holding periods get different oracle benchmarks based on prediction horizon
- **Aggregation**: Weighted final score based on horizon representation in actual strategy
- **Enhancement**: Captures TiRex's true multi-step forecasting capability
- **Acronym**: MHDCA

#### **Confidence Leak Problem Resolution (CLPR)**
- **Definition**: Systematic preservation of TiRex confidence information through entire ODEB evaluation pipeline
- **Current Issue**: Position dataclass lacks confidence field despite TiRex using confidence for position sizing
- **Solution**: Enhanced Position dataclass with confidence, regime, and prediction phase fields
- **Validation**: End-to-end confidence flow verification
- **Key Files**: `src/sage_forge/reporting/performance.py:381-392` (requires enhancement)
- **Acronym**: CLPR

#### **Regime-Aware Evaluation Weighting (RAEW)**
- **Definition**: ODEB analysis weighted by regime stability and transition characteristics during evaluation period
- **Methodology**: More stable regimes receive higher confidence weighting in oracle calculation
- **Integration**: Accounts for TiRex's regime-based confidence threshold variations
- **Implementation**: Position-level or window-level regime weighting options
- **Acronym**: RAEW

### **Layer 5: Walk-Forward Analysis (Temporal Validation)**

#### **Walk-Forward Simulation Gap Elimination (WFSGE)**
- **Definition**: Replacement of mock position generation with authentic TiRex strategy backtesting in walk-forward windows
- **Current Issue**: Framework exists but uses synthetic positions instead of real strategy results
- **Solution**: Execute real TiRex strategy on window data with confidence preservation
- **Validation**: Authentic Strategy vs Oracle comparison with preserved confidence patterns
- **Key Files**: `src/sage_forge/optimization/tirex_parameter_optimizer.py:527-548` (requires replacement)
- **Acronym**: WFSGE

#### **Authentic Strategy Backtesting Integration (ASBI)**
- **Definition**: Real TiRex strategy execution within walk-forward windows with full confidence preservation
- **Architecture**: Extract positions WITH confidence from actual backtesting results
- **Oracle Creation**: Build oracle using authentic position confidence patterns with perfect timing
- **Integration**: Maintains Temporal Causal Ordering Preservation throughout analysis
- **Acronym**: ASBI

#### **Look-Ahead Bias Prevention Validation (LABPV)**
- **Definition**: Comprehensive validation ensuring no future information leakage in regime-adaptive ODEB analysis
- **Validation Proof**: All regime detection uses price_buffer (historical prices only)
- **Timeline Verification**: T-1 context → regime detection + prediction → T execution
- **Critical Property**: Walk-forward analysis remains valid with regime adaptation
- **Achievement**: Confirmed bias-free integration of adaptive regime detection
- **Acronym**: LABPV

### **System Integration Layer**

#### **Phase-Based Implementation Strategy (PBIS)**
- **Definition**: Structured implementation approach separating immediate needs from future enhancements
- **Phase 1 (Immediate)**: Confidence Leak Problem Resolution + Regime-Aware Evaluation Weighting
- **Phase 2 (Enhancement)**: Multi-Horizon Directional Capture Analysis + Trajectory Fidelity Oracle Modeling  
- **Phase 3 (Advanced)**: Full Walk-Forward Simulation Gap Elimination with multi-horizon trajectory analysis
- **Rationale**: Maintains proven patterns while systematically adding missing capabilities
- **Acronym**: PBIS

#### **Comprehensive Integration Testing Framework (CITF)**
- **Definition**: Multi-layer validation ensuring all canonicalized concepts work together without interference
- **Test Categories**: 
  - Temporal Causality Tests (TCOP validation)
  - Confidence Flow Tests (CLPR validation)  
  - Regime Continuity Tests (RACP validation)
  - Oracle Fidelity Tests (TFOM + IRPB validation)
- **Success Criteria**: All integration tests pass with no look-ahead bias introduction
- **Acronym**: CITF

---

## 🚀 Phase 1 Implementation Plan (Immediate - Phase 3B)

### **Critical Path: Confidence Leak Problem Resolution (CLPR)**

#### **Step 1: Enhanced Position Dataclass**
```python
@dataclass
class TiRexNativePosition:
    # Existing fields
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    size_usd: float
    pnl: float
    direction: int
    
    # NEW: TiRex-specific fields for CLPR
    confidence: float           # From TiRex prediction
    market_regime: str          # Regime when position opened  
    regime_stability: float     # Regime persistence during position
    prediction_phase: str       # STABLE_WINDOW vs CONTEXT_BOUNDARY
    
    # ENHANCEMENT: Multi-horizon support (Phase 2)
    trajectory_confidence: Optional[List[float]] = None  # For multi-step positions
```

#### **Step 2: Confidence Inheritance Architecture (CIA)**  
```python
class ConfidenceInheritanceOracle:
    def create_oracle_with_confidence_fidelity(self, tirex_positions):
        oracle_positions = []
        for pos in tirex_positions:
            oracle_pos = TiRexNativePosition(
                # Perfect timing but same confidence distribution
                confidence=pos.confidence,          # Inherit TiRex confidence
                market_regime=pos.market_regime,    # Same regime context
                regime_stability=pos.regime_stability,
                prediction_phase=pos.prediction_phase,
                
                # Oracle advantages
                entry_timing="perfect_open",        # Perfect entry timing
                exit_timing="perfect_close",        # Perfect exit timing
                size_usd=pos.size_usd,             # Same position sizing
                pnl=self.calculate_oracle_pnl(pos) # Perfect execution PnL
            )
            oracle_positions.append(oracle_pos)
        return oracle_positions
```

#### **Step 3: Regime-Aware Evaluation Weighting (RAEW)**
```python
class RegimeAwareODEB:
    def calculate_regime_weighted_capture(self, positions, market_data):
        regime_results = {}
        
        for regime in self.get_unique_regimes(positions):
            regime_positions = [p for p in positions if p.market_regime == regime]
            
            # Weight oracle calculation by regime stability
            stability_weight = np.mean([p.regime_stability for p in regime_positions])
            
            # Create regime-specific oracle
            regime_oracle = self.create_regime_oracle(regime_positions, stability_weight)
            
            # Calculate regime-specific ODEB
            regime_odeb = self.calculate_confidence_weighted_odeb(
                regime_positions, regime_oracle
            )
            
            regime_results[regime] = {
                'odeb_score': regime_odeb,
                'stability_weight': stability_weight,
                'position_count': len(regime_positions)
            }
        
        return self.aggregate_regime_weighted_results(regime_results)
```

### **Integration Testing: Comprehensive Integration Testing Framework (CITF)**

#### **Test 1: Temporal Causality Validation (TCOP)**
```python
def test_temporal_causal_ordering():
    """Validate no look-ahead bias in regime-adaptive ODEB."""
    
    # Verify regime detection uses only historical context
    assert regime_detector.uses_only_historical_context()
    
    # Verify prediction timeline: T-1 context → T prediction → T execution  
    timeline = validate_prediction_timeline(strategy_positions)
    assert timeline.no_future_information_used()
    
    # Verify walk-forward windows preserve temporal ordering
    for window in walk_forward_windows:
        assert window.respects_temporal_causality()
```

#### **Test 2: Confidence Flow Validation (CLPR)**
```python
def test_confidence_leak_resolution():
    """Validate confidence preservation through entire pipeline."""
    
    # Verify TiRex confidence flows to position
    tirex_prediction = tirex_model.predict(bar)
    strategy_position = strategy.create_position(tirex_prediction)
    assert strategy_position.confidence == tirex_prediction.confidence
    
    # Verify position confidence flows to ODEB
    odeb_result = odeb_analyzer.analyze(strategy_position, oracle_position)
    assert odeb_result.uses_confidence_weighting()
    
    # Verify oracle inherits confidence distribution
    oracle_position = oracle_creator.create_oracle(strategy_position)
    assert oracle_position.confidence == strategy_position.confidence
```

---

## 🔮 Phase 2 Enhancement Plan (Multi-Horizon Trajectory)

### **Multi-Horizon Directional Capture Analysis (MHDCA)**

#### **Trajectory Fidelity Oracle Modeling (TFOM)**
```python
class MultiHorizonTrajectoryOracle:
    def __init__(self, prediction_horizons=[1, 5, 15, 30]):
        self.prediction_horizons = prediction_horizons
        
    def create_trajectory_oracle(self, tirex_strategy_positions, market_data):
        oracle_results = {}
        
        for horizon in self.prediction_horizons:
            # Get TiRex trajectory predictions for this horizon
            trajectory_predictions = self.get_tirex_trajectories(
                context=market_data, 
                prediction_length=horizon
            )
            
            # Create oracle with same confidence profile but perfect trajectory alignment
            oracle_trajectories = self.align_trajectories_to_perfect_timing(
                trajectory_predictions, market_data
            )
            
            # Calculate horizon-specific ODEB with trajectory confidence weighting
            oracle_results[horizon] = self.calculate_trajectory_confidence_odeb(
                strategy_positions, oracle_trajectories
            )
        
        return self.aggregate_multi_horizon_results(oracle_results)
```

#### **Enhanced Position Architecture for Multi-Horizon**
```python
@dataclass  
class MultiHorizonTiRexPosition(TiRexNativePosition):
    # Multi-horizon trajectory support
    trajectory_confidence: List[float]      # Confidence at each prediction step
    prediction_horizon: int                 # Number of bars predicted ahead
    trajectory_regime_stability: List[float] # Regime stability over trajectory
    
    # Oracle trajectory comparison
    oracle_trajectory: Optional[List[float]] = None
    trajectory_capture_efficiency: Optional[float] = None
```

---

## 🏁 Phase 3 Advanced Plan (Walk-Forward Simulation Gap Elimination)

### **Authentic Strategy Backtesting Integration (ASBI)**

#### **Walk-Forward Simulation Gap Elimination (WFSGE)**
```python
class AuthenticWalkForwardEngine:
    def execute_tirex_window_with_trajectory_analysis(self, window, parameters):
        # Execute REAL TiRex strategy with multi-horizon predictions
        strategy_config = self.build_tirex_config(parameters)
        strategy_config.prediction_horizons = [1, 5, 15, 30]  # Multi-horizon
        
        # Run authentic backtest with trajectory tracking
        backtest_result = self.run_tirex_backtest_with_trajectory_tracking(
            window, strategy_config
        )
        
        # Extract positions WITH full confidence and trajectory preservation
        authentic_positions = self.extract_trajectory_positions(backtest_result)
        
        # Create multi-horizon trajectory oracle
        trajectory_oracle = self.create_multi_horizon_oracle(
            authentic_positions, window.market_data
        )
        
        # Run comprehensive trajectory ODEB analysis
        return self.calculate_trajectory_odeb_with_confidence_weighting(
            authentic_positions, trajectory_oracle
        )
```

---

## 📊 Success Criteria & Validation Gates

### **Phase 1 Success Gates (Immediate)**
- ✅ **CLPR Resolution**: All TiRex confidence information preserved through ODEB pipeline
- ✅ **CIA Implementation**: Oracle inherits TiRex confidence distributions correctly  
- ✅ **RAEW Integration**: Regime-aware weighting produces differentiated ODEB scores
- ✅ **TCOP Preservation**: No look-ahead bias introduced by regime adaptation
- ✅ **CITF Validation**: All integration tests pass with >95% confidence

### **Phase 2 Enhancement Gates**  
- ✅ **MHDCA Implementation**: Multi-horizon ODEB analysis operational
- ✅ **TFOM Integration**: Trajectory oracle maintains confidence fidelity over N-bar predictions
- ✅ **Multi-Horizon Validation**: Different holding periods show appropriate oracle benchmarking
- ✅ **Performance Validation**: Enhanced ODEB provides more nuanced strategy evaluation

### **Phase 3 Advanced Gates**
- ✅ **WFSGE Completion**: Mock walk-forward replaced with authentic strategy execution  
- ✅ **ASBI Integration**: Real TiRex strategy results with trajectory confidence preservation
- ✅ **Full Pipeline Validation**: End-to-end authentic TiRex Strategy vs Trajectory Oracle analysis
- ✅ **Production Readiness**: System scales to full walk-forward parameter optimization

---

## 🎯 Key Implementation Files & Integration Points

### **Core Enhancement Files**
- **Position Enhancement**: `src/sage_forge/reporting/performance.py` (add TiRexNativePosition)
- **Oracle Creation**: `src/sage_forge/benchmarking/trajectory_fidelity_oracle.py` (new)
- **Regime-Aware ODEB**: `src/sage_forge/benchmarking/regime_aware_odeb.py` (new)  
- **Walk-Forward Enhancement**: `src/sage_forge/optimization/authentic_walk_forward_engine.py` (new)

### **Integration Testing Files**
- **CITF Framework**: `test_comprehensive_integration_framework.py` (new)
- **TCOP Validation**: `test_temporal_causal_ordering_preservation.py` (new)
- **CLPR Testing**: `test_confidence_leak_problem_resolution.py` (new)

### **Planning & Documentation Files**  
- **This Document**: `docs/planning/tirex_native_odeb_comprehensive_architecture.md`
- **Project Glossary**: `sage-forge-professional/docs/glossary/tirex_native_odeb_concepts_glossary.md`
- **Phase Implementation Tracking**: `docs/planning/phase_implementation_tracking.md` (new)

---

## 📚 Cross-References & Dependencies

### **Prerequisite Components (Phase 3A Complete)**
- TiRex→NT integration with Creative Bridge Configuration Resolution (CBCR)
- Order filling validation with Temporal Causal Ordering Preservation (TCOP)
- Comprehensive regression guards for Context Boundary Prediction Phases (CBPP)
- Adversarial foundation audit with multi-layer validation framework

### **Integration Dependencies**
- **NTPA Compliance**: All enhancements maintain NT-native patterns
- **CFUP Integration**: Multi-agent orchestration for complex ODEB analysis
- **FPPA Compatibility**: Enhanced visualizations for multi-horizon trajectory analysis
- **APCF Documentation**: All concepts canonicalized for SR&ED compliance

---

**Document Status**: ✅ **IMPLEMENTATION READY**  
**Next Action**: Phase 1 CLPR implementation with enhanced Position dataclass  
**Critical Path**: Confidence Leak Problem Resolution enables all subsequent enhancements  
**Success Metric**: End-to-end TiRex confidence preservation through ODEB analysis

---

**Last Updated**: 2025-08-06  
**Version**: 1.0  
**Implementation Phase**: Phase 3B - TiRex-Native ODEB Integration  
**Canonicalization**: All concepts follow project-wide naming convention with glossary integration