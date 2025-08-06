# Canonicalized Concepts Glossary

**Project**: SAGE-Forge NT Integration  
**Created**: 2025-08-06  
**Purpose**: Comprehensive glossary of all "Name-It-To-Tame-It" concepts for precise conceptual boundaries  
**Status**: Phase 3B Implementation Ready

---

## 📖 Glossary Organization

This glossary follows **Layered Architecture Conceptual Hierarchy** with unique, representative naming convention. All terms are **acronymizable** for concise reference while maintaining full descriptive clarity.

---

## 🏗️ Layer 1: TiRex Native Model (Foundational Intelligence)

### **Multi-Horizon Trajectory Forecasting Capability (MHTFC)**
- **Definition**: TiRex's native ability to predict 1-768+ bars ahead with full quantile uncertainty distributions
- **Current State**: Underutilized in current implementation (using prediction_length=1 only)
- **Enhancement Potential**: Multi-step oracle trajectory modeling for realistic benchmarking
- **Key Implementation**: `repos/tirex/src/tirex/models/tirex.py:137-187` (_forecast_tensor method)
- **Integration Files**: `src/sage_forge/models/tirex_model.py:239` (prediction_length parameter)
- **Discovery Context**: TiRex repository analysis revealing prediction_length=768 capability
- **Usage**: "The MHTFC enables trajectory-based oracle modeling instead of single-point predictions."

### **Temporal Causal Ordering Preservation (TCOP)**  
- **Definition**: All regime detection and predictions use strictly historical context (T-1) without look-ahead bias
- **Validation Method**: Regime detection uses `price_buffer` (historical prices only), predictions use context up to T-1  
- **Critical Property**: Preserves walk-forward analysis validity with regime adaptation
- **Key Implementation**: `src/sage_forge/models/tirex_model.py:199-229` (get_market_regime method)
- **Integration Files**: `src/sage_forge/strategies/tirex_sage_strategy.py:362-365` (bar processing timeline)
- **Discovery Context**: Adversarial analysis confirming no future information leakage in regime detection
- **Usage**: "TCOP validation ensures regime-adaptive ODEB analysis remains bias-free."

### **Context Boundary Prediction Phases (CBPP)**
- **Definition**: Three distinct TiRex prediction phases with different reliability characteristics
- **Phase Details**:  
  - WARM_UP_PERIOD (bars 0-127): No predictions, insufficient context
  - CONTEXT_BOUNDARY (bar 128): First prediction with minimum context  
  - STABLE_WINDOW (bars 129+): Rolling window predictions with full context
- **ODEB Integration Rule**: Only STABLE_WINDOW predictions should participate in evaluation
- **Key Implementation**: `src/sage_forge/models/tirex_model.py:375` (prediction_phase determination)
- **Integration Files**: `src/sage_forge/models/prediction_boundary_validator.py` (comprehensive validation utility)
- **Discovery Context**: Boundary condition analysis resolving 22 vs 23 prediction discrepancy
- **Usage**: "CBPP segregation ensures ODEB analysis excludes developmental predictions from evaluation."

### **Regime-Aware Continuity Pattern (RACP)**
- **Definition**: TiRex maintains 128-bar context continuity across regime changes without context reset
- **Behavioral Characteristic**: Regime detection serves as confidence threshold adjustment, not context boundary trigger
- **Integration Requirement**: Walk-forward windows must respect context continuity for authentic TiRex modeling
- **Key Implementation**: `src/sage_forge/strategies/tirex_sage_strategy.py:416-452` (adaptive threshold management)
- **Integration Files**: `src/sage_forge/models/tirex_model.py:90-98` (rolling buffer management)
- **Discovery Context**: Investigation of TiRex native regime handling revealing continuity preservation
- **Usage**: "RACP ensures walk-forward analysis respects TiRex's proven rolling context methodology."

---

## 🔧 Layer 2: TiRex Translated Strategy (Operational Intelligence)

### **Confidence-Weighted Position Sizing Protocol (CWPSP)**
- **Definition**: Position sizing algorithm using TiRex confidence scores combined with regime-specific multipliers
- **Current Implementation**: `base_size * regime_multiplier * confidence_factor`  
- **Critical Issue**: Confidence information lost downstream in ODEB position dataclass
- **Key Implementation**: `src/sage_forge/strategies/tirex_sage_strategy.py:469-491` (position size calculation)
- **Integration Files**: `src/sage_forge/risk/position_sizer.py` (position sizing utilities)
- **Discovery Context**: Analysis of TiRex strategy revealing confidence-based sizing with information loss
- **Usage**: "CWPSP demonstrates sophisticated confidence utilization that must be preserved in ODEB analysis."

### **Adaptive Confidence Threshold Management (ACTM)**  
- **Definition**: Dynamic confidence thresholds (8%-20%) adjusted based on detected market regime characteristics
- **Threshold Mapping**: High volatility markets = lower thresholds, Low volatility markets = higher thresholds
- **Oracle Integration Requirement**: Oracle should inherit same threshold sensitivity patterns for authentic benchmarking
- **Key Implementation**: `src/sage_forge/strategies/tirex_sage_strategy.py:416-452` (regime-based threshold calculation)
- **Integration Files**: `src/sage_forge/strategies/adaptive_regime.py:197-216` (regime transition handling)  
- **Discovery Context**: Comprehensive analysis of regime-adaptive confidence thresholds in TiRex strategy
- **Usage**: "ACTM provides regime-aware signal filtering that oracle benchmarking must replicate."

### **Creative Bridge Configuration Resolution (CBCR)**
- **Definition**: Multi-path configuration handling ensuring proper TiRex→NT instrument binding across different config types
- **Achievement Status**: Successfully resolved Phase 3A critical integration barriers  
- **Pattern Implementation**: Fallback configuration discovery with direct instrument_id subscription bypass
- **Key Implementation**: `src/sage_forge/strategies/tirex_sage_strategy.py:298-327` (configuration bridge logic)
- **Integration Files**: `src/sage_forge/backtesting/tirex_backtest_engine.py` (configuration management)
- **Discovery Context**: Phase 3A debugging resolving configuration access and bar subscription issues
- **Usage**: "CBCR pattern enables robust TiRex integration across various NT configuration scenarios."

---

## 🎯 Layer 3: ODEB-ized Oracle (Omniscient Benchmarking Intelligence)

### **Trajectory Fidelity Oracle Modeling (TFOM)**
- **Definition**: Oracle maintains TiRex's multi-step forecast confidence profile while optimizing execution timing across trajectories
- **Revolutionary Approach**: N-bar perfect trajectory alignment instead of traditional single-bar perfect timing
- **Implementation Strategy**: Oracle uses same quantile uncertainty distributions with perfect entry/exit execution
- **Enhancement Path**: Multi-horizon trajectory predictions for realistic oracle benchmarking
- **Key Implementation**: To be implemented in `src/sage_forge/benchmarking/trajectory_fidelity_oracle.py`
- **Integration Files**: Integration with `src/sage_forge/models/tirex_model.py` (multi-horizon predictions)
- **Discovery Context**: Analysis of TiRex multi-horizon capability revealing trajectory-based oracle potential  
- **Usage**: "TFOM creates realistic oracle benchmarks maintaining TiRex's uncertainty characteristics."

### **Internal Rival Paradigm Benchmarking (IRPB)**
- **Definition**: ODEB benchmarks TiRex Strategy against "Omniscient TiRex Oracle" rather than external market benchmarks
- **Paradigm Shift**: Strategy vs Perfect-Information-Same-Model instead of Strategy vs Market-Index comparison
- **Core Principle**: Oracle maintains TiRex's prediction regime, confidence patterns, and position sizing logic
- **Measurement Focus**: Capture percentage of oracle-level performance achievable by translated strategy
- **Key Implementation**: `sage-forge-professional/docs/implementation/tirex/odeb-benchmark-specification.md` (specification)
- **Integration Files**: `src/sage_forge/reporting/performance.py` (ODEB framework integration)
- **Discovery Context**: Clarification of ODEB methodology as internal rather than external benchmarking
- **Usage**: "IRPB provides meaningful performance measurement against TiRex's optimal theoretical performance."

### **Confidence Inheritance Architecture (CIA)**  
- **Definition**: Oracle inherits TiRex confidence distributions and position sizing patterns while providing perfect timing execution
- **Design Options**:
  - Option A: Same confidence distribution + perfect timing  
  - Option B: Perfect confidence (1.0) + TiRex-like position sizing
- **Integration Purpose**: Resolves Confidence Leak Problem through end-to-end confidence preservation
- **Key Implementation**: To be implemented in `src/sage_forge/benchmarking/confidence_inheritance_oracle.py`
- **Integration Files**: Enhanced `src/sage_forge/reporting/performance.py` (Position dataclass with confidence)
- **Discovery Context**: Analysis revealing confidence information loss in current ODEB Position dataclass
- **Usage**: "CIA ensures oracle benchmarking maintains TiRex's confidence-based decision patterns."

---

## 📊 Layer 4: ODEB Method (Capture Efficiency Quantification)

### **Multi-Horizon Directional Capture Analysis (MHDCA)**
- **Definition**: ODEB evaluation across multiple TiRex prediction horizons (1, 5, 15, 30 bars) with appropriate weighting
- **Methodology**: Different holding periods receive different oracle benchmarks based on prediction horizon length
- **Aggregation Strategy**: Weighted final score based on horizon representation in actual strategy performance
- **Enhancement Value**: Captures TiRex's true multi-step forecasting capability for comprehensive evaluation
- **Key Implementation**: To be implemented in `src/sage_forge/benchmarking/multi_horizon_odeb_analyzer.py`
- **Integration Files**: Integration with `src/sage_forge/models/tirex_model.py` (multi-horizon prediction support)
- **Discovery Context**: TiRex repository analysis revealing underutilized multi-horizon forecasting capability
- **Usage**: "MHDCA provides nuanced ODEB analysis respecting different prediction horizon characteristics."

### **Confidence Leak Problem Resolution (CLPR)**  
- **Definition**: Systematic preservation of TiRex confidence information through entire ODEB evaluation pipeline
- **Current Issue Identified**: Position dataclass lacks confidence field despite TiRex using confidence for position sizing
- **Solution Architecture**: Enhanced Position dataclass with confidence, regime, and prediction phase fields
- **Validation Requirement**: End-to-end confidence flow verification from TiRex prediction through ODEB analysis
- **Key Implementation**: Enhancement of `src/sage_forge/reporting/performance.py:381-392` (Position dataclass)
- **Integration Files**: All ODEB analysis files requiring confidence-aware position handling
- **Discovery Context**: Deep-dive analysis revealing critical information loss in ODEB position representation
- **Usage**: "CLPR ensures TiRex's sophisticated confidence-based decision making is preserved in evaluation."

### **Regime-Aware Evaluation Weighting (RAEW)**
- **Definition**: ODEB analysis weighted by regime stability and transition characteristics during evaluation periods
- **Weighting Methodology**: More stable regimes receive higher confidence weighting in oracle calculation methodology
- **Integration Strategy**: Accounts for TiRex's regime-based confidence threshold variations in evaluation
- **Implementation Options**: Position-level regime weighting vs window-level aggregated weighting
- **Key Implementation**: To be implemented in `src/sage_forge/benchmarking/regime_aware_odeb.py`
- **Integration Files**: Integration with `src/sage_forge/strategies/adaptive_regime.py` (regime detection)
- **Discovery Context**: Analysis of TiRex regime sensitivity revealing need for regime-aware evaluation
- **Usage**: "RAEW provides regime-sensitive ODEB analysis reflecting TiRex's adaptive threshold behavior."

---

## ⏭️ Layer 5: Walk-Forward Analysis (Temporal Validation)

### **Walk-Forward Simulation Gap Elimination (WFSGE)**
- **Definition**: Replacement of mock position generation with authentic TiRex strategy backtesting in walk-forward windows
- **Current Issue Identified**: Framework exists but uses synthetic positions instead of real strategy results  
- **Solution Strategy**: Execute real TiRex strategy on window data with complete confidence preservation
- **Validation Method**: Authentic Strategy vs Oracle comparison with preserved confidence patterns
- **Key Implementation**: Enhancement of `src/sage_forge/optimization/tirex_parameter_optimizer.py:527-548`
- **Integration Files**: `src/sage_forge/backtesting/tirex_backtest_engine.py` (authentic backtesting integration)
- **Discovery Context**: Walk-forward analysis revealing mock position usage instead of authentic strategy execution
- **Usage**: "WFSGE provides authentic walk-forward validation using real TiRex strategy performance."

### **Authentic Strategy Backtesting Integration (ASBI)**
- **Definition**: Real TiRex strategy execution within walk-forward windows with full confidence and regime preservation
- **Architecture Strategy**: Extract positions WITH confidence from actual backtesting results rather than simulation
- **Oracle Creation Method**: Build oracle using authentic position confidence patterns with perfect timing execution
- **Integration Requirement**: Maintains Temporal Causal Ordering Preservation throughout analysis process
- **Key Implementation**: To be implemented in `src/sage_forge/optimization/authentic_walk_forward_engine.py`
- **Integration Files**: Integration with `src/sage_forge/backtesting/tirex_backtest_engine.py`
- **Discovery Context**: Walk-forward analysis gaps revealing need for authentic strategy execution
- **Usage**: "ASBI ensures walk-forward ODEB analysis uses genuine TiRex strategy decision patterns."

### **Look-Ahead Bias Prevention Validation (LABPV)**
- **Definition**: Comprehensive validation ensuring no future information leakage in regime-adaptive ODEB analysis
- **Validation Proof Method**: All regime detection confirmed to use price_buffer (historical prices only)
- **Timeline Verification**: T-1 context → regime detection + prediction → T execution validation
- **Critical Property**: Walk-forward analysis remains valid with regime adaptation enhancements
- **Achievement Status**: Confirmed bias-free integration of adaptive regime detection capabilities
- **Key Implementation**: `test_temporal_causal_ordering_preservation.py` (comprehensive validation suite)
- **Integration Files**: All regime-adaptive components requiring bias prevention validation
- **Discovery Context**: Adversarial analysis of regime adaptation confirming temporal causal ordering preservation
- **Usage**: "LABPV provides rigorous validation that regime-adaptive enhancements maintain walk-forward validity."

---

## 🔧 System Integration Layer

### **Phase-Based Implementation Strategy (PBIS)**  
- **Definition**: Structured implementation approach separating immediate critical needs from future enhancement capabilities
- **Phase Breakdown**:
  - Phase 1 (Immediate): Confidence Leak Problem Resolution + Regime-Aware Evaluation Weighting
  - Phase 2 (Enhancement): Multi-Horizon Directional Capture Analysis + Trajectory Fidelity Oracle Modeling
  - Phase 3 (Advanced): Full Walk-Forward Simulation Gap Elimination with multi-horizon trajectory analysis
- **Strategic Rationale**: Maintains proven patterns while systematically adding missing capabilities
- **Key Implementation**: `sage-forge-professional/docs/planning/phase_implementation_tracking.md` (implementation coordination)
- **Integration Files**: All enhancement files following phased development approach
- **Discovery Context**: Comprehensive analysis revealing implementation complexity requiring phased approach
- **Usage**: "PBIS ensures systematic enhancement delivery without disrupting proven TiRex integration patterns."

### **Comprehensive Integration Testing Framework (CITF)**
- **Definition**: Multi-layer validation ensuring all canonicalized concepts work together without interference or conflicts
- **Test Categories**: 
  - Temporal Causality Tests (TCOP validation)
  - Confidence Flow Tests (CLPR validation)  
  - Regime Continuity Tests (RACP validation)
  - Oracle Fidelity Tests (TFOM + IRPB validation)
- **Success Criteria**: All integration tests pass with no look-ahead bias introduction
- **Key Implementation**: `test_comprehensive_integration_framework.py` (multi-layer testing suite)
- **Integration Files**: All component integration requiring comprehensive validation
- **Discovery Context**: Complex integration analysis revealing need for systematic validation framework
- **Usage**: "CITF provides confidence that all canonicalized concepts integrate without compromising system integrity."

---

## 🔄 Previously Discovered Concepts (Reorganized)

### **Legacy Terms Integrated into New Hierarchy**

#### **From Phase 3A Implementation**
- **Creative Bridge Configuration Resolution (CBCR)** - Now Layer 2: Operational Intelligence
- **Context Boundary Prediction Phases (CBPP)** - Now Layer 1: Foundational Intelligence
- **Temporal Causal Ordering Preservation (TCOP)** - Now Layer 1: Foundational Intelligence

#### **From ODEB Analysis**  
- **Internal Rival Paradigm Benchmarking (IRPB)** - Now Layer 3: Omniscient Benchmarking Intelligence
- **Confidence Leak Problem Resolution (CLPR)** - Now Layer 4: Capture Efficiency Quantification

#### **From Walk-Forward Analysis**
- **Walk-Forward Simulation Gap Elimination (WFSGE)** - Now Layer 5: Temporal Validation

### **Deprecated Terms (Replaced by Canonicalized Versions)**
- "CONFIDENCE LEAK PROBLEM" → **Confidence Leak Problem Resolution (CLPR)**
- "INTERNAL RIVAL PARADIGM" → **Internal Rival Paradigm Benchmarking (IRPB)**
- "WALK-FORWARD SIMULATION GAP" → **Walk-Forward Simulation Gap Elimination (WFSGE)**
- "REGIME-AWARE CONTINUITY PATTERN" → **Regime-Aware Continuity Pattern (RACP)**
- "TRAJECTORY FIDELITY ORACLE" → **Trajectory Fidelity Oracle Modeling (TFOM)**
- "TEMPORAL CAUSAL ORDERING" → **Temporal Causal Ordering Preservation (TCOP)**

---

## 📚 Cross-Reference Matrix

### **File-to-Concept Mapping**

#### **Core TiRex Integration Files**
- `src/sage_forge/models/tirex_model.py` → MHTFC, TCOP, CBPP, RACP
- `src/sage_forge/strategies/tirex_sage_strategy.py` → CWPSP, ACTM, CBCR
- `repos/tirex/src/tirex/models/tirex.py` → MHTFC (native capability)

#### **ODEB Framework Files**  
- `src/sage_forge/reporting/performance.py` → CLPR (enhancement needed)
- `sage-forge-professional/docs/implementation/tirex/odeb-benchmark-specification.md` → IRPB
- To be created: `src/sage_forge/benchmarking/*.py` → TFOM, CIA, MHDCA, RAEW

#### **Walk-Forward Analysis Files**
- `src/sage_forge/optimization/tirex_parameter_optimizer.py` → WFSGE (enhancement needed)
- To be created: `src/sage_forge/optimization/authentic_walk_forward_engine.py` → ASBI

#### **Testing Framework Files**
- To be created: `test_comprehensive_integration_framework.py` → CITF
- `src/sage_forge/models/prediction_boundary_validator.py` → CBPP validation

### **Concept-to-Phase Mapping**

#### **Phase 1 (Immediate Implementation)**
- CLPR - Confidence Leak Problem Resolution  
- CIA - Confidence Inheritance Architecture
- RAEW - Regime-Aware Evaluation Weighting
- CITF - Comprehensive Integration Testing Framework

#### **Phase 2 (Enhancement Implementation)**
- MHDCA - Multi-Horizon Directional Capture Analysis
- TFOM - Trajectory Fidelity Oracle Modeling
- Enhanced CBPP integration
- Advanced RACP utilization

#### **Phase 3 (Advanced Implementation)**  
- WFSGE - Walk-Forward Simulation Gap Elimination
- ASBI - Authentic Strategy Backtesting Integration
- Full MHTFC utilization
- Advanced LABPV validation

---

## 🎯 Usage Guidelines

### **Acronym Usage Convention**
- **Full Term First Reference**: "Multi-Horizon Trajectory Forecasting Capability (MHTFC)"
- **Subsequent References**: "MHTFC" or "the MHTFC system"  
- **Documentation Headers**: Use full terms in headers, acronyms in content
- **Code Comments**: Use acronyms with brief explanation: `# CLPR: Preserve TiRex confidence through pipeline`

### **Concept Boundary Enforcement**
- **Layer Separation**: Each concept belongs to exactly one layer of the architecture
- **Cross-Layer Integration**: Use defined integration points, avoid direct cross-layer dependencies
- **Naming Conflicts**: All terms are unique across layers to prevent conceptual confusion
- **Evolution Path**: New concepts must fit into existing layer hierarchy or propose new layer

### **Implementation Reference Standards**
- **Planning Documents**: Always reference relevant canonicalized concepts
- **Code Comments**: Use acronyms to indicate which concept a code section implements
- **Testing**: Test names should indicate which concept(s) they validate
- **Documentation**: Cross-reference related concepts in each concept's definition

---

**Glossary Status**: ✅ **CANONICALIZED AND IMPLEMENTATION READY**  
**Total Concepts**: 18 canonicalized terms across 6 architectural layers  
**Next Action**: Reference glossary terms in all Phase 1 implementation work  
**Maintenance**: Update glossary as new concepts emerge during implementation

---

**Last Updated**: 2025-08-06  
**Version**: 1.0  
**Integration**: All concepts integrated into comprehensive architecture plan  
**Validation**: All terms follow unique, representative, acronymizable naming convention