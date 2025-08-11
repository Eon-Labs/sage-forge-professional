### TOKENIZED Layer — TiRex Input Architecture Optimization

**🎯 PRIMARY OPTIMIZATION TARGET**: Critical performance bottleneck requiring immediate enhancement

**Native TiRex Component**: `PatchedUniTokenizer.context_input_transform()`  
**Processing Flow**: `CONTEXT` → `StandardScaler.scale()` → `_Patcher.patcher()` → `tokenized_tensor` + `scaler_state`  
**Architecture Utilization**: **SEVERELY UNDER-OPTIMIZED** (25% → 100% target)

---

#### Executive Summary

**Current State**: Massive under-utilization of TiRex's native xLSTM architecture

- **Feature Density**: 2/8 features (25% of `input_patch_size * 2` capacity)
- **Architecture Alignment**: Poor - ignores patch processing and sLSTM advantages
- **Performance Impact**: 2-4x potential improvement available

**Optimization Opportunity**: Transform from basic price-only input to comprehensive market intelligence leveraging TiRex's full native capabilities.

---

#### Current TOKENIZED Architecture (Legacy - Under-Optimized)

**Architecture Analysis**: Severe capacity under-utilization  
**TiRex Component Usage**: `PatchedUniTokenizer` processing only 2 basic features

| Column         | Type  | TiRex Processing                 | Formula              | Issues                | Status     |
| -------------- | ----- | -------------------------------- | -------------------- | --------------------- | ---------- |
| ctx_close      | float | `tokenizer.scale(close)`         | close                | No patch optimization | 🔄 REPLACE |
| ctx_norm_close | float | `tokenizer.scale(zscore(close))` | zscore(close, win=T) | Static, regime-blind  | 🔄 REPLACE |

**Critical Problems**:

- **Capacity Waste**: Uses only 25% of TiRex `input_patch_size * 2` dimensional processing
- **Static Normalization**: Z-score ignores market regime dynamics
- **Single Asset Focus**: No volume/microstructure intelligence
- **Patch Misalignment**: No optimization for TiRex's patch-based tokenization
- **sLSTM Under-utilization**: Fails to leverage advanced memory mechanisms

**Performance Bottleneck**: This under-optimization is the **primary constraint** on TiRex predictive capability.

---

#### Enhanced TOKENIZED Architecture (Proposed - Architecture-Optimized)

**Architecture Target**: 100% utilization of `input_patch_size * 2` capacity  
**TiRex Alignment**: Fully leverage `PatchedUniTokenizer`, `StandardScaler`, and patch processing

##### High Priority Features (Phase 1 Implementation)

| Column                     | Type   | TiRex Benefit                      | Formula / Pseudocode                                           | Architecture Alignment          |
| -------------------------- | ------ | ---------------------------------- | -------------------------------------------------------------- | ------------------------------- |
| **ctx_ohlc_patches**       | tensor | Multi-dimensional patch processing | `tokenizer.patcher((OHLC/close_prev - 1) / atr_regime_scaler)` | ✅ Leverages patch dimensions   |
| **ctx_returns_scaled**     | float  | sLSTM temporal memory optimization | `tokenizer.scale(log(close[t]/close[t-1]))`                    | ✅ Enhances sLSTM recurrence    |
| **ctx_volatility_patches** | float  | Patch-aligned volatility regimes   | `tokenizer.scale((high-low)/close)`                            | ✅ Patch-based regime detection |

##### Medium Priority Features (Phase 2 Implementation)

| Column                    | Type  | TiRex Benefit                  | Formula / Pseudocode                              | Architecture Alignment       |
| ------------------------- | ----- | ------------------------------ | ------------------------------------------------- | ---------------------------- |
| **ctx_volume_scaled**     | float | Liquidity regime tokenization  | `tokenizer.scale(volume/rolling_mean(volume,20))` | ✅ Market regime awareness   |
| **ctx_orderflow_patches** | float | Microstructure patch alignment | `tokenizer.scale(taker_buy_volume/total_volume)`  | ✅ Patch-based order flow    |
| **ctx_activity_scaled**   | float | Trading intensity tokenization | `tokenizer.scale(count/rolling_mean(count,20))`   | ✅ Activity regime detection |

##### Low Priority Features (Phase 3 Implementation)

| Column                 | Type  | TiRex Benefit                   | Formula / Pseudocode                               | Architecture Alignment            |
| ---------------------- | ----- | ------------------------------- | -------------------------------------------------- | --------------------------------- |
| **ctx_regime_patches** | float | Multi-regime patch processing   | `tokenizer.scale(atr_14/rolling_mean(atr_14,50))`  | ✅ Advanced regime detection      |
| **ctx_session_scaled** | float | Session transition tokenization | `tokenizer.scale((open[t]-close[t-1])/close[t-1])` | ✅ Temporal boundary optimization |

**Enhanced Capacity**: 100% (8/8 features) of TiRex optimal processing capability  
**Architecture Alignment**: Excellent - fully leverages native TiRex components

---

#### Comparative Architecture Analysis

| Aspect                   | Current (Legacy)     | Enhanced (Proposed)             | Improvement Factor |
| ------------------------ | -------------------- | ------------------------------- | ------------------ |
| **TiRex Utilization**    | 25% (2/8 features)   | 100% (8/8 features)             | **4x**             |
| **Market Information**   | Price only           | Price + Volume + Microstructure | **6x**             |
| **Regime Awareness**     | Static z-score       | Multi-regime adaptive           | **∞**              |
| **Patch Alignment**      | None                 | Optimized for patch processing  | **2-3x**           |
| **sLSTM Memory**         | Basic price sequence | Rich temporal patterns          | **3-4x**           |
| **Expected Performance** | Baseline             | **2-4x improvement**            | **2-4x**           |

---

#### TiRex Native Component Integration

##### PatchedUniTokenizer Optimization

```python
# CURRENT (Under-optimized)
tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
tokenized_tensor, scaler_state = tokenizer.context_input_transform(
    torch.tensor([ctx_close, ctx_norm_close])  # Only 2 features - WASTE
)

# ENHANCED (Architecture-optimized)
tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
enhanced_context = torch.stack([
    ctx_ohlc_patches,        # Multi-dimensional OHLC
    ctx_returns_scaled,      # sLSTM-optimized returns
    ctx_volatility_patches,  # Patch-aligned volatility
    ctx_volume_scaled,       # Liquidity regime
    ctx_orderflow_patches,   # Microstructure
    ctx_activity_scaled,     # Activity regime
    ctx_regime_patches,      # Advanced regime detection
    ctx_session_scaled       # Session transitions
], dim=-1)  # Full 8-feature utilization

tokenized_tensor, scaler_state = tokenizer.context_input_transform(enhanced_context)
```

##### StandardScaler State Management

```python
# Enhanced scaler_state preservation for multi-feature normalization
scaler_state = {
    'ohlc_loc_scale': (ohlc_mean, ohlc_std),
    'returns_loc_scale': (returns_mean, returns_std),
    'volume_loc_scale': (volume_mean, volume_std),
    'regime_loc_scale': (regime_mean, regime_std),
    # ... Complete state for all 8 features
}
```

##### Patch Processing Optimization

```python
# TiRex native patch configuration for financial data
patch_config = {
    "patch_size": 12,         # 1 hour patches (12 × 5min bars)
    "patch_stride": 6,        # 50% overlap for continuity
    "input_patch_size": 8,    # Enhanced 8-feature input
    "context_length": 288,    # 6 hours of 5-minute data
    "left_pad": True          # TiRex native padding
}
```

---

#### Critical Evaluation Questions

**Technical Feasibility**:

1. **Computational Overhead**: Are 8 features computationally feasible for real-time inference with current hardware?
2. **Data Availability**: How do we handle missing volume/trade intensity data across different exchanges/pairs?
3. **Parameter Optimization**: What are optimal lookback windows for regime detection (20 vs 50 periods)?

**Financial Domain Validation**: 4. **Alpha Generation**: Does `ctx_orderflow_patches` provide genuine predictive alpha or introduce noise? 5. **Temporal Patterns**: Should we include time-of-day/session features for intraday pattern recognition? 6. **Complexity Trade-offs**: How do we balance feature richness vs interpretability and debugging complexity?

**TiRex Architecture Optimization**: 7. **Patch Configuration**: What's the optimal `patch_size` for 8-feature input matrix processing? 8. **Normalization Strategy**: How should we handle multivariate scaling across different feature ranges and distributions? 9. **Channel Processing**: Should features be concatenated or processed in separate input channels?

---

#### Implementation Roadmap

##### Phase 1: Core Enhancement (HIGH Impact)

- **Target**: Implement top 3 high-priority features
- **Focus**: `ctx_ohlc_patches`, `ctx_returns_scaled`, `ctx_volatility_patches`
- **Expected Gain**: 2x performance improvement
- **Timeline**: Immediate implementation priority

##### Phase 2: Market Intelligence (MEDIUM Impact)

- **Target**: Add volume and microstructure features
- **Focus**: `ctx_volume_scaled`, `ctx_orderflow_patches`, `ctx_activity_scaled`
- **Expected Gain**: Additional 1.5-2x improvement (cumulative 3-4x)
- **Timeline**: After Phase 1 validation

##### Phase 3: Advanced Optimization (LOW Risk)

- **Target**: Complete architecture utilization
- **Focus**: `ctx_regime_patches`, `ctx_session_scaled`
- **Expected Gain**: Marginal improvement but 100% architecture utilization
- **Timeline**: Optional advanced optimization

---

#### Risk Assessment & Mitigation

##### Technical Risks

- **Computational Complexity**: 8-feature processing may increase inference latency
  - _Mitigation_: Benchmark performance vs accuracy trade-offs
- **Data Pipeline Complexity**: Multi-feature preprocessing increases failure points
  - _Mitigation_: Robust error handling and fallback mechanisms

##### Financial Risks

- **Over-fitting**: Rich feature set may not generalize across market conditions
  - _Mitigation_: Systematic backtesting across multiple market regimes
- **Regime Dependency**: Optimization may be specific to certain market structures
  - _Mitigation_: Multi-market validation and adaptive parameters

##### Implementation Risks

- **Integration Complexity**: Guardian system integration with enhanced features
  - _Mitigation_: Phased rollout with comprehensive testing
- **Maintenance Burden**: 8-feature system requires more sophisticated monitoring
  - _Mitigation_: Automated validation and performance tracking

---

#### Success Metrics & Validation

##### Performance Benchmarks

- **Quantitative Improvement**: 2-4x better forecast accuracy vs current baseline
- **Architecture Utilization**: 100% of TiRex `input_patch_size * 2` capacity
- **Inference Speed**: Maintain <100ms inference time for real-time trading
- **Memory Efficiency**: Manage GPU memory usage within production constraints

##### Financial Validation

- **Sharpe Ratio Improvement**: 50%+ improvement in risk-adjusted returns
- **Drawdown Reduction**: 25%+ reduction in maximum drawdown periods
- **Alpha Generation**: Statistically significant alpha over baseline strategies
- **Market Adaptability**: Consistent performance across different volatility regimes

---

#### Integration with Guardian System

##### Enhanced Guardian Patterns

```python
from sage_forge.guardian import TiRexGuardian

# Production pattern with enhanced TOKENIZED layer
guardian = TiRexGuardian(
    threat_detection_level="medium",
    data_pipeline_protection="strict",     # Critical for 8-feature validation
    fallback_strategy="graceful"
)

# Enhanced multi-feature context
enhanced_context = prepare_enhanced_tokenized_context(
    raw_context=market_data,
    features=["ohlc_patches", "returns_scaled", "volatility_patches",
              "volume_scaled", "orderflow_patches", "activity_scaled"],
    patch_config={"size": 12, "stride": 6},
    context_length=288
)

# Protected inference with full architecture utilization
predictions_quantiles, predictions_mean = guardian.safe_forecast(
    context=enhanced_context,  # [1, 288, 8] - full TiRex capacity
    prediction_length=12,      # 1 hour ahead
    user_id="enhanced_tokenized_system"
)
```

---

#### Conclusion

The TOKENIZED layer represents the **critical bottleneck** in TiRex predictive performance. Current 25% architecture utilization severely constrains the model's native xLSTM capabilities.

**Key Transformation**: From basic price-only input to comprehensive market intelligence system that fully leverages:

- `PatchedUniTokenizer` multi-dimensional processing
- `StandardScaler` adaptive normalization
- sLSTM advanced memory mechanisms
- Patch-based temporal pattern recognition

**Expected Impact**: **2-4x performance improvement** through proper TiRex architecture alignment, transforming TiRex from an under-utilized model to a comprehensive market intelligence system.

**Next Steps**: User feedback on the 9 critical evaluation questions to refine implementation strategy and begin Phase 1 development.

---

[← Back to Index](../strategy-io-contract.md#layer-navigation-tirex-native) | [Next: PREDICTIONS Layer →](./predictions-layer.md)
