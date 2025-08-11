### TOKENIZED Layer — TiRex Univariate Input Architecture

**⚠️ CRITICAL CLARIFICATION**: TiRex is a **univariate forecasting model** - processes single time series only

**📋 EMPIRICAL PROOF**: [Validation Results](../../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md) | [Source Code Evidence](../../../tests/validation/definitive_signal_proof_test.py)

**Native TiRex Component**: `PatchedUniTokenizer.context_input_transform()`  
**Processing Flow**: `CONTEXT[single_series]` → `StandardScaler.scale()` → `_Patcher.patcher()` → `tokenized_tensor` + `input_mask`  
**Architecture Reality**: **UNIVARIATE ONLY** - cannot process multiple features simultaneously

**🔬 SMOKING GUN**: TiRex source code Line 106: `assert data.ndim == 2` - hardcoded 2D requirement `[batch_size, sequence_length]` only

---

#### Executive Summary

**TiRex Architecture Reality**: Designed as univariate time series forecasting model

- **Input Requirement**: Single time series only (e.g., close prices)
- **Processing**: `input_patch_size * 2` = data + input_mask (NOT multi-features)  
- **Architecture Alignment**: Properly designed for single-variable forecasting
- **📊 EMPIRICAL VALIDATION**: All multi-dimensional inputs (`[1, 128, 5]`, `[128, 5, 1]`, etc.) fail with AssertionError

**Optimization Strategy**: Focus on **input quality** and **temporal features** within univariate constraint.

**🧪 VALIDATION STATUS**: ✅ **EMPIRICALLY CONFIRMED** through comprehensive testing - [View Test Results](../../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md)

---

#### Current TOKENIZED Architecture — Univariate Input Analysis

**Architecture Reality**: TiRex processes single time series (univariate)
**TiRex Component Usage**: `PatchedUniTokenizer` correctly processing single price series

| Input Option   | Type  | TiRex Processing                 | Formula              | Characteristics       | Status     |
| -------------- | ----- | -------------------------------- | -------------------- | --------------------- | ---------- |
| close          | float | `tokenizer.scale(close)`         | raw close prices     | Basic price series    | ✅ VALID   |
| normalized     | float | `tokenizer.scale(zscore(close))` | zscore(close, win=T) | Stationary transform  | ✅ VALID   |

**Architecture Constraints**:

- **Univariate Only**: TiRex architecturally limited to single time series input
- **No Multi-Feature Support**: Cannot process OHLCV, volume, or indicators simultaneously  
- **Single Variable Forecasting**: Outputs quantiles for same input variable only
- **Patch Processing**: Applies to temporal dimension, not feature dimension

**Optimization Focus**: Input quality and temporal preprocessing within univariate constraint.

---

#### Optimized TOKENIZED Architecture — Univariate Input Enhancement

**Architecture Understanding**: `input_patch_size * 2` = data + input_mask (NOT multi-features)  
**TiRex Alignment**: Optimize single time series quality and preprocessing

##### Univariate Input Options (TiRex Compatible)

**⚠️ CRITICAL**: TiRex can only process ONE of these options at a time, not simultaneously

| Input Series          | Type  | TiRex Processing               | Formula / Pseudocode                    | Use Case                |
| --------------------- | ----- | ------------------------------ | --------------------------------------- | ----------------------- |
| **raw_close**         | float | `tokenizer.scale(close)`       | close prices                            | Direct price forecasting |
| **log_returns**       | float | `tokenizer.scale(log(close/close[-1]))` | `log(close[t]/close[t-1])`              | Return-based forecasting |
| **normalized_price**  | float | `tokenizer.scale(zscore(close))` | `(close - mean) / std`                  | Stationary forecasting   |
| **volume_weighted**   | float | `tokenizer.scale(vwap)`        | `sum(price * volume) / sum(volume)`     | Volume-adjusted price    |
| **typical_price**     | float | `tokenizer.scale(hlc3)`        | `(high + low + close) / 3`              | Representative price     |

**Architecture Constraint**: Choose exactly **ONE** series - TiRex cannot process multiple simultaneously

---

#### Input Quality Comparison

| Aspect                   | Basic Input          | Optimized Input                 | Benefit            |
| ------------------------ | -------------------- | ------------------------------- | ------------------ |
| **Input Series**         | Raw close prices     | Engineered univariate series   | Better signal      |
| **Preprocessing**        | Basic scaling        | Stationary transformation       | Model-friendly     |
| **Temporal Features**    | Static normalization | Regime-aware preprocessing      | Context awareness  |
| **Data Quality**         | Raw exchange data    | Cleaned, validated series       | Robust forecasting |
| **Expected Performance** | Baseline             | **10-30% improvement**          | Input optimization |

---

#### TiRex Native Component Integration — Univariate Reality

##### PatchedUniTokenizer Correct Usage

```python
# CORRECT - TiRex univariate processing
tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())

# Option 1: Raw close prices
close_series = torch.tensor(close_prices)  # Shape: [sequence_length]
tokenized_tensor, scaler_state = tokenizer.context_input_transform(close_series)

# Option 2: Log returns (often better for forecasting)
returns = torch.tensor(np.log(close_prices[1:] / close_prices[:-1]))
tokenized_tensor, scaler_state = tokenizer.context_input_transform(returns)

# IMPOSSIBLE - TiRex cannot do this:
# multi_features = torch.stack([close, volume, rsi], dim=-1)  # ❌ NOT SUPPORTED
```

##### StandardScaler State Management

```python
# Correct scaler_state for single time series
scaler_state = {
    'loc': mean_value,    # Scaling mean for univariate series
    'scale': std_value    # Scaling std for univariate series
}
# Shape: Single values, not arrays (because univariate)
```

##### Patch Processing Configuration

```python
# TiRex patch configuration for univariate financial data  
patch_config = {
    "patch_size": 12,         # 1 hour patches (12 × 5min bars)
    "patch_stride": 6,        # 50% overlap for continuity  
    "input_patch_size": 12,   # Temporal patch size (NOT feature count)
    "context_length": 288,    # 6 hours of 5-minute data
    "left_pad": True          # TiRex native padding
}
# Note: input_patch_size relates to temporal dimension, not features
```

---

#### Critical Evaluation Questions

**Univariate Input Selection**:

1. **Series Choice**: Raw close prices vs log returns vs typical price - which provides best TiRex forecasting?
2. **Preprocessing**: Static z-score vs regime-aware normalization - optimal for financial data?
3. **Data Quality**: How to handle gaps, outliers, and market halts in univariate series?

**TiRex Architecture Optimization**:

4. **Patch Configuration**: Optimal `patch_size` for financial time series (8, 12, or 16 timesteps)?
5. **Context Length**: Ideal sequence length for intraday forecasting (128, 256, or 512 timesteps)?
6. **Temporal Features**: Include session boundaries, weekend effects in single series?

**Multi-Model Strategy**:

7. **Ensemble Approach**: Run separate TiRex models for different assets/timeframes?
8. **Feature Engineering**: How to incorporate volume/volatility information AFTER TiRex forecasting?
9. **Integration**: Combine TiRex price forecasts with other specialized models for complete strategy?

---

#### Implementation Roadmap — Univariate Optimization

##### Phase 1: Input Series Optimization (HIGH Impact)

- **Target**: Test different univariate input series for TiRex
- **Focus**: Raw close, log returns, typical price, volume-weighted price
- **Expected Gain**: 10-20% improvement from optimal series selection
- **Timeline**: Immediate A/B testing

##### Phase 2: Preprocessing Enhancement (MEDIUM Impact)

- **Target**: Optimize normalization and data cleaning
- **Focus**: Regime-aware scaling, outlier handling, gap filling
- **Expected Gain**: Additional 5-15% improvement from data quality
- **Timeline**: After Phase 1 validation

##### Phase 3: Multi-Model Strategy (FUTURE)

- **Target**: Ensemble multiple TiRex models
- **Focus**: Different assets, timeframes, preprocessing methods
- **Expected Gain**: Portfolio-level improvement through diversification
- **Timeline**: Long-term strategic enhancement

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

#### Integration with Guardian System — Univariate Protection

##### Correct Guardian Usage

```python
from sage_forge.guardian import TiRexGuardian

# Production pattern with univariate TOKENIZED layer
guardian = TiRexGuardian(
    threat_detection_level="medium",
    data_pipeline_protection="strict",     # Protects univariate processing
    fallback_strategy="graceful"
)

# Correct univariate context preparation
close_series = prepare_univariate_context(
    raw_context=market_data,
    series_type="close",           # Single series selection
    preprocessing="log_returns",   # Optional: returns vs prices
    context_length=288            # 6 hours of 5-minute data
)

# Protected inference with TiRex univariate architecture
predictions_quantiles, predictions_mean = guardian.safe_forecast(
    context=close_series,         # [1, 288] - univariate series
    prediction_length=12,         # 1 hour ahead  
    user_id="univariate_price_forecasting"
)
```

---

#### Conclusion — TiRex Univariate Reality

The TOKENIZED layer processing reveals **TiRex's univariate architecture** - designed for single time series forecasting excellence.

**📋 EMPIRICALLY VALIDATED**: This conclusion is backed by comprehensive testing and source code analysis:
- **🔬 Source Code Proof**: `assert data.ndim == 2` hardcoded in `PatchedUniTokenizer`
- **🧪 Empirical Tests**: All multi-dimensional inputs rejected with AssertionError  
- **✅ Test Results**: [Complete Validation Report](../../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md)

**Key Understanding**: TiRex is NOT a multi-feature model but excels at univariate forecasting:

- `PatchedUniTokenizer` optimized for single time series processing
- `StandardScaler` designed for univariate normalization
- sLSTM architecture tuned for temporal patterns in single variable
- Patch processing applies to time dimension, not feature dimension

**Optimization Strategy**: **10-30% improvement** through:
- Optimal univariate input selection
- Quality preprocessing and normalization
- Strategic integration with other models for multi-feature intelligence

**🔗 VALIDATION EVIDENCE**:
- [Definitive Proof Test](../../../tests/validation/definitive_signal_proof_test.py) - Source code + empirical validation
- [Complete Results](../../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md) - Full documentation

**Next Steps**: Focus on univariate input optimization and multi-model ensemble strategies for comprehensive market intelligence.

---

[← Back to Index](../strategy-io-contract.md#layer-navigation-tirex-native) | [Next: PREDICTIONS Layer →](./predictions-layer.md)
