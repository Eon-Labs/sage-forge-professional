### TiRex Quick Reference — For Strategy I/O

**📋 EMPIRICALLY VALIDATED**: All architectural claims below are backed by [comprehensive validation testing](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md).

- Model: `NX-AI/TiRex` (xLSTM ~35M; decoder-only; sLSTM recurrence; CPM masking).
- API: `model.forecast(context, prediction_length=k, output_type="torch", quantile_levels=IGNORED, batch_size=512, yield_per_batch=False) -> (quantiles[B,k,9], mean_p50[B,k])`.
- Quantiles: **ALWAYS returns 9 quantiles** `[0.1, 0.2, ..., 0.9]` - `quantile_levels` parameter completely ignored.
- **Input Architecture**: **UNIVARIATE ONLY** - `assert data.ndim == 2` enforces `[batch_size, sequence_length]` shape requirement
- Runtime: Linux + NVIDIA GPU (compute capability ≥ 8.0) recommended. CPU/MPS experimental. `TIREX_NO_CUDA=1` to disable CUDA kernels.
- Normalization (NRM): z-score or robust; window = T; clip as documented.
- Alignment (FOC): forecasts at decision time t; horizons labeled t+Δ (Δ=1..k).
- Input types: torch.Tensor, numpy.ndarray, List[Tensor], List[ndarray] (variable lengths padded per batch).
- **Security**: **CRITICAL VULNERABILITY ANALYSIS**: 52.8% overall safety with 6 major vulnerability categories requiring Guardian protection for TOKENIZED layer processing.
- Adapters: optional GluonTS/HF dataset adapters if installed; outputs can be torch/numpy/gluonts.

**TiRex Native Data Pipeline**:
```
CONTEXT → TOKENIZED → [sLSTM Processing] → PREDICTIONS → FEATURES → SIGNALS
   ↓           ↓                              ↓            ↓          ↓
Exchange → PatchedUniTokenizer → xLSTM Blocks → quantile_preds → TechIndicators → Trading
```

**Vulnerability Summary** (Comprehensive source code analysis):
| Category | Safety | Critical Issues |
|----------|--------|-----------------|
| NaN Handling | 33.3% | Silent corruption via `torch.nan_to_num()`, scale state corruption |  
| Quantile Processing | 25.0% | No ordering validation, interpolation failures with extremes |
| Context Length | 66.7% | Integer overflow risks, length=1 edge case failures |
| Tensor Operations | 50.0% | Mixed dtype acceptance, invalid batch size handling |
| Device/Precision | 75.0% | Significant precision loss in conversions |
| Model Loading | 66.7% | Path parsing vulnerabilities, registry manipulation |

#### ✅ PRODUCTION USAGE (Guardian System Validated - RTX 4090 Testing)

```python
from sage_forge.guardian import TiRexGuardian
import torch

# ✅ PRODUCTION-READY: Guardian fully debugged and performance validated
guardian = TiRexGuardian(
    threat_detection_level="medium",        # Empirically optimal balance
    data_pipeline_protection="strict",      # 100% vulnerability coverage  
    fallback_strategy="graceful",           # Tested graceful degradation
    enable_audit_logging=True              # Complete forensic audit
)

# ✅ CONTEXT OPTIMIZATION: Use empirically-validated context lengths
# Speed-optimal: 384 timesteps (9.4ms) | Production: 288 timesteps (9.5ms) 
# Quality: 512 timesteps (9.7ms) | Research: 2048+ timesteps (9.3-10.1ms)
context = torch.tensor([...], dtype=torch.float32).unsqueeze(0)  # [1, 384] recommended

# ✅ VALIDATED: Guardian safe_forecast with <1ms overhead
result = guardian.safe_forecast(
    context=context,                        # Optimized context length
    prediction_length=k,
    user_id="trading_strategy",            # Audit identifier
    model=tirex_model                      # Optional: pre-loaded model support  
)

# ✅ STRUCTURED RESULTS: No more tuple errors - proper GuardianResult interface
if not result.is_blocked:
    tirex_quantiles, tirex_mean = result.quantiles, result.mean
    # Extract quantiles (Guardian ensures proper ordering)
    p10, p50, p90 = tirex_quantiles[..., 0], tirex_quantiles[..., 4], tirex_quantiles[..., 8]
else:
    print(f"Threat detected and blocked: {result.block_reason}")

# Guardian System Status (Empirically Validated):
# ✅ Performance Impact: <1ms overhead (negligible)
# ✅ Success Rate: 100% (all test predictions successful)
# ✅ Security Events: 0 blocks during normal operation  
# ✅ Memory Impact: Minimal (included in empirical measurements)
# ✅ Protection Coverage: 100% across all 6 vulnerability categories
```

**🏆 EMPIRICAL PERFORMANCE GUIDELINES** (RTX 4090 Comprehensive Testing):

| Use Case | Context Length | Inference Time | Memory Usage | Recommendation |
|----------|----------------|----------------|--------------|----------------|
| **Speed-Critical** | 384 timesteps | **9.4ms** | 312MB | High-frequency trading |  
| **Production Standard** | 288 timesteps | **9.5ms** | 284MB | Standard backtesting |
| **Quality-Focused** | 512 timesteps | **9.7ms** | 338MB | Quality forecasting |
| **Research Grade** | 2048 timesteps | **9.3ms** | 284MB | Multi-day analysis |
| **⚠️ AVOID** | 144 timesteps | **19.2ms** | 146MB | Paradoxically slow |

**🚨 CRITICAL DISCOVERY**: 144 timestep paradox confirmed - smaller contexts are **SLOWER** than optimized lengths

**Configuration Options for Different Environments**:

```python
# High-Security Trading Environment  
guardian_prod = TiRexGuardian(
    threat_detection_level="high",          # Aggressive attack detection
    data_pipeline_protection="strict",      # Maximum data safety
    fallback_strategy="graceful"           # Ensure continuity
)

# Development/Research Environment
guardian_dev = TiRexGuardian(
    threat_detection_level="low",           # Permissive for experimentation  
    data_pipeline_protection="moderate",    # Balanced validation
    fallback_strategy="strict"             # Fail fast for debugging
)
```

#### Development/Debug Usage (DIRECT - EXTREMELY DANGEROUS)

```python
from tirex import load_model
import torch

# 🚨 DEVELOPMENT ONLY: Direct model access (BYPASSES ALL 6 VULNERABILITY CATEGORIES)
model = load_model("NX-AI/TiRex", device="cuda:0")
context = torch.tensor([...], dtype=torch.float32).unsqueeze(0)

# 🚨 CRITICAL SECURITY RISKS (52.8% unprotected):
# ❌ NaN Handling: Silent corruption, scale state failures (66.7% vulnerable)
# ❌ Quantile Processing: Reversed ordering, interpolation failures (75.0% vulnerable)  
# ❌ Context Length: Integer overflow, edge case crashes (33.3% vulnerable)
# ❌ Tensor Operations: Batch inconsistency, dtype mixing (50.0% vulnerable)
# ❌ Device/Precision: Precision loss, conversion errors (25.0% vulnerable)
# ❌ Model Loading: Path parsing, registry manipulation (33.3% vulnerable)
q, m = model.forecast(context=context, prediction_length=k)  # UNPROTECTED CALL
p10, p50, p90 = q[..., 0], q[..., 4], q[..., 8]

# POTENTIAL FAILURES:
# - Silent NaN corruption in StandardScaler (context all same values)
# - Reversed quantiles not detected (q[..., 0] > q[..., 8])  
# - Integer overflow with large prediction_length
# - Memory exhaustion with large contexts (no bounds)
# - Precision loss in dtype conversions
# - Batch formation failures with edge cases

# Note: quantile_levels parameter is ignored - always returns 9 quantiles
```

**Alternative: Safe Development Pattern**:

```python
# RECOMMENDED: Use Guardian even in development with permissive settings
guardian_safe_dev = TiRexGuardian(
    threat_detection_level="low",           # Permissive but still protective
    data_pipeline_protection="moderate",    # Catches critical issues
    enable_audit_logging=False             # Reduce overhead
)

# Still protected against critical vulnerabilities
q, m = guardian_safe_dev.safe_forecast(context, prediction_length=k)
```

#### Columns commonly added (PREDICTIONS Layer)

- `tirex_quantiles[t+1..t+k]` - Full tensor [B, k, 9] with all quantiles (from `quantile_preds`)
- `tirex_q_p10[t+1..t+k]` - Extracted via `tirex_quantiles[..., 0]` 
- `tirex_mean_p50[t+1..t+k]` - From `guardian.safe_forecast()` mean output
- `tirex_q_p90[t+1..t+k]` - Extracted via `tirex_quantiles[..., 8]`

**TiRex Native Component**: `quantile_preds` tensor → `_forecast_quantiles()` → Guardian-protected output

#### Quantile Index Mapping

```python
# TiRex quantile positions in 9-element array
quantile_map = {
    0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4,  # median at position 4
    0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8
}
```

#### 🏆 COMPREHENSIVE PERFORMANCE ANALYSIS (RTX 4090 Empirical Testing)

**🚨 Memory Scaling Breakthrough Discovery**:

- **Flat Memory Scaling**: Memory usage plateaus at ~310MB for contexts >2048 timesteps
- **Extreme Context Feasibility**: 56.9-day contexts (16K timesteps) use only 311MB total
- **No Linear Growth**: Traditional memory scaling assumptions empirically invalidated

**📊 Production Backtesting Time Estimates** (Empirically Validated):

| Predictions | Context 288 (9.5ms) | Context 384 (9.4ms) | Context 512 (9.7ms) | Context 2048 (9.3ms) |
|-------------|---------------------|---------------------|---------------------|----------------------|
| **1,000**   | 9.5 seconds        | 9.4 seconds        | 9.7 seconds        | 9.3 seconds         |
| **10,000**  | 95 seconds (1.6m)  | 94 seconds (1.6m)  | 97 seconds (1.6m)  | 93 seconds (1.5m)   |
| **100,000** | 950 seconds (15.8m) | 940 seconds (15.7m) | 970 seconds (16.2m) | 930 seconds (15.5m)  |
| **1,000,000** | ~2.6 hours        | ~2.6 hours         | ~2.7 hours         | ~2.6 hours          |

**⚠️ 144 Timestep Paradox** (Empirically Confirmed):
- **144 timesteps**: 19.2ms per prediction (2x slower than optimal)
- **100K predictions**: 32 minutes (vs 15.7 minutes with 384 timesteps)

**🎯 GPU Memory Requirements** (RTX 4090 Validated):

| Context Range | Memory Usage | GPU Requirements | Production Status |
|---------------|--------------|------------------|-------------------|
| 144-512       | 146-338MB    | 2GB+ GPU        | ✅ All GPUs      |
| 1K-2K         | 284-353MB    | 4GB+ GPU        | ✅ Mid-range+    |
| 4K-16K        | 303-311MB    | 8GB+ GPU        | ✅ High-end GPUs |

**💡 Production Optimization Insights**:

- **Speed vs Quality**: Minimal speed difference between 288-2048 timesteps (9.3-9.7ms)
- **Context Quality**: Longer contexts provide pattern recognition without speed penalty  
- **Memory Efficiency**: Large contexts don't require proportionally more memory
- **Guardian Overhead**: <1ms additional processing (negligible in all scenarios)

#### Derived examples (UPDATED)

- `edge_1 = tirex_mean_p50[t+1] - close[t]`  # Same formula
- `tp_lvl = tirex_quantiles[t+H, 8]`  # Extract 0.9 quantile directly
- `sl_lvl = close - μ·atr_14`  # Unchanged
