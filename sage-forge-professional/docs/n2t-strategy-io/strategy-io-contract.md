### Strategy I/O Contract — Template (Tabular Markdown)

Fill this once per study. Keep names stable; add new rows—not new names.

#### Spec Meta (MCC, FOC, IRP)

- strategy_name:
- context_len (T):
- prediction_len (k):
- frequency:
- timezone: UTC
- device: cuda:0 (default) | cuda:N | cpu
- input_types: torch | numpy | list[torch] | list[numpy]
- output_type: torch | numpy | gluonts (default: torch)
- batch_size: 512 (default)
- yield_per_batch: false (default)
- quantile_levels (FOC): IGNORED by TiRex - always returns 9 quantiles [0.1..0.9]
- predict_kwargs: max_context=, max_accelerated_rollout_steps=
- runtime: gpu(cc≥8.0) recommended; torch=, cuda=, TIREX_NO_CUDA={0|1}

#### Input Security & Validation (ISV) — MANDATORY

**CRITICAL**: Comprehensive vulnerability analysis reveals TiRex has **52.8% overall safety** with 6 major vulnerability categories requiring immediate protection.

**Vulnerability Assessment Summary**:

| Category            | Safety Rate | Critical Issues                           | Protection Required    |
| ------------------- | ----------- | ----------------------------------------- | ---------------------- |
| NaN Handling        | 33.3%       | Silent corruption, scale state corruption | DataPipelineShield     |
| Quantile Processing | 25.0%       | Reversed ordering, interpolation failures | Quantile validation    |
| Context Length      | 66.7%       | Integer overflow, padding edge cases      | Length bounds checking |
| Tensor Operations   | 50.0%       | Batch inconsistency, dtype mixing         | Operation validation   |
| Device/Precision    | 75.0%       | Precision loss, device mismatch           | Conversion monitoring  |
| Model Loading       | 66.7%       | Path parsing, registry manipulation       | Loading validation     |

**Enhanced Security Controls**:

| Security Control     | Specification                             | Implementation     | Severity | Guardian Component |
| -------------------- | ----------------------------------------- | ------------------ | -------- | ------------------ |
| NaN_Detection        | Reject >20% NaN (empirically validated)   | InputShield        | CRITICAL | Input validation   |
| Infinity_Guard       | Zero-tolerance ±inf (causes corruption)   | InputShield        | CRITICAL | Input validation   |
| Context_Quality      | Min 3 timesteps, <100K length bounds      | DataPipelineShield | CRITICAL | Data pipeline      |
| Scaling_Safety       | Prevent NaN scale state corruption        | DataPipelineShield | CRITICAL | Data pipeline      |
| Quantile_Ordering    | Auto-correct reversed quantile arrays     | DataPipelineShield | HIGH     | Output validation  |
| Batch_Consistency    | Validate batch size >0, dtype consistency | DataPipelineShield | HIGH     | Data pipeline      |
| Precision_Monitoring | Track conversion accuracy loss            | DataPipelineShield | MEDIUM   | Data pipeline      |
| Attack_Detection     | Pattern recognition, threat scoring       | InputShield        | MEDIUM   | Threat detection   |

**Legacy Manual Validation** (NOT RECOMMENDED - Use Guardian Instead):

```python
def validate_tirex_input(context: torch.Tensor) -> torch.Tensor:
    # WARNING: Manual validation incomplete - missing 5/6 vulnerability categories
    if torch.isnan(context).float().mean() > 0.2:
        raise ValueError("Excessive NaN ratio - potential attack")
    if torch.isinf(context).any():
        raise ValueError("Infinite values detected - model will fail")
    if torch.any(torch.abs(context) > 1e6):
        raise ValueError("Extreme values detected - unrealistic data")
    return context  # Still vulnerable to scaling, quantile, context length issues
```

#### TiRex Guardian Integration (TGI) — PRODUCTION REQUIREMENT

**CRITICAL**: Direct TiRex calls are PROHIBITED in production. Enhanced Guardian system provides 5-layer protection with 100% vulnerability coverage.

**Enhanced Guardian Architecture** (Based on comprehensive source code analysis):

| Guardian Component       | Purpose                      | Vulnerabilities Protected               | Implementation     | Status    |
| ------------------------ | ---------------------------- | --------------------------------------- | ------------------ | --------- |
| **Guardian Entry**       | Main protective interface    | All categories                          | `TiRexGuardian()`  | MANDATORY |
| **Input Shield**         | Input attack protection      | NaN/inf/extreme value injection         | Layer 1 validation | MANDATORY |
| **Data Pipeline Shield** | Data processing safety       | Scaling, quantiles, context, tensor ops | Layer 2 validation | MANDATORY |
| **Circuit Shield**       | Failure handling & fallbacks | Model failures, cascading errors        | Layer 3 protection | MANDATORY |
| **Output Shield**        | Business logic validation    | Forecast reasonableness, ordering       | Layer 4 validation | MANDATORY |
| **Audit Shield**         | Forensic security logging    | Complete inference audit trail          | Layer 5 monitoring | MANDATORY |

**Protection Coverage Matrix**:

| Vulnerability Category | Input Shield | DataPipeline Shield      | Circuit Shield         | Output Shield         | Coverage |
| ---------------------- | ------------ | ------------------------ | ---------------------- | --------------------- | -------- |
| NaN Handling           | ✅ Primary   | ✅ Scaling safety        | -                      | ✅ Output corruption  | 100%     |
| Quantile Processing    | -            | ✅ Ordering validation   | -                      | ✅ Consistency checks | 100%     |
| Context Length         | -            | ✅ Bounds checking       | -                      | -                     | 100%     |
| Tensor Operations      | -            | ✅ Batch validation      | -                      | -                     | 100%     |
| Device/Precision       | -            | ✅ Conversion monitoring | -                      | -                     | 100%     |
| Model Loading          | -            | -                        | ✅ Registry protection | -                     | 100%     |

**Enhanced Guardian Integration Pattern**:

```python
from sage_forge.guardian import TiRexGuardian

# PRODUCTION PATTERN (Enhanced with DataPipelineShield)
guardian = TiRexGuardian(
    threat_detection_level="medium",        # "low", "medium", "high"
    fallback_strategy="graceful",           # "graceful", "strict", "minimal"
    data_pipeline_protection="strict"       # "strict", "moderate", "permissive"
)

# Protected inference with 5-layer validation
tirex_quantiles, tirex_mean = guardian.safe_forecast(
    context=raw_context,     # No pre-validation needed - Guardian handles all
    prediction_length=k,
    user_id="strategy_system"  # Optional for audit trails
)

# Guardian automatically provides:
# - Input attack protection (empirically validated)
# - Data pipeline safety (context, scaling, quantiles)
# - Circuit breaking with graceful fallbacks
# - Output validation and auto-correction
# - Complete audit logging

# PROHIBITED PATTERN (Critical Security Risk)
# quantiles, mean = model.forecast(context, prediction_length=k)  # 47.2% unprotected
```

**Guardian Configuration Options**:

```python
# High-Security Environment
guardian_strict = TiRexGuardian(
    threat_detection_level="high",          # Aggressive threat detection
    data_pipeline_protection="strict",      # Strict data validation
    enable_audit_logging=True              # Complete forensic logging
)

# Development Environment
guardian_dev = TiRexGuardian(
    threat_detection_level="low",           # Permissive for debugging
    data_pipeline_protection="moderate",    # Balanced validation
    fallback_strategy="strict"             # Fail fast for debugging
)
```

**Guardian System Location**: `src/sage_forge/guardian/` - Complete defensive architecture  
**Empirical Evidence**: Comprehensive source code analysis with test validation (`test_tirex_data_pipeline_vulnerabilities.py`)

**Deployment Requirements**:

| Environment | Threat Level | Data Pipeline | Fallback Strategy | Use Case                  |
| ----------- | ------------ | ------------- | ----------------- | ------------------------- |
| Production  | medium/high  | strict        | graceful          | Live trading systems      |
| Staging     | medium       | strict        | graceful          | Pre-production validation |
| Development | low          | moderate      | strict            | Research and debugging    |
| Testing     | high         | strict        | graceful          | Security validation       |

#### Data Dictionary & Feature Registry — TiRex Native Pipeline

**Architecture**: Native TiRex data processing pipeline with architecture-aligned terminology from comprehensive source code analysis.

**📋 EMPIRICAL VALIDATION**: Claims in this document are backed by [definitive empirical testing](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md) and [source code analysis](../../tests/validation/definitive_signal_proof_test.py).

**DSM Source of Truth**: Column nomenclature follows DSM. Timestamps are UTC (millisecond precision), `open_time` = beginning of candle period.

##### TiRex Native Data Flow

```
CONTEXT → TOKENIZED → [sLSTM Processing] → PREDICTIONS → FEATURES → SIGNALS
   ↓           ↓                              ↓            ↓          ↓
Exchange → Patch/Scale → xLSTM Embeddings → Quantiles → Indicators → Trading
(11 cols)   (1 univar)                      (4 cols)    (5 cols)   (3 cols)
```

##### Layer Navigation (TiRex Native)

| Layer                                            | Columns | Status           | File                       | TiRex Component         | Focus                                                                                                                       |
| ------------------------------------------------ | ------- | ---------------- | -------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 📊 [CONTEXT](./layers/context-layer.md)          | 11      | ✅ Complete      | `context-layer.md`         | `context: torch.Tensor` | Exchange data                                                                                                               |
| 🔧 [TOKENIZED](./layers/tokenized-layer.md)      | 1       | ✅ **VALIDATED** | `tokenized-layer.md`       | `PatchedUniTokenizer`   | **[Empirically proven](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md) univariate** |
| 🎯 [PREDICTIONS](./layers/predictions-layer.md)  | 4       | ✅ Stable        | `predictions-layer.md`     | `quantile_preds`        | TiRex outputs                                                                                                               |
| ⚙️ [FEATURES](./layers/features-layer.md)        | 5       | ✅ Stable        | `features-layer.md`        | Post-processing         | Technical indicators                                                                                                        |
| 🚨 [SIGNALS](./layers/signals-layer.md)          | 3       | ✅ Stable        | `signals-layer.md`         | Trading logic           | Decision layer                                                                                                              |
| 🔗 [PIPELINE](./layers/pipeline-dependencies.md) | —       | ✅ Mapped        | `pipeline-dependencies.md` | Data lineage            | Dependency flow                                                                                                             |

##### Architecture Summary

- **Total Columns**: 23 (11 CONTEXT + 1 TOKENIZED + 4 PREDICTIONS + 5 FEATURES + 3 SIGNALS)
- **Architecture Reality**: TOKENIZED layer is univariate by design - **[EMPIRICALLY PROVEN](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md)**
- **Expected Performance**: **10-30% improvement** from optimal univariate input selection
- **Critical Understanding**: TiRex processes single time series only
- **🔬 SOURCE CODE PROOF**: `assert data.ndim == 2` in `PatchedUniTokenizer` enforces `[batch_size, sequence_length]` only

##### Quick Reference

- **TOKENIZED Reality**: **[EMPIRICALLY VALIDATED](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md)** - Univariate input only — [Analysis →](./layers/tokenized-layer.md#univariate-input-options-tirex-compatible)
- **Optimization Strategy**: Input quality and preprocessing within univariate constraint — [Strategy →](./layers/tokenized-layer.md#implementation-roadmap--univariate-optimization)
- **Critical Questions**: Univariate input selection and multi-model integration — [Questions →](./layers/tokenized-layer.md#critical-evaluation-questions)
- **🔗 VALIDATION TESTS**: [Definitive Proof](../../tests/validation/definitive_signal_proof_test.py) | [Complete Results](../implementation/tirex/empirical-validation/TIREX_EMPIRICAL_FINDINGS_COMPREHENSIVE.md)

#### Signals & Risk (SDL, RBR, UQC)

| Signal          | Side  | Entry                          | Exit                    | TP               | SL             | Position Sizing | Cooldown |
| --------------- | ----- | ------------------------------ | ----------------------- | ---------------- | -------------- | --------------- | -------- |
| TIR_TREND_BREAK | Long  | edge_1>λ·atr_14 && close>ma_20 | edge_1<0 or close<ma_20 | tirex_q_p90[t+H] | close-μ·atr_14 | pos_size        | 5 bars   |
| TIR_MR          | Short | edge_1<-λ·atr_14 && rsi_14>70  | rsi_14<50               | tirex_q_p10[t+H] | close+μ·atr_14 | pos_size        | 5 bars   |

#### Backtest & Evaluation (BTP, CAL, OSS, EMT, BAT)

- splits: walk-forward; fixed T/k; frozen features
- metrics: MAE, RMSE, Pinball; PnL, Sharpe, MDD
- assumptions: fees_bps=, slip_bps=, latency_ms=, order=, liq_model=

#### Repro & Governance (ARL, GRT)

- pins: data_hash=, model_hash=, seed=, torch=, cuda=
- approvals/audit:
- rollback plan:
