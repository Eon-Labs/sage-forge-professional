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

**CRITICAL**: TiRex performs NO input validation. Production deployment requires mandatory validation layer.

| Security Control | Specification | Implementation | Severity |
|------------------|---------------|----------------|----------|
| NaN_Detection | Reject if >20% NaN values in context | `validate_nan_ratio(context)` | CRITICAL |
| Infinity_Guard | Reject infinite values (±inf) | `validate_finite_values(context)` | CRITICAL |
| Bounds_Checking | Clip to reasonable range (±1e6) | `validate_value_bounds(context)` | HIGH |
| Data_Quality | Require >80% finite values | `validate_data_quality(context)` | HIGH |
| Attack_Detection | Monitor suspicious patterns | `log_input_anomalies(context)` | MEDIUM |

**Validation Implementation**:
```python
def validate_tirex_input(context: torch.Tensor) -> torch.Tensor:
    # CRITICAL: NaN ratio check  
    if torch.isnan(context).float().mean() > 0.2:
        raise ValueError("Excessive NaN ratio - potential attack")
    
    # CRITICAL: Infinity detection
    if torch.isinf(context).any():
        raise ValueError("Infinite values detected - model will fail")
    
    # HIGH: Value bounds (reasonable market data range)
    if torch.any(torch.abs(context) > 1e6):
        raise ValueError("Extreme values detected - unrealistic data")
    
    return context
```

#### TiRex Guardian Integration (TGI) — PRODUCTION REQUIREMENT

**CRITICAL**: Direct TiRex calls are PROHIBITED in production. All inference must use protective middleware.

| Guardian Component | Purpose | Implementation | Status |
|-------------------|---------|----------------|--------|
| **Guardian Entry** | Main protective interface | `from sage_forge.guardian import TiRexGuardian` | MANDATORY |
| **Input Shield** | Empirically-validated input protection | `guardian.safe_forecast()` method | MANDATORY |
| **Circuit Shield** | Failure handling & fallbacks | Automatic within guardian | MANDATORY |
| **Output Shield** | Business logic validation | Validates forecast reasonableness | RECOMMENDED |
| **Threat Detection** | Attack pattern recognition | Monitors suspicious inputs | RECOMMENDED |
| **Audit Trail** | Forensic security logging | Complete inference audit | RECOMMENDED |

**Guardian Integration Pattern**:
```python
from sage_forge.guardian import TiRexGuardian

# PRODUCTION PATTERN (Required)
guardian = TiRexGuardian()  # The protective middleware
tirex_quantiles, tirex_mean = guardian.safe_forecast(
    context=validated_context, 
    prediction_length=k
)

# PROHIBITED PATTERN (Security Risk)
# quantiles, mean = model.forecast(context, prediction_length=k)  # Direct calls banned
```

**Guardian System Location**: `src/sage_forge/guardian/` - Complete defensive architecture  
**Empirical Evidence**: `docs/implementation/tirex/empirical-validation/` - Vulnerability analysis and mitigation strategies

#### Data Dictionary & Feature Registry (SCU, DPF, CLC, FTR, NRM)

Note: Column nomenclature follows DSM as the source of truth. Timestamps are UTC and represent the BEGINNING of each candle period. Precision: milliseconds.

| Column                   | Layer     | Type     | Definition                    | Formula / Pseudocode                                                                | Tools        | Lineage          | LeakageGuard |
| ------------------------ | --------- | -------- | ----------------------------- | ----------------------------------------------------------------------------------- | ------------ | ---------------- | ------------ |
| open_time                | RAW       | datetime | Bar open (UTC, ms precision)  | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| open                     | RAW       | float    | —                             | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| high                     | RAW       | float    | —                             | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| low                      | RAW       | float    | —                             | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| close                    | RAW       | float    | —                             | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| volume                   | RAW       | float    | base qty                      | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| close_time               | RAW       | datetime | Bar close (UTC, ms precision) | close_time = open_time + interval - 1ms                                             | DSM (FCP)    | exchange via DSM | —            |
| quote_asset_volume       | RAW       | float    | quote volume                  | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| count                    | RAW       | int      | number of trades              | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| taker_buy_volume         | RAW       | float    | taker buy base asset volume   | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| taker_buy_quote_volume   | RAW       | float    | taker buy quote asset volume  | —                                                                                   | DSM (FCP)    | exchange via DSM | —            |
| ctx_close                | MODEL_IN  | float    | raw close (model context)     | close                                                                               | —            | from close       | roll ≤ t     |
| ctx_norm_close           | MODEL_IN  | float    | normalized close              | zscore(close, win=T)                                                                | numpy/polars | from close       | roll ≤ t     |
| tirex_quantiles[t+1..t+k] | MODEL_OUT | tensor   | full quantile tensor [B,k,9]  | (Q, M) = model.forecast(validate_tirex_input(context), prediction_length=k); use Q | tirex        | from ctx\_\*     | uses ≤ t     |
| tirex_mean_p50[t+1..t+k] | MODEL_OUT | vector   | median forecast path          | M from model.forecast() — same as tirex_quantiles[..., 4]                          | tirex        | from ctx\_\*     | uses ≤ t     |
| tirex_q_p10[t+1..t+k]    | MODEL_OUT | vector   | lower band (p10)              | tirex_quantiles[..., 0] # Extract 0.1 quantile                                     | tirex        | from ctx\_\*     | uses ≤ t     |
| tirex_q_p90[t+1..t+k]    | MODEL_OUT | vector   | upper band (p90)              | tirex_quantiles[..., 8] # Extract 0.9 quantile                                     | tirex        | from ctx\_\*     | uses ≤ t     |
| edge_1                   | DERIVED   | float    | 1-step edge                   | tirex_mean_p50[t+1] - close[t]                                                      | —            | from close/model | roll ≤ t     |
| atr_14                   | DERIVED   | float    | vol proxy                     | ATR(14)                                                                             | ta-lib       | OHLC             | roll ≤ t     |
| ma_20                    | DERIVED   | float    | moving average                | SMA(close, 20)                                                                      | pandas_ta    | from close       | roll ≤ t     |
| rsi_14                   | DERIVED   | float    | momentum oscillator           | RSI(close, 14)                                                                      | pandas_ta    | from close       | roll ≤ t     |
| pos_size                 | DERIVED   | float    | risk scaling                  | risk_budget/(atr_14\*tick_value)                                                    | —            | from atr_14      | roll ≤ t     |
| sig_long                 | SIGNAL    | bool     | entry                         | edge_1>λ·atr_14 && close>MA20                                                       | —            | from derived     | roll ≤ t     |
| tp_lvl                   | SIGNAL    | float    | take-profit                   | tirex_q_p90[t+H]                                                                    | —            | from model       | set at t     |
| sl_lvl                   | SIGNAL    | float    | stop-loss                     | close-μ·atr_14                                                                      | —            | from ATR         | set at t     |

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
