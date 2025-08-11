### TiRex Quick Reference — For Strategy I/O

- Model: `NX-AI/TiRex` (xLSTM ~35M; decoder-only; sLSTM recurrence; CPM masking).
- API: `model.forecast(context, prediction_length=k, output_type="torch", quantile_levels=IGNORED, batch_size=512, yield_per_batch=False) -> (quantiles[B,k,9], mean_p50[B,k])`.
- Quantiles: **ALWAYS returns 9 quantiles** `[0.1, 0.2, ..., 0.9]` - `quantile_levels` parameter completely ignored.
- Runtime: Linux + NVIDIA GPU (compute capability ≥ 8.0) recommended. CPU/MPS experimental. `TIREX_NO_CUDA=1` to disable CUDA kernels.
- Normalization (NRM): z-score or robust; window = T; clip as documented.
- Alignment (FOC): forecasts at decision time t; horizons labeled t+Δ (Δ=1..k).
- Input types: torch.Tensor, numpy.ndarray, List[Tensor], List[ndarray] (variable lengths padded per batch).
- **Security**: NO input validation - requires mandatory validation wrapper (see contract ISV section).
- Adapters: optional GluonTS/HF dataset adapters if installed; outputs can be torch/numpy/gluonts.

#### Production Usage (GUARDIAN REQUIRED)

```python
from sage_forge.guardian import TiRexGuardian
import torch

# PRODUCTION PATTERN: Always use Guardian (protects against all empirically-validated attacks)
guardian = TiRexGuardian()  # The protective middleware

# Prepare context
context = torch.tensor([...], dtype=torch.float32).unsqueeze(0)  # [1, T]

# Protected inference - Guardian handles ALL security automatically
tirex_quantiles, tirex_mean = guardian.safe_forecast(
    context=context, 
    prediction_length=k
)  # Always returns [B, k, 9] with comprehensive protection

# Extract quantiles from FULL tensor (Guardian ensures safe outputs)
p10, p50, p90 = tirex_quantiles[..., 0], tirex_quantiles[..., 4], tirex_quantiles[..., 8]
assert torch.allclose(tirex_quantiles[..., 4], tirex_mean)  # Median matches mean
```

#### Development/Debug Usage (DIRECT - NOT FOR PRODUCTION)

```python
from tirex import load_model
import torch

# DEVELOPMENT ONLY: Direct model access (bypasses all protection)
model = load_model("NX-AI/TiRex", device="cuda:0")
context = torch.tensor([...], dtype=torch.float32).unsqueeze(0)

# ⚠️ WARNING: No protection against NaN injection, infinity attacks, extreme values
q, m = model.forecast(context=context, prediction_length=k)  # Vulnerable to all attacks
p10, p50, p90 = q[..., 0], q[..., 4], q[..., 8]

# Note: quantile_levels parameter is ignored by TiRex - always returns 9 quantiles
```

#### Columns commonly added (MODEL_OUT)

- `tirex_quantiles[t+1..t+k]` - Full tensor [B, k, 9] with all quantiles
- `tirex_q_p10[t+1..t+k]` - Extracted via `tirex_quantiles[..., 0]`
- `tirex_mean_p50[t+1..t+k]` - From model.forecast() mean output  
- `tirex_q_p90[t+1..t+k]` - Extracted via `tirex_quantiles[..., 8]`

#### Quantile Index Mapping

```python
# TiRex quantile positions in 9-element array
quantile_map = {
    0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4,  # median at position 4
    0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8
}
```

#### Derived examples (UPDATED)

- `edge_1 = tirex_mean_p50[t+1] - close[t]`  # Same formula
- `tp_lvl = tirex_quantiles[t+H, 8]`  # Extract 0.9 quantile directly
- `sl_lvl = close - μ·atr_14`  # Unchanged
