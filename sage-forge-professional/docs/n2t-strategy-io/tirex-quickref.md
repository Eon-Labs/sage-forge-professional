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

#### Minimal pseudocode (CORRECTED)

```python
from tirex import load_model
import torch

# Load model
model = load_model("NX-AI/TiRex", device="cuda:0")  # for CPU: set env TIREX_NO_CUDA=1 and device="cpu"

# Prepare context with MANDATORY validation
context = torch.tensor([...], dtype=torch.float32).unsqueeze(0)  # [1, T]

# CRITICAL: Input validation (production requirement)
def validate_tirex_input(context):
    if torch.isnan(context).float().mean() > 0.2:
        raise ValueError("Excessive NaN ratio")
    if torch.isinf(context).any():
        raise ValueError("Infinite values detected")
    return context

# Safe inference - quantile_levels parameter ignored
validated_context = validate_tirex_input(context)
q, m = model.forecast(context=validated_context, prediction_length=k)  # Always returns [B, k, 9]

# Extract quantiles from FULL tensor (not selective)
p10, p50, p90 = q[..., 0], q[..., 4], q[..., 8]  # Positions 0, 4, 8 in 9-quantile array
assert torch.allclose(q[..., 4], m)  # Median (position 4) matches mean
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
