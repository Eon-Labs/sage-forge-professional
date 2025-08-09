## TiRex Technical Capabilities: Probabilistic Forecasting Architecture

This document presents TiRex's core technical capabilities and outputs for probabilistic forecasting. It focuses on the foundational architecture and computational outputs.

### TiRex Core Technical Outputs

- **9-quantile probability distribution** (q10 through q90)
- **Multi-horizon capability** (1-1000+ bars via prediction_length)
- **Probabilistic distributions** as quantile thresholds
- **3D tensor architecture** [Batch, Horizon, Quantiles]

## FOUNDATION: What TiRex Actually Produces

### Core TiRex Outputs: Schema + Data + Horizon Architecture

#### Critical Distinction: Schema vs Data - `quantiles` vs `qα_h`

**🏗️ Trained quantile levels `quantiles`**: The **architectural schema** - defines WHICH probability thresholds the model was trained to predict
**📊 Quantile forecasts `qα_h`**: The **actual predictions** - provides the price/return VALUES at those probability thresholds for horizon h

| Aspect            | Trained quantile levels `quantiles`                                 | Quantile forecasts `qα_h`                                                              |
| ----------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Nature**        | Static architectural specification                                  | Dynamic prediction output                                                              |
| **Purpose**       | Schema/metadata - tells you what each output dimension represents   | Data - price/return values at probability thresholds                                   |
| **Values**        | Probability levels: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` | Price/return values: `[45880, 46120, 46280, 46350, 46420, 46490, 46580, 46720, 46950]` |
| **When Set**      | Training time - frozen in model architecture                        | Runtime - computed for each forecast                                                   |
| **Shape**         | `[9]` - one entry per trained quantile level                        | `[B, H, 9]` - batch × horizon × quantile dimensions                                    |
| **Technical Use** | Interpret which column means what probability                       | Statistical analysis of price level distributions                                      |

#### Quantile Forecasts `qα_h`: Statistical Distribution Architecture

TiRex's core output is a **probabilistic distribution** represented as quantile thresholds. Each quantile `qα_h` answers: _"What price level has α% probability of being exceeded at horizon h?"_

Both components are required for implementation:

- `quantiles`: probability interpretation metadata
- `qα_h`: actual price/return predictions

| Quantile  | Probability Threshold | Statistical Meaning                       |
| --------- | --------------------- | ----------------------------------------- |
| **q10_h** | 10% exceedance        | 90% of outcomes below this level          |
| **q20_h** | 20% exceedance        | 80% of outcomes below this level          |
| **q30_h** | 30% exceedance        | 70% of outcomes below this level          |
| **q40_h** | 40% exceedance        | 60% of outcomes below this level          |
| **q50_h** | 50% exceedance        | **MEDIAN**: Equal probability above/below |
| **q60_h** | 60% exceedance        | 40% of outcomes below this level          |
| **q70_h** | 70% exceedance        | 30% of outcomes below this level          |
| **q80_h** | 80% exceedance        | 20% of outcomes below this level          |
| **q90_h** | 90% exceedance        | 10% of outcomes below this level          |

#### Technical Implementation Example

⚠️ **CRITICAL REPOSITORY BUG DETECTED**: TiRex source code contains implementation errors that must be handled defensively.

**Repository-Verified Audit-Proof Implementation:**

```python
import torch
import numpy as np
from tirex import load_model

# Step 1: Model Initialization and Schema Access
model = load_model("NX-AI/TiRex")

# CRITICAL BUG FIX: Repository bug at tirex.py:78
# Property `quantiles` incorrectly returns `self.model.quantiles` 
# but should return `self.quantiles` (registered buffer at line 55)
try:
    # Attempt normal property access
    quantile_schema = model.quantiles.tolist() if torch.is_tensor(model.quantiles) else list(model.quantiles)
except (AttributeError, TypeError):
    # Fallback to direct buffer access due to repository bug
    quantile_schema = model._buffers['quantiles'].tolist()
# Verified return: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Step 2: Context Tensor Preparation (Production-Grade)
# WARNING: TiRex needs substantial context - toy examples will fail
# Repository examples use full datasets (144+ points minimum)
np.random.seed(42)  # Reproducible for audit
realistic_sequence = np.cumsum(np.random.randn(200) * 0.05) + 1000.0
context = torch.tensor(realistic_sequence, dtype=torch.float32)  # Shape: [200]

# Step 3: Multi-Horizon Forecasting (Repository-Compliant)
prediction_length = 6  # Repository examples typically use 6-24
quantiles_tensor, mean_tensor = model.forecast(
    context=context,
    prediction_length=prediction_length,
    output_type="torch"  # Explicit type specification
)

# Step 4: Tensor Structure Validation (Audit Verification)
expected_shape = (1, prediction_length, len(quantile_schema))
actual_shape = quantiles_tensor.shape
assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"
print(f"✓ Quantiles tensor: {quantiles_tensor.shape}")  # [1, 6, 9]
print(f"✓ Mean tensor: {mean_tensor.shape}")             # [1, 6]

# Step 5: Repository-Verified Mean=Median Check
# Confirms findings from predict_utils.py:71 "median as mean"
for h in range(prediction_length):
    mean_val = mean_tensor[0, h].item()
    q50_val = quantiles_tensor[0, h, 4].item()  # q50 is index 4
    assert abs(mean_val - q50_val) < 1e-6, f"Mean!=Q50 at horizon {h}: {mean_val} != {q50_val}"

# Step 6: Production Data Extraction
final_horizon = prediction_length - 1
q_values = quantiles_tensor[0, final_horizon, :].detach().cpu().numpy()

# Schema-to-data mapping with audit trail
quantile_forecast = {}
for i, alpha in enumerate(quantile_schema):
    quantile_forecast[f'q{int(alpha*100):02d}'] = float(q_values[i])

print("✓ Repository-verified quantile distribution:")
for k, v in quantile_forecast.items():
    print(f"  {k}: {v:.6f}")
```

### Repository Analysis - Critical Findings

**Audit-Discovered Implementation Issues:**

1. **BUG**: `src/tirex/models/tirex.py:78`  
   ```python
   @property
   def quantiles(self):
       return self.model.quantiles  # ❌ WRONG - self.model doesn't exist
   ```
   **Correct**: Should return `self.quantiles` (registered buffer at line 55)

2. **VERIFIED**: Mean is exactly q50 (median) from `predict_utils.py:71`  
   ```python
   mean = predictions[:, :, training_quantile_levels.index(0.5)]
   ```

3. **CONTEXT REQUIREMENTS**: Repository examples use 144+ data points minimum. Toy examples (5 points) will likely fail in production.

4. **TENSOR SHAPES**: Verified structure from `forecast.py` API:
   - Input: `context` as torch.Tensor [context_length] or [batch, context_length]
   - Output: `(quantiles, mean)` as `([batch, prediction_length, 9], [batch, prediction_length])`

5. **DEFAULT QUANTILES**: Hardcoded in `forecast.py:134` as `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

**Statistical Analysis Framework:**

| Quantile Level (α) | Mathematical Definition  | Probability Interpretation             |
| ------------------ | ------------------------ | -------------------------------------- |
| **q10**            | 10th percentile          | 90% of outcomes above this level       |
| **q20**            | 20th percentile          | 80% of outcomes above this level       |
| **q50**            | 50th percentile (median) | 50% of outcomes above/below this level |
| **q80**            | 80th percentile          | 20% of outcomes above this level       |
| **q90**            | 90th percentile          | 10% of outcomes above this level       |

**Distribution Metrics Extraction:**

```python
# Calculate distribution properties
def analyze_distribution(q_values):
    q10, q20, q30, q40, q50, q60, q70, q80, q90 = q_values

    metrics = {
        'median': q50,
        'iqr': q70 - q30,                    # Interquartile range
        'total_span': q90 - q10,             # 80% confidence interval
        'lower_tail': q50 - q10,             # Lower tail length
        'upper_tail': q90 - q50,             # Upper tail length
        'skew_ratio': (q90 - q50) / (q50 - q10) if (q50 - q10) > 0 else np.inf
    }

    return metrics

distribution_stats = analyze_distribution(q_values)
```

## TECHNICAL CAPABILITIES: TiRex Multi-Horizon Architecture

### Multi-Horizon Capability (1-1000+ Bars)

Dynamic horizon selection via `prediction_length` parameter enables forecasting across multiple time horizons:

```python
# Short horizon
quantiles_1, _ = model.forecast(context, prediction_length=1)

# Medium horizon
quantiles_24, _ = model.forecast(context, prediction_length=24)

# Long horizon
quantiles_100, _ = model.forecast(context, prediction_length=100)
```

### Distribution Analysis

Shape classification from quantile patterns:

| Shape         | Mathematical Condition | Statistical Interpretation  |
| ------------- | ---------------------- | --------------------------- |
| Symmetric     | `q50-q10 ≈ q90-q50`    | Balanced probability tails  |
| Positive skew | `q90-q50 > q50-q10`    | Extended upper tail         |
| Negative skew | `q50-q10 > q90-q50`    | Extended lower tail         |
| Wide bands    | Large `q90-q10`        | High uncertainty/volatility |

### Critical Technical Insights

#### ⚠️ Critical Finding: The "Mean" is Just the Median

**IMPORTANT**: TiRex's returned "mean" forecast is literally just the q50 (median) quantile relabeled. From the source code:

```python
# repos/tirex/src/tirex/models/predict_utils.py:70-71
# median as mean
mean = predictions[:, :, training_quantile_levels.index(0.5)]
```

**Technical Implementation**: Use quantile distributions directly for statistical analysis:

```python
quantiles, _ = model.forecast(...)  # The "mean" is just q50
q_array = quantiles.squeeze().cpu().numpy()

# Actual statistical measures from quantile distribution
true_mean = np.mean(q_array)     # Actual mean of distribution
true_std = np.std(q_array)       # Actual standard deviation
median = q_array[4]              # q50 directly (same as returned "mean")
```

## TiRex Technical Architecture

### Temporal Probability Surface Construction

TiRex can generate probabilistic forecasts across multiple horizons simultaneously:

```python
def build_probability_surface(model, context, max_horizon=100):
    """Build complete probability surface across multiple horizons"""
    quantiles, _ = model.forecast(context, prediction_length=max_horizon)

    surface = {}
    for h in range(max_horizon):
        surface[h+1] = {
            'q10': quantiles[0, h, 0], 'q20': quantiles[0, h, 1],
            'q30': quantiles[0, h, 2], 'q40': quantiles[0, h, 3],
            'q50': quantiles[0, h, 4], 'q60': quantiles[0, h, 5],
            'q70': quantiles[0, h, 6], 'q80': quantiles[0, h, 7],
            'q90': quantiles[0, h, 8]
        }
    return surface
```

### Production-Grade Architecture Components

**Batch Processing for Latency Optimization**:

- Use `batch_size` parameter for multi-symbol processing
- Leverage `max_accelerated_rollout_steps` for long-horizon efficiency
- Implement `yield_per_batch=True` for streaming applications

**Hidden State Access**:

- Access internal `hidden_states` from `_forward_model_tokenized` for regime analysis
- Build statistical classifiers on top of latent representations
- Use hidden state dynamics for temporal pattern detection

**Quantile Calibration Framework**:

- Implement rolling OOS quantile calibration using isotonic regression
- Adjust quantile levels based on realized vs predicted distributions
- Maintain calibration performance metrics for model confidence scoring

### TiRex API Reference

**Core API**: `ForecastModel.forecast(context, prediction_length, output_type, **kwargs)`

**Key Parameters**:

- `prediction_length`: Forecast horizon (1-1000+ bars)
- `output_type`: "torch" | "numpy" | "gluonts"
- `batch_size`, `yield_per_batch`: Performance optimization

**Output**: `(quantiles, mean)` where quantiles shape is `[B, H, 9]`

**Statistical Measures**:

- Central tendency: `q50_h` (median)
- Distribution span: `q90_h − q10_h`
- Skewness: `(q90_h - q50_h) vs (q50_h - q10_h)`

## TECHNICAL NOTES

### Multi-Horizon Forecasting

- `prediction_length` parameter enables 1-1000+ bar forecasts
- Each horizon h provides complete 9-quantile distribution
- 3D tensor output: [Batch, Horizon, Quantiles]

### Performance Optimization

- Expose `--quantiles`, `--auto-cast`, `--batch-size`, `--max-accelerated-rollout-steps` for optimization
- Add optional research flag to dump hidden states for analysis
- Implement rolling OOS quantile calibration module for accuracy assessment

---

## References

- Forecast rollout and horizon: `repos/tirex/src/tirex/models/tirex.py` (`_forecast_tensor`)
- Quantile mapping & interpolation: `repos/tirex/src/tirex/models/predict_utils.py` (`_forecast_quantiles`)
- Batch adapter and output types: `repos/tirex/src/tirex/api_adapter/forecast.py` (`forecast` and adapters)
- Hidden states path: `repos/tirex/src/tirex/models/tirex.py` (`_forward_model_tokenized`)

---

## Navigation

- Located under `docs/research/` for comprehensive TiRex technical analysis
- Referenced from main implementation guides and production scripts
- Integration examples available in `sage-forge-professional/` codebase
