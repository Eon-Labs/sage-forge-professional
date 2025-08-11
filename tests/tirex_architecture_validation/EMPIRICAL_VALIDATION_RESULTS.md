# TiRex Architecture Empirical Validation Results

## Executive Summary

**DEFINITIVE CONCLUSION**: TiRex is **univariate-only** and processes single time series exclusively.

**Key Evidence**: Direct source code analysis reveals `assert data.ndim == 2` in `PatchedUniTokenizer.context_input_transform()`, enforcing exactly 2-dimensional input with no provision for multi-feature processing.

## Critical Discovery

### Source Code Evidence

**File**: `/repos/tirex/src/tirex/models/components.py`  
**Line 106**: `assert data.ndim == 2`

**Context**:
```python
def context_input_transform(self, data: torch.Tensor):
    assert data.ndim == 2  # ← THE SMOKING GUN
    data, scale_state = self.scaler.scale(data)
    return self.patcher(data), {SCALER_STATE: scale_state}
```

### Empirical Validation Results

| Input Shape | Dimensions | Expected | Result | Status |
|-------------|------------|----------|--------|--------|
| `[1, 128]` | 2D | ✅ Accept | ✅ Accepted | **CONFIRMED** |
| `[128]` | 1D | ❌ Reject | ❌ AssertionError | **CONFIRMED** |
| `[1, 128, 5]` | 3D | ❌ Reject | ❌ AssertionError | **CONFIRMED** |

**Test Output**:
```
TEST 1: 2D Input [batch_size, sequence_length]
   Shape: torch.Size([1, 128]), Dimensions: 2
   ✅ SUCCESS: 2D input accepted
   Output shape: torch.Size([1, 11, 12])

TEST 2: 1D Input [sequence_length]  
   Shape: torch.Size([128]), Dimensions: 1
   ✅ EXPECTED ASSERTION: ndim==2 requirement enforced

TEST 3: 3D Input [batch, sequence, features]
   Shape: torch.Size([1, 128, 5]), Dimensions: 3
   ✅ EXPECTED ASSERTION: ndim==2 requirement enforced
```

## Architectural Analysis

### Input Requirements

**Required Shape**: `[batch_size, sequence_length]`
- **Dimension 0**: Batch dimension
- **Dimension 1**: Time sequence dimension  
- **NO Dimension 2**: Feature dimension explicitly prohibited

### Valid Examples
- ✅ `[1, 128]` - Single sequence of 128 timesteps
- ✅ `[4, 256]` - Batch of 4 sequences, 256 timesteps each
- ✅ `[8, 64]` - Batch of 8 sequences, 64 timesteps each

### Invalid Examples  
- ❌ `[128]` - 1D sequence (missing batch dimension)
- ❌ `[1, 128, 5]` - 3D with features (violates `ndim==2`)
- ❌ `[128, 5, 1]` - 3D any arrangement
- ❌ `[1, 128, 2]` - Even minimal 2-feature input rejected

## Documentation Correction Validation

### Original Incorrect Claims ❌
- "2→8 feature expansion"  
- "25% → 100% TiRex utilization"
- "Multi-dimensional OHLC processing"
- "8-feature architecture optimization"

### Corrected Reality ✅
- **Univariate input only**: `[batch_size, sequence_length]`
- **Single value per timestep**: No multi-feature support
- **Architecture constraint**: Hardcoded `assert data.ndim == 2`
- **Optimization focus**: Input quality within univariate constraint

## Impact Assessment

### Documentation Status
**✅ CORRECTIONS WERE ACCURATE**
- All impossible multi-feature claims removed
- Updated to reflect univariate reality
- Corrected optimization strategy
- Fixed architectural diagrams

### Optimization Strategy Revision
**Previous (Impossible)**: 2→8 feature multi-dimensional processing  
**Current (Correct)**: Univariate input quality optimization

**Focus Areas**:
1. **Input Series Selection**: Close vs returns vs VWAP vs typical price
2. **Preprocessing Quality**: Normalization, detrending, outlier handling  
3. **Multi-Model Ensemble**: Separate models for different features
4. **Temporal Optimization**: Sequence length, patch size tuning

## Technical Implications

### TiRex Usage Patterns

**✅ CORRECT Pattern**:
```python
# Single time series input
close_prices = torch.tensor([[100.1, 100.2, 100.3, ...]])  # [1, sequence_length]
predictions = tirex_model(close_prices)
```

**❌ IMPOSSIBLE Pattern**:
```python
# Multi-feature input - WILL FAIL
ohlcv_features = torch.tensor([[[open, high, low, close, volume], ...]])  # [1, seq, 5]
predictions = tirex_model(ohlcv_features)  # AssertionError: ndim == 2
```

### Multi-Feature Strategy
Since TiRex is univariate-only, multi-feature strategies require:
1. **Feature Selection**: Choose single best series (close, returns, etc.)
2. **Multiple Models**: Run separate TiRex instances for different series
3. **Post-Processing**: Combine predictions with additional indicators
4. **Ensemble Methods**: Aggregate multiple univariate forecasts

## Final Architectural Truth

**TiRex Identity**: 
- ✅ Powerful **univariate** time series forecaster
- ❌ NOT a multi-feature ML model

**Capabilities**:
- ✅ Exceptional single time series forecasting with uncertainty quantification
- ✅ Batch processing of multiple univariate series
- ❌ Multi-feature/multi-dimensional input processing

**Integration Strategy**:
- Use TiRex for **price forecasting foundation** 
- Combine with other models/indicators for comprehensive trading signals
- Focus on input quality optimization within architectural constraints

## Test Files Created

1. `test_tirex_input_shapes.py` - Initial validation attempt
2. `test_tirex_architecture_empirical.py` - Comprehensive testing framework  
3. `test_tirex_definitive_proof.py` - **Definitive source code + empirical proof**

**Recommendation**: Run `test_tirex_definitive_proof.py` for complete validation of TiRex's univariate architecture.