# TiRex Security Vulnerabilities: Empirical Analysis

**Classification**: Security Analysis  
**Severity**: CRITICAL  
**Status**: Confirmed via Empirical Testing  
**Date**: 2025-01-08  

## Executive Summary

Empirical testing reveals **8 confirmed attack vectors** against TiRex API with **zero built-in defenses**. Most critical: **100% NaN input accepted** and produces numeric forecasts, representing complete input validation failure.

**Risk Assessment**: **HIGH** - Production deployment without input validation exposes system to adversarial manipulation.

---

## Vulnerability Matrix

| Attack Vector | Input Pattern | Success Rate | Output Impact | Risk Level |
|---------------|---------------|--------------|---------------|------------|
| **All-NaN Attack** | 100% NaN values | 100% | Normal-looking forecasts | **CRITICAL** |
| **Infinity Injection** | 3% inf values | 100% | Output becomes NaN | **CRITICAL** |
| **Extreme Values** | ±1e10 magnitudes | 100% | Millions in output | **HIGH** |
| **Block NaN Injection** | 20% NaN block | 100% | Slight degradation | **HIGH** |
| **Alternating NaN** | 50% alternating NaN | 100% | Functional output | **HIGH** |
| **Scattered NaN** | 12% random NaN | 100% | Minor impact | **MEDIUM** |
| **Leading NaN** | 30% prefix NaN | 100% | Acceptable output | **MEDIUM** |
| **Trailing NaN** | 30% suffix NaN | 100% | Acceptable output | **MEDIUM** |

**Overall Assessment**: **8/8 attacks successful** - No defensive mechanisms detected

---

## Critical Vulnerability 1: All-NaN Attack

### Attack Description

**Technique**: Submit context tensor with 100% NaN values
**Expected Behavior**: Model should reject invalid input  
**Actual Behavior**: Model accepts and produces numeric forecasts

### Empirical Evidence

```python
# Attack payload
attack_context = torch.full((1, 50), float('nan'))
nan_ratio = torch.isnan(attack_context).float().mean().item()  # 100.0%

# Attack execution  
q, m = model.forecast(attack_context, prediction_length=5)

# Result
✓ Attack succeeded: torch.Size([1, 5, 9])
Output: [-0.093, -0.076, -0.081, -0.085, -0.089]
Output quality: NaN=False, Inf=False
```

### Root Cause Analysis

**Source Location**: `/repos/tirex/src/tirex/models/tirex.py:118`
```python
# Vulnerable code
input_token = torch.nan_to_num(input_token, nan=self.nan_mask_value)  # nan_mask_value = 0
```

**Vulnerability Mechanism**:
1. All NaN values converted to zeros (`nan_mask_value = 0`)
2. Model processes zero-sequence as valid input
3. Produces deterministic output for predictable zero pattern
4. No validation that input was originally NaN

**Security Impact**: 
- **Input Manipulation**: Attackers can force model to process predictable zero sequences
- **Forecast Corruption**: Completely invalid inputs generate seemingly valid forecasts
- **Decision Compromise**: Trading decisions based on corrupted model outputs

---

## Critical Vulnerability 2: Infinity Propagation

### Attack Description

**Technique**: Inject small percentage of infinity values
**Impact**: Entire model output becomes NaN, causing system failure

### Empirical Evidence

```python
# Attack: 3% infinity injection
attack_context = normal_context.clone()
inf_mask = torch.rand_like(attack_context) < 0.03  # 3% positions
attack_context[inf_mask] = float('inf')

# Result
✓ Attack succeeded: torch.Size([1, 5, 9])
Output range: [nan, nan]
Output quality: NaN=True, Inf=False
```

**Attack Amplification**: Only 3% corrupted input → 100% corrupted output

### Root Cause

**Floating Point Arithmetic**: Infinity values propagate through neural network computations
**No Bounds Checking**: Model performs no input sanitization for inf/-inf values

**Security Impact**:
- **Denial of Service**: Small input corruption causes complete model failure
- **System Instability**: NaN outputs cascade through trading system
- **Amplification Attack**: Minimal input corruption → maximal system impact

---

## High-Risk Vulnerability: Extreme Value Injection

### Attack Description

**Technique**: Inject extremely large/small finite values to cause output overflow

### Empirical Evidence

```python
# Attack: Extreme value injection (±1e10)
extreme_mask = torch.rand_like(context) < 0.05  # 5% positions
context[extreme_mask] = torch.randint(0, 2, (...))*2e10 - 1e10

# Result
✓ Attack succeeded: torch.Size([1, 5, 9])
Output range: [-18,990,720.000, -15,808,064.000]
```

**Security Impact**: Model produces forecasts in millions when normal range is units

### Business Logic Attack

**Scenario**: Attacker injects extreme historical prices
**Result**: Model forecasts unrealistic future prices (millions)  
**Impact**: Trading algorithms make catastrophic position sizing decisions

---

## Attack Vector Analysis: NaN Pattern Exploitation  

### Predictable Zero Injection

**Core Vulnerability**: `torch.nan_to_num(input, nan=0)` creates predictable patterns

```python
# Adversarial pattern design
adversarial_input = [
    [float('nan')] * 25,  # First 25 values → all zeros
    [1.0] * 25,           # Last 25 values → preserved  
]

# Model processes as:
processed_input = [
    [0.0] * 25,   # Predictable zero sequence
    [1.0] * 25,   # Normal values
]
```

**Exploitation Potential**:
- **Pattern Recognition**: Attacker learns model response to specific zero patterns
- **Output Prediction**: Can predict model behavior for crafted inputs  
- **Systematic Manipulation**: Design inputs to bias forecasts in desired direction

---

## Defense Evasion Analysis

### Current Protection: **NONE**

**Input Validation**: ❌ None detected  
**Bounds Checking**: ❌ Accepts any finite value  
**NaN Detection**: ❌ Silently converts to zero  
**Range Validation**: ❌ No reasonable bounds enforced  
**Quality Gates**: ❌ No minimum data quality requirements  

### Attack Sophistication Required

**Technical Skill**: **LOW** - Simple tensor manipulation  
**Domain Knowledge**: **LOW** - No understanding of model internals required  
**Access Requirements**: **API ACCESS ONLY** - No special privileges needed  
**Detection Difficulty**: **HIGH** - Attacks produce normal-looking output  

---

## Mitigation Strategy

### Immediate Protections (Pre-Production)

**1. Input Validation Layer**
```python
def validate_context(context: torch.Tensor) -> bool:
    """Mandatory input validation before TiRex inference"""
    
    # Check NaN ratio
    nan_ratio = torch.isnan(context).float().mean()
    if nan_ratio > 0.2:  # Reject >20% NaN
        raise ValueError(f"Excessive NaN ratio: {nan_ratio:.1%}")
    
    # Check infinity values  
    if torch.isinf(context).any():
        raise ValueError("Infinite values detected")
    
    # Check value bounds (reasonable market data range)
    if torch.any(torch.abs(context) > 1e6):
        raise ValueError("Extreme values detected")
    
    # Check minimum data quality
    finite_ratio = torch.isfinite(context).float().mean()
    if finite_ratio < 0.8:  # Require >80% finite values
        raise ValueError(f"Insufficient data quality: {finite_ratio:.1%}")
    
    return True
```

**2. Input Sanitization**
```python
def sanitize_context(context: torch.Tensor) -> torch.Tensor:
    """Clean input before model inference"""
    
    # Clip extreme values to reasonable range
    context = torch.clamp(context, min=-1e6, max=1e6)
    
    # Replace inf/-inf with NaN for explicit handling
    context = torch.where(torch.isinf(context), torch.nan, context)
    
    # Forward-fill NaN values (limited scope)
    # Only if NaN ratio < 20%
    
    return context
```

### Defense-in-Depth Architecture

**Layer 1**: Input Validation (reject malicious inputs)  
**Layer 2**: Input Sanitization (clean borderline inputs)  
**Layer 3**: Output Validation (detect corrupted forecasts)  
**Layer 4**: Monitoring & Alerting (detect attack patterns)  

### Production Monitoring

**Attack Detection Signatures**:
- High NaN ratio in inputs (>10%)
- Infinity values in input streams  
- Extreme value spikes in historical data
- Output forecasts outside reasonable bounds
- Repeated identical zero patterns in processed inputs

---

## Integration Requirements

### N2T-Strategy-IO Contract Updates

**Add Security Section**:
```markdown
#### Input Security & Validation (ISV)

| Control | Specification | Implementation |
|---------|---------------|----------------|
| NaN_Detection | Reject if >20% NaN values | Pre-inference validation |
| Bounds_Checking | Clip to [-1e6, 1e6] range | Input sanitization |
| Infinity_Guard | Convert inf to NaN, then validate | Preprocessing step |
| Quality_Gate | Require >80% finite values | Data quality check |
| Attack_Monitoring | Log suspicious input patterns | Security telemetry |
```

### Strategy Implementation

**Mandatory Validation**:
```python
# In strategy signal generation
def generate_tirex_signals(context_data):
    # REQUIRED: Validate before TiRex inference  
    validate_context(context_data)
    
    # REQUIRED: Sanitize input
    clean_context = sanitize_context(context_data)
    
    # Safe inference
    quantiles, mean = model.forecast(clean_context, ...)
    
    # RECOMMENDED: Validate output bounds
    validate_forecast_outputs(quantiles, mean)
    
    return signals
```

---

## Recommendations

### Immediate Actions (Critical)

1. **Implement Input Validation** - Mandatory for production deployment
2. **Add Security Documentation** - Update all TiRex integration guides  
3. **Create Attack Detection** - Monitor for suspicious input patterns
4. **Establish Security Gates** - No TiRex inference without validation

### Medium-Term Actions

1. **Penetration Testing** - Professional security assessment
2. **Fuzzing Campaign** - Systematic input space exploration  
3. **Anomaly Detection** - ML-based attack pattern recognition
4. **Security Training** - Educate developers on ML vulnerabilities

### Long-Term Actions

1. **Upstream Fix** - Report vulnerabilities to TiRex developers
2. **Alternative Models** - Evaluate more secure forecasting options
3. **Secure ML Pipeline** - Design attack-resistant inference architecture

---

## Conclusion

**Critical Finding**: TiRex has **zero built-in security controls** and accepts completely invalid inputs without validation.

**Production Impact**: Deployment without comprehensive input validation exposes trading systems to **adversarial forecast manipulation**.

**Immediate Requirement**: **Input validation layer is mandatory** - not optional - for any production TiRex deployment.

**Security Posture**: Current TiRex integration represents **unacceptable risk** without proper defensive controls.

---

## References

**Vulnerability Testing**: `../../../tests/validation/tirex-empirical/test_ultraverify.py`  
**Attack Vectors**: Function `verify_nan_injection_vulnerability()`  
**Evidence Data**: `ultraverify_results.json` - "nan_injection_vulnerability" section  
**Source Code**: TiRex repository analysis in comprehensive findings document