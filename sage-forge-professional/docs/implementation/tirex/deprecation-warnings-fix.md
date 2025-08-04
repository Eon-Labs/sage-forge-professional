# TiRex Deprecation Warnings Fix & Prevention

**Date**: August 3, 2025  
**Impact**: HIGH - Clean test output and future PyTorch compatibility  
**Status**: ✅ RESOLVED

---

## 🚨 **Root Cause Analysis**

### **Warning Details**
```
/home/tca/eon/nt/.venv/lib/python3.12/site-packages/xlstm/blocks/slstm/cell.py:543: 
FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. 
Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.

/home/tca/eon/nt/.venv/lib/python3.12/site-packages/xlstm/blocks/slstm/cell.py:568: 
FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. 
Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
```

### **Root Cause**
1. **Third-Party Dependency Issue**: Warnings originate from the `xlstm` library (xLSTM implementation)
2. **PyTorch Version Mismatch**: The `xlstm` library uses deprecated PyTorch 1.x API patterns
3. **PyTorch 2.x Evolution**: PyTorch 2.5+ has deprecated `torch.cuda.amp` in favor of `torch.amp`
4. **TiRex Dependency Chain**: TiRex → xLSTM → deprecated PyTorch API

### **Dependency Analysis**
- **Our Code**: SAGE-Forge Professional (✅ Clean)
- **TiRex Model**: NX-AI/TiRex (✅ Clean - just loads the model)
- **xLSTM Library**: xlstm/blocks/slstm/cell.py (❌ Uses deprecated API)
- **PyTorch**: 2.5.1+cu121 (✅ Latest with deprecation warnings)

---

## 🔧 **Solution Strategy**

### **1. Warning Suppression (Immediate Fix)**
Since this is a third-party library issue that doesn't affect functionality, we suppress the specific warnings in our codebase.

### **2. Future-Proofing (Long-term)**
Document the issue for potential future xLSTM library updates or alternative implementations.

### **3. Testing Impact**
Ensure warning suppression doesn't mask legitimate issues in our own code.

---

## ✅ **Implementation**

### **Fix 1: TiRex Model Wrapper**
Updated `src/sage_forge/models/tirex_model.py` to suppress third-party warnings:

```python
import warnings
import torch

# Suppress specific third-party deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message=".*torch.cuda.amp.custom_bwd.*")
```

### **Fix 2: Test Suite Warning Management**
Updated test files to handle third-party warnings appropriately:

```python
import warnings

# Context manager for clean test output
class SuppressThirdPartyWarnings:
    def __enter__(self):
        warnings.filterwarnings('ignore', category=FutureWarning, 
                               message=".*torch.cuda.amp.*")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.resetwarnings()
```

### **Fix 3: Global Warning Configuration**
Created centralized warning management in `src/sage_forge/core/config.py`:

```python
def configure_warnings():
    """Configure warnings for SAGE-Forge Professional."""
    import warnings
    
    # Suppress known third-party deprecation warnings
    third_party_warnings = [
        ".*torch.cuda.amp.custom_fwd.*",
        ".*torch.cuda.amp.custom_bwd.*",
        ".*TORCH_CUDA_ARCH_LIST.*"  # Additional CUDA warning
    ]
    
    for pattern in third_party_warnings:
        warnings.filterwarnings('ignore', category=FutureWarning, message=pattern)
        warnings.filterwarnings('ignore', category=UserWarning, message=pattern)
```

---

## 🛡️ **Prevention Strategy**

### **1. Dependency Version Pinning**
```toml
# pyproject.toml - Pin problematic dependencies
[tool.uv.sources]
torch = ">=2.4.0,<2.6.0"  # Stable PyTorch version
transformers = ">=4.35.0"  # TiRex-compatible version
```

### **2. Warning Monitoring**
- **Allowed**: Third-party deprecation warnings (documented and suppressed)
- **Blocked**: Any warnings from our SAGE-Forge code
- **Action**: Review and fix any new warnings in our codebase immediately

### **3. Regular Dependency Updates**
- **Quarterly Review**: Check for xLSTM library updates
- **PyTorch Updates**: Test compatibility with new PyTorch versions
- **Alternative Libraries**: Monitor for alternative xLSTM implementations

---

## 📊 **Testing Impact**

### **Before Fix**
```
🔄 Running: Signal Threshold Fix Regression Test
[... 50+ lines of deprecation warnings ...]
✅ PASSED: Signal Threshold Fix Regression Test
```

### **After Fix**
```
🔄 Running: Signal Threshold Fix Regression Test
✅ PASSED: Signal Threshold Fix Regression Test
```

**Result**: Clean test output without functional changes.

---

## 🎯 **Project Memory: Key Learnings**

### **1. Third-Party Warning Management**
- **Never suppress our own warnings** - they indicate real issues
- **Selectively suppress third-party warnings** - when they don't affect functionality
- **Document all suppressed warnings** - for future reference

### **2. PyTorch Deprecation Patterns**
- **PyTorch 2.x Migration**: Many 1.x APIs are deprecated but still functional
- **Common Pattern**: `torch.cuda.amp.*` → `torch.amp.*` with `device_type` parameter
- **Timeline**: Deprecation warnings appear 2-3 versions before removal

### **3. TiRex Integration Considerations**
- **Model Loading**: No warnings from TiRex model itself
- **xLSTM Dependency**: Source of deprecation warnings
- **GPU Operations**: xLSTM uses deprecated CUDA AMP APIs
- **Functionality**: Warnings don't affect model performance

### **4. Best Practices**
```python
# ✅ GOOD: Specific third-party warning suppression
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message=".*torch.cuda.amp.custom_fwd.*")

# ❌ BAD: Blanket warning suppression
warnings.filterwarnings('ignore')  # Masks all warnings including our bugs

# ✅ GOOD: Context-specific suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model = load_tirex_model()  # Only suppress during model loading
```

---

## 🔮 **Future Monitoring**

### **Expected Timeline**
- **PyTorch 2.6+**: May remove deprecated APIs entirely
- **xLSTM Updates**: Library may update to use new PyTorch APIs
- **TiRex Updates**: May switch to different xLSTM implementation

### **Action Items**
1. **Monitor xLSTM releases** for PyTorch 2.x compatibility updates
2. **Test PyTorch upgrades** in development environment first
3. **Review warning suppressions** quarterly
4. **Consider alternative xLSTM libraries** if maintenance issues arise

---

## 📝 **Files Modified**

### **Core Integration**
- `src/sage_forge/models/tirex_model.py`: Warning suppression in model initialization
- `src/sage_forge/core/config.py`: Centralized warning configuration

### **Test Files**
- `tests/regression/test_signal_threshold_fix.py`: Clean test output
- `tests/functional/validate_nt_compliance.py`: Warning management
- `tests/validation/*.py`: Consistent warning handling

### **Documentation**
- `docs/implementation/tirex/deprecation-warnings-fix.md`: This document
- `README.md`: Updated with clean testing experience

---

**Issue Status**: ✅ **RESOLVED**  
**Test Output**: 🧹 **CLEAN**  
**Future-Proofed**: 📋 **DOCUMENTED**  
**Project Memory**: 🧠 **CAPTURED**

---

**Key Learning**: Always distinguish between **our code warnings** (fix immediately) and **third-party library warnings** (document, suppress selectively, monitor for updates).