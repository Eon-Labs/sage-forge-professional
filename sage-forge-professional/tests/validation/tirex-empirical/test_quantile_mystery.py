#!/usr/bin/env python3
"""
Deep dive into TiRex quantile behavior - something unexpected happening
"""
import sys
sys.path.append('/home/tca/eon/nt/repos/tirex/src')

import torch
import warnings
warnings.filterwarnings('ignore')

def investigate_quantile_behavior():
    """Figure out what's actually happening with quantiles"""
    print("=" * 60)
    print("INVESTIGATING: TiRex Quantile Behavior Mystery")
    print("=" * 60)
    
    from tirex import load_model
    
    # Load model
    model = load_model("NX-AI/TiRex", device="cuda:0")
    print(f"Model quantiles: {model.quantiles}")
    
    # Test context
    context = torch.randn(1, 50)
    
    # Test 1: What happens with different quantile_levels?
    test_cases = [
        [0.1, 0.5, 0.9],
        [0.5],
        [0.05, 0.95], 
        [0.001, 0.999],
        [0.2, 0.3, 0.7, 0.8],  # Mix of available ones
        [0.15, 0.25, 0.75, 0.85],  # None available, need interpolation
    ]
    
    print("\nTesting different quantile_levels requests:")
    print("-" * 50)
    
    for i, levels in enumerate(test_cases):
        print(f"\nTest {i+1}: quantile_levels = {levels}")
        
        try:
            q, m = model.forecast(context, prediction_length=5, quantile_levels=levels)
            print(f"  Input levels: {levels}")
            print(f"  Output shape: {q.shape}")
            print(f"  Expected shape: [1, 5, {len(levels)}]")
            
            if q.shape[2] == len(levels):
                print(f"  ✓ Correct: Got {len(levels)} quantiles as requested")
                print(f"  Values (t=0): {q[0, 0, :].tolist()}")
            elif q.shape[2] == 9:
                print(f"  ⚠️  BUG: Got 9 quantiles, expected {len(levels)}")
                print(f"  Values (t=0): {q[0, 0, :].tolist()}")
            else:
                print(f"  ❌ Unexpected: Got {q.shape[2]} quantiles")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Test 2: Compare default vs explicit all quantiles
    print(f"\n" + "=" * 50)
    print("COMPARING: Default vs Explicit All Quantiles")
    print("=" * 50)
    
    # Default behavior
    q_default, m_default = model.forecast(context, prediction_length=5)
    print(f"Default call - shape: {q_default.shape}")
    print(f"Default values (t=0): {q_default[0, 0, :].tolist()}")
    
    # Explicit all quantiles
    all_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_explicit, m_explicit = model.forecast(context, prediction_length=5, quantile_levels=all_levels)
    print(f"Explicit all - shape: {q_explicit.shape}")
    print(f"Explicit values (t=0): {q_explicit[0, 0, :].tolist()}")
    
    # Are they identical?
    if torch.allclose(q_default, q_explicit, atol=1e-6):
        print("✓ Default and explicit all quantiles are identical")
    else:
        print("⚠️  Default and explicit all quantiles differ!")
        print(f"Max difference: {(q_default - q_explicit).abs().max():.8f}")
    
    # Test 3: What about just subset?
    print(f"\n" + "=" * 50)
    print("TESTING: Actual Subset Behavior") 
    print("=" * 50)
    
    subset_levels = [0.1, 0.5, 0.9]
    q_subset, m_subset = model.forecast(context, prediction_length=5, quantile_levels=subset_levels)
    
    print(f"Subset request: {subset_levels}")
    print(f"Subset shape: {q_subset.shape}")
    
    if q_subset.shape[2] == 3:
        print("✓ Got 3 quantiles as expected")
        print(f"Subset values: {q_subset[0, 0, :].tolist()}")
        
        # Check if these match the corresponding values from full set
        full_q = q_default[0, 0, :]  # All 9 quantiles
        expected_subset = full_q[[0, 4, 8]]  # Positions for 0.1, 0.5, 0.9
        actual_subset = q_subset[0, 0, :]
        
        print(f"Expected from full: {expected_subset.tolist()}")
        print(f"Actually got:      {actual_subset.tolist()}")
        
        if torch.allclose(expected_subset, actual_subset, atol=1e-6):
            print("✓ Subset correctly extracted from full quantiles")
        else:
            print("⚠️  Subset values don't match expected positions!")
            
    else:
        print(f"⚠️  Got {q_subset.shape[2]} quantiles instead of 3")
    
    return True

def test_interpolation_edge_cases():
    """Test what actually happens with interpolation"""
    print(f"\n" + "=" * 60)
    print("TESTING: Interpolation Edge Cases")
    print("=" * 60)
    
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cuda:0")
    context = torch.randn(1, 30)
    
    # Case 1: Levels that require interpolation
    interp_levels = [0.15, 0.25, 0.75, 0.85]  # Between training levels
    print(f"Case 1: Interpolation required {interp_levels}")
    
    try:
        q_interp, m_interp = model.forecast(context, prediction_length=3, quantile_levels=interp_levels)
        print(f"✓ Interpolation worked: {q_interp.shape}")
        print(f"Values: {q_interp[0, 0, :].tolist()}")
        
        # Check if values are properly ordered
        vals = q_interp[0, 0, :].tolist()
        if vals == sorted(vals):
            print("✓ Interpolated quantiles are properly ordered")
        else:
            print("⚠️  Interpolated quantiles not ordered correctly!")
            
    except Exception as e:
        print(f"✗ Interpolation failed: {e}")
    
    # Case 2: Mix of exact and interpolated
    mixed_levels = [0.1, 0.15, 0.5, 0.85, 0.9]  # Some exact, some interpolated
    print(f"\nCase 2: Mixed exact/interpolated {mixed_levels}")
    
    try:
        q_mixed, m_mixed = model.forecast(context, prediction_length=3, quantile_levels=mixed_levels)
        print(f"✓ Mixed levels worked: {q_mixed.shape}")
        print(f"Values: {q_mixed[0, 0, :].tolist()}")
        
        # The 0.1, 0.5, 0.9 should match exact values
        q_exact, _ = model.forecast(context, prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
        
        # Check positions 0, 2, 4 (0.1, 0.5, 0.9 in mixed array)
        if torch.allclose(q_mixed[0, 0, [0, 2, 4]], q_exact[0, 0, :], atol=1e-6):
            print("✓ Exact quantiles match in mixed array")
        else:
            print("⚠️  Exact quantiles don't match in mixed array")
            
    except Exception as e:
        print(f"✗ Mixed levels failed: {e}")
        
    return True

def main():
    """Investigate quantile mysteries"""
    print("QUANTILE BEHAVIOR INVESTIGATION")
    print("Trying to understand what's really happening...")
    
    try:
        investigate_quantile_behavior()
        test_interpolation_edge_cases()
        
        print(f"\n" + "=" * 60)
        print("INVESTIGATION COMPLETE")
        print("=" * 60)
        print("Key findings will determine how to update documentation...")
        
    except Exception as e:
        print(f"Investigation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)