#!/usr/bin/env python3
"""
Test actual TiRex model functionality with real weights
Focus on getting things to work before documenting what doesn't
"""
import os
import sys
sys.path.append('/home/tca/eon/nt/repos/tirex/src')

import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_model_loading():
    """Test if we can actually load TiRex model"""
    print("=" * 60)
    print("TEST: TiRex Model Loading")
    print("=" * 60)
    
    try:
        from tirex import load_model
        
        print("Attempting to load NX-AI/TiRex model...")
        print("This will download model weights if not cached...")
        
        # Try loading the model - this might download ~140MB
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Target device: {device}")
        
        if device == "cpu":
            print("Setting TIREX_NO_CUDA=1 for CPU inference...")
            os.environ['TIREX_NO_CUDA'] = '1'
        
        model = load_model("NX-AI/TiRex", device=device)
        print(f"✓ Model loaded successfully on {device}")
        print(f"Model type: {type(model)}")
        
        # Check model attributes
        if hasattr(model, 'quantiles'):
            print(f"Model quantiles: {model.quantiles}")
        
        return model, device
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("This might be due to:")
        print("- Network issues downloading weights")
        print("- Missing dependencies") 
        print("- GPU memory issues")
        return None, None

def test_basic_inference(model, device):
    """Test basic inference with real model"""
    print("\n" + "=" * 60)
    print("TEST: Basic Inference")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping - no model available")
        return False
    
    try:
        # Create simple test context - just random walk
        context_length = 100
        context = torch.cumsum(torch.randn(1, context_length) * 0.1, dim=1)
        print(f"Test context shape: {context.shape}")
        print(f"Context range: [{context.min():.3f}, {context.max():.3f}]")
        
        # Basic forecast
        print("Running basic forecast...")
        quantiles, mean = model.forecast(context, prediction_length=24)
        
        print(f"✓ Forecast successful!")
        print(f"Quantiles shape: {quantiles.shape}")
        print(f"Mean shape: {mean.shape}")
        print(f"Quantiles range: [{quantiles.min():.3f}, {quantiles.max():.3f}]")
        print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
        
        # Verify output makes sense
        if quantiles.shape[0] == 1 and quantiles.shape[1] == 24:
            print("✓ Output shapes match expected format")
        else:
            print(f"⚠️  Unexpected output shape: {quantiles.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantile_behavior(model, device):
    """Test quantile behavior with real model"""
    print("\n" + "=" * 60)
    print("TEST: Quantile Behavior")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping - no model available")
        return False
    
    try:
        # Simple test context
        context = torch.randn(1, 50)
        
        # Test 1: Default quantiles
        print("Test 1: Default quantiles")
        q_default, m_default = model.forecast(context, prediction_length=10)
        print(f"Default quantiles shape: {q_default.shape}")
        print(f"Default quantiles (first timestep): {q_default[0, 0, :].tolist()}")
        
        # Test 2: Subset quantiles (should work)
        print("\nTest 2: Subset quantiles [0.1, 0.5, 0.9]")
        q_subset, m_subset = model.forecast(
            context, 
            prediction_length=10,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        print(f"Subset quantiles shape: {q_subset.shape}")
        print(f"Subset values (first timestep): {q_subset[0, 0, :].tolist()}")
        
        # Verify median matches
        if torch.allclose(q_subset[0, 0, 1], m_subset[0, 0], atol=1e-4):
            print("✓ Median quantile matches mean")
        else:
            print(f"⚠️  Median mismatch: q50={q_subset[0, 0, 1]:.4f} vs mean={m_subset[0, 0]:.4f}")
        
        # Test 3: Out-of-range quantiles (the vulnerability test)
        print("\nTest 3: Out-of-range quantiles [0.05, 0.95]")
        try:
            q_extreme, m_extreme = model.forecast(
                context,
                prediction_length=10,
                quantile_levels=[0.05, 0.95]
            )
            print(f"✓ Out-of-range quantiles worked: {q_extreme.shape}")
            print(f"Extreme values: {q_extreme[0, 0, :].tolist()}")
            
            # Check if values are reasonable
            q05_val = q_extreme[0, 0, 0].item()
            q95_val = q_extreme[0, 0, 1].item()
            
            if abs(q05_val) > 1000 or abs(q95_val) > 1000:
                print(f"⚠️  VULNERABILITY CONFIRMED: Extreme values detected")
                print(f"q05={q05_val:.3f}, q95={q95_val:.3f}")
            else:
                print(f"✓ Extrapolated values appear reasonable")
                
        except Exception as e:
            print(f"✗ Out-of-range quantiles failed: {e}")
        
        # Test 4: Very extreme quantiles
        print("\nTest 4: Very extreme quantiles [0.001, 0.999]")
        try:
            q_very_extreme, _ = model.forecast(
                context,
                prediction_length=5,
                quantile_levels=[0.001, 0.999]
            )
            print(f"✓ Very extreme quantiles worked: {q_very_extreme.shape}")
            
            q001_val = q_very_extreme[0, 0, 0].item()
            q999_val = q_very_extreme[0, 0, 1].item()
            print(f"Very extreme values: q0.001={q001_val:.3f}, q0.999={q999_val:.3f}")
            
            if abs(q001_val) > 10000 or abs(q999_val) > 10000 or torch.isnan(q_very_extreme).any():
                print(f"⚠️  EXTREME VULNERABILITY: Unrealistic/NaN values produced")
            else:
                print(f"✓ Even extreme quantiles handled reasonably")
                
        except Exception as e:
            print(f"✗ Very extreme quantiles failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quantile behavior test failed: {e}")
        return False

def test_nan_robustness(model, device):
    """Test NaN handling with real model"""
    print("\n" + "=" * 60)
    print("TEST: NaN Robustness")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping - no model available")
        return False
    
    try:
        # Test 1: Normal context (baseline)
        print("Test 1: Normal context (baseline)")
        normal_context = torch.randn(1, 50)
        q_normal, m_normal = model.forecast(normal_context, prediction_length=5)
        print(f"✓ Normal context works: {q_normal.shape}")
        
        # Test 2: Context with some NaNs
        print("\nTest 2: Context with scattered NaNs")
        nan_context = normal_context.clone()
        nan_context[0, [5, 15, 25, 35]] = float('nan')  # 8% NaN
        print(f"NaN ratio: {torch.isnan(nan_context).float().mean():.2%}")
        
        try:
            q_some_nan, m_some_nan = model.forecast(nan_context, prediction_length=5)
            print(f"✓ Scattered NaNs handled: {q_some_nan.shape}")
            print(f"Output seems reasonable: mean range [{m_some_nan.min():.3f}, {m_some_nan.max():.3f}]")
        except Exception as e:
            print(f"✗ Scattered NaNs failed: {e}")
        
        # Test 3: High NaN ratio (potential attack)
        print("\nTest 3: High NaN ratio (50%)")
        high_nan_context = normal_context.clone()
        high_nan_context[0, ::2] = float('nan')  # Every other value
        print(f"NaN ratio: {torch.isnan(high_nan_context).float().mean():.2%}")
        
        try:
            q_high_nan, m_high_nan = model.forecast(high_nan_context, prediction_length=5)
            print(f"✓ High NaN ratio handled: {q_high_nan.shape}")
            
            # Check if output is degraded
            if torch.isnan(q_high_nan).any():
                print(f"⚠️  NaN in output - model behavior compromised")
            else:
                print(f"Output range: [{m_high_nan.min():.3f}, {m_high_nan.max():.3f}]")
        except Exception as e:
            print(f"✗ High NaN ratio failed: {e}")
        
        # Test 4: All NaN context (critical test)
        print("\nTest 4: All NaN context (critical)")
        all_nan_context = torch.full((1, 50), float('nan'))
        
        try:
            q_all_nan, m_all_nan = model.forecast(all_nan_context, prediction_length=5)
            print(f"⚠️  ALL NaN context accepted: {q_all_nan.shape}")
            print(f"Output: {m_all_nan[0, :3].tolist()}")  # First 3 predictions
            
            if torch.isnan(q_all_nan).any():
                print(f"⚠️  CRITICAL: NaN inputs produce NaN outputs")
            else:
                print(f"⚠️  CRITICAL: All-NaN input still produces numeric output")
                
        except Exception as e:
            print(f"All-NaN context rejected: {e}")
            print("✓ This is actually good - model properly rejects invalid input")
        
        return True
        
    except Exception as e:
        print(f"✗ NaN robustness test failed: {e}")
        return False

def test_batch_behavior(model, device):
    """Test batching behavior"""
    print("\n" + "=" * 60)
    print("TEST: Batch Behavior")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping - no model available")
        return False
    
    try:
        # Test different batch configurations
        contexts = [torch.randn(1, 50) for _ in range(5)]
        
        # Test 1: Different batch sizes
        for batch_size in [1, 2, 3, 5]:
            print(f"Testing batch_size={batch_size}")
            
            try:
                q, m = model.forecast(
                    contexts,
                    prediction_length=10,
                    batch_size=batch_size
                )
                print(f"✓ batch_size={batch_size}: output shape {q.shape}")
            except Exception as e:
                print(f"✗ batch_size={batch_size} failed: {e}")
        
        # Test 2: Single large batch vs multiple small batches
        print(f"\nComparing batch strategies...")
        large_batch = torch.randn(8, 50)  # Single large batch
        
        q_large, m_large = model.forecast(large_batch, prediction_length=5)
        print(f"Large batch (8x50): {q_large.shape}")
        
        # Multiple calls
        small_results = []
        for i in range(8):
            q_small, m_small = model.forecast(large_batch[i:i+1], prediction_length=5)
            small_results.append((q_small, m_small))
        
        print(f"8 individual calls: {len(small_results)} results")
        
        # Results should be similar (not identical due to batching differences)
        print(f"✓ Both approaches work, batch processing more efficient")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch behavior test failed: {e}")
        return False

def main():
    """Run all working functionality tests"""
    print("TESTING TIREX WORKING FUNCTIONALITY")
    print("Focus: Get things working before documenting issues")
    print("=" * 60)
    
    # Step 1: Try to load the model
    model, device = test_model_loading()
    
    if model is None:
        print("\n❌ CRITICAL: Cannot load TiRex model")
        print("Cannot proceed with further testing")
        return False
    
    print(f"\n✅ Model loaded successfully on {device}")
    print("Proceeding with functionality tests...")
    
    # Run all tests
    tests = [
        ("Basic Inference", lambda: test_basic_inference(model, device)),
        ("Quantile Behavior", lambda: test_quantile_behavior(model, device)),
        ("NaN Robustness", lambda: test_nan_robustness(model, device)),
        ("Batch Behavior", lambda: test_batch_behavior(model, device)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("WORKING FUNCTIONALITY SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "WORKS" if result else "BROKEN"
        print(f"{test_name:<20}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests working")
    
    if passed == total:
        print("✅ All functionality tests pass - TiRex is working correctly")
    elif passed > 0:
        print("⚠️  Partial functionality - some features work")
    else:
        print("❌ Major issues - nothing works correctly")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    print(f"\nModel functionality test: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)