#!/usr/bin/env python3
"""
TiRex Input Architecture Validation Tests

CRITICAL OBJECTIVE: Definitively determine TiRex's input architecture capabilities:
- Can TiRex process multivariate input (multiple features simultaneously)?
- What are the exact input shape requirements?
- How does PatchedUniTokenizer handle different input dimensions?

These empirical tests will resolve the documentation contradictions with hard evidence.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_1_basic_univariate_input():
    """Test 1: Verify TiRex accepts basic univariate input (EXPECTED: SUCCESS)"""
    print("=" * 60)
    print("TEST 1: Basic Univariate Input (Expected: SUCCESS)")
    print("=" * 60)
    
    try:
        # Import TiRex components
        from repos.nautilus_trader.nautilus_trader.adapters.tirex.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        # Create basic univariate time series
        univariate_data = torch.randn(1, 128)  # [batch_size=1, sequence_length=128]
        
        print(f"Input shape: {univariate_data.shape}")
        print(f"Input dimensions: {univariate_data.ndim}")
        
        # Initialize PatchedUniTokenizer
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        # Test tokenization
        tokenized_tensor, scaler_state = tokenizer.context_input_transform(univariate_data)
        
        print(f"✅ SUCCESS: Univariate input processed")
        print(f"   Tokenized shape: {tokenized_tensor.shape}")
        print(f"   Scaler state keys: {list(scaler_state.keys())}")
        
        return True, "Univariate input successful"
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False, str(e)

def test_2_multivariate_input_attempt():
    """Test 2: Attempt multivariate input with multiple features (HYPOTHESIS: WILL FAIL)"""
    print("\n" + "=" * 60)
    print("TEST 2: Multivariate Input Attempt (Expected: FAILURE)")
    print("=" * 60)
    
    try:
        from repos.nautilus_trader.nautilus_trader.adapters.tirex.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        # Create multivariate time series (OHLCV-like)
        multivariate_data = torch.randn(1, 128, 5)  # [batch_size=1, sequence_length=128, features=5]
        
        print(f"Input shape: {multivariate_data.shape}")
        print(f"Input dimensions: {multivariate_data.ndim}")
        
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        # This should fail if TiRex is truly univariate
        tokenized_tensor, scaler_state = tokenizer.context_input_transform(multivariate_data)
        
        print(f"⚠️  UNEXPECTED SUCCESS: Multivariate input was accepted!")
        print(f"   Tokenized shape: {tokenized_tensor.shape}")
        print(f"   This contradicts univariate hypothesis!")
        
        return True, "Multivariate input unexpectedly successful - TiRex may support multi-features"
        
    except Exception as e:
        print(f"✅ EXPECTED FAILURE: {str(e)}")
        print(f"   This confirms TiRex is univariate-only")
        return False, f"Expected failure confirms univariate: {str(e)}"

def test_3_different_2d_shapes():
    """Test 3: Try different 2D input shapes to understand requirements"""
    print("\n" + "=" * 60)
    print("TEST 3: Different 2D Input Shapes")
    print("=" * 60)
    
    shapes_to_test = [
        (1, 64),    # Small sequence
        (1, 128),   # Standard sequence  
        (1, 256),   # Larger sequence
        (2, 128),   # Batch size 2
        (4, 64),    # Batch size 4
    ]
    
    results = []
    
    try:
        from repos.nautilus_trader.nautilus_trader.adapters.tirex.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        for shape in shapes_to_test:
            try:
                test_data = torch.randn(shape)
                print(f"\nTesting shape: {shape}")
                
                tokenized_tensor, scaler_state = tokenizer.context_input_transform(test_data)
                
                print(f"   ✅ SUCCESS: {shape} → {tokenized_tensor.shape}")
                results.append((shape, True, tokenized_tensor.shape))
                
            except Exception as e:
                print(f"   ❌ FAILED: {shape} → {str(e)}")
                results.append((shape, False, str(e)))
        
        return True, results
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False, f"Import error: {str(e)}"

def test_4_forced_3d_input():
    """Test 4: Force 3D input in different ways to see error messages"""
    print("\n" + "=" * 60)
    print("TEST 4: Force 3D Input - Error Analysis")
    print("=" * 60)
    
    test_cases = [
        ((1, 128, 2), "minimal_multivariate"),
        ((1, 128, 8), "proposed_8_features"),
        ((1, 64, 16), "high_feature_count"),
        ((2, 128, 5), "batch_with_features"),
    ]
    
    try:
        from repos.nautilus_trader.nautilus_trader.adapters.tirex.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        for shape, description in test_cases:
            try:
                test_data = torch.randn(shape)
                print(f"\nTesting {description}: {shape}")
                
                tokenized_tensor, scaler_state = tokenizer.context_input_transform(test_data)
                
                print(f"   ⚠️  UNEXPECTED SUCCESS: {description} worked!")
                print(f"      Input: {shape} → Output: {tokenized_tensor.shape}")
                
            except Exception as e:
                print(f"   ✅ EXPECTED FAILURE: {str(e)}")
                print(f"      This confirms dimension restrictions")
        
        return True, "3D input testing completed"
        
    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        return False, str(e)

def test_5_model_forward_pass():
    """Test 5: Try actual TiRex model forward pass with different inputs"""
    print("\n" + "=" * 60)
    print("TEST 5: TiRex Model Forward Pass")
    print("=" * 60)
    
    try:
        # Try to import and initialize TiRex model
        from repos.nautilus_trader.nautilus_trader.adapters.tirex.tirex import TiRexZero
        
        print("Attempting to load TiRex model...")
        
        # Try to initialize model (this might fail if no checkpoint available)
        try:
            model = TiRexZero()  # This will likely fail without proper config
            print("✅ Model initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Model init failed (expected): {str(e)}")
            print("   This is likely due to missing model checkpoints")
            return False, f"Model initialization failed: {str(e)}"
            
        # Test univariate input
        univariate_data = torch.randn(1, 128)
        print(f"\nTesting univariate forward pass: {univariate_data.shape}")
        
        try:
            output = model(univariate_data)
            print(f"✅ Univariate forward pass successful: {output.shape}")
            
        except Exception as e:
            print(f"❌ Univariate forward failed: {str(e)}")
            
        # Test multivariate input 
        multivariate_data = torch.randn(1, 128, 5)
        print(f"\nTesting multivariate forward pass: {multivariate_data.shape}")
        
        try:
            output = model(multivariate_data)
            print(f"⚠️  UNEXPECTED: Multivariate forward successful: {output.shape}")
            print("    This would contradict univariate hypothesis!")
            
        except Exception as e:
            print(f"✅ EXPECTED: Multivariate forward failed: {str(e)}")
            print("    This confirms univariate architecture")
            
        return True, "Model forward testing completed"
        
    except ImportError as e:
        print(f"❌ Could not import TiRex model: {str(e)}")
        return False, f"Import failed: {str(e)}"

def run_all_tests():
    """Run all validation tests and compile evidence"""
    print("🔬 TIREX INPUT ARCHITECTURE VALIDATION TESTS")
    print("=" * 60)
    print("OBJECTIVE: Resolve univariate vs multivariate capability question")
    print("HYPOTHESIS: TiRex is univariate-only (processes single time series)")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Univariate Input", test_1_basic_univariate_input),
        ("Multivariate Input", test_2_multivariate_input_attempt),
        ("2D Shape Variations", test_3_different_2d_shapes),
        ("3D Input Analysis", test_4_forced_3d_input),
        ("Model Forward Pass", test_5_model_forward_pass),
    ]
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            test_results.append((test_name, success, details))
            
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            test_results.append((test_name, False, f"Test crashed: {str(e)}"))
    
    # Summary report
    print("\n" + "=" * 60)
    print("🔬 EMPIRICAL TEST RESULTS SUMMARY")
    print("=" * 60)
    
    univariate_works = False
    multivariate_works = False
    
    for test_name, success, details in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        
        if "Univariate" in test_name and success:
            univariate_works = True
        if "Multivariate" in test_name and success:
            multivariate_works = True
    
    print("\n" + "=" * 60)
    print("🎯 DEFINITIVE CONCLUSIONS")
    print("=" * 60)
    
    if univariate_works and not multivariate_works:
        print("✅ CONFIRMED: TiRex is UNIVARIATE ONLY")
        print("   - Can process single time series")
        print("   - Cannot process multiple features simultaneously")
        print("   - Documentation correction was ACCURATE")
        
    elif univariate_works and multivariate_works:
        print("⚠️  UNEXPECTED: TiRex appears to support BOTH")
        print("   - Univariate input works")
        print("   - Multivariate input also works")
        print("   - Need to investigate further!")
        
    elif not univariate_works:
        print("❌ CRITICAL: TiRex setup issues")
        print("   - Cannot test properly due to import/setup problems")
        print("   - Need to fix TiRex environment first")
        
    else:
        print("🤔 INCONCLUSIVE: Mixed results")
        print("   - Need additional investigation")
    
    return test_results

if __name__ == "__main__":
    results = run_all_tests()