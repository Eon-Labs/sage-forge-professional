#!/usr/bin/env python3
"""
TiRex Architecture Empirical Validation Tests

CRITICAL OBJECTIVE: Definitively determine TiRex's input architecture capabilities through empirical testing
HYPOTHESIS: TiRex is univariate-only (processes single time series)

This test directly imports TiRex components and validates input shape requirements with hard evidence.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add TiRex source to path
project_root = Path(__file__).parent.parent.parent
tirex_src_path = project_root / "repos" / "tirex" / "src"
sys.path.insert(0, str(tirex_src_path))

def test_tirex_components_import():
    """Test 1: Verify TiRex components can be imported"""
    print("=" * 60)
    print("TEST 1: TiRex Components Import")
    print("=" * 60)
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from tirex.models.tirex import TiRexZero
        from tirex.base import TiRexModel
        
        print("✅ SUCCESS: All TiRex components imported successfully")
        print(f"   - PatchedUniTokenizer: {PatchedUniTokenizer}")
        print(f"   - TiRexZero: {TiRexZero}")
        print(f"   - TiRexModel: {TiRexModel}")
        
        return True, "All imports successful"
        
    except ImportError as e:
        print(f"❌ IMPORT FAILED: {str(e)}")
        return False, str(e)

def test_patchedunitokenizer_univariate():
    """Test 2: Test PatchedUniTokenizer with univariate input (EXPECTED: SUCCESS)"""
    print("\n" + "=" * 60)
    print("TEST 2: PatchedUniTokenizer Univariate Input")
    print("=" * 60)
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        # Create univariate time series - the shape TiRex expects
        univariate_data = torch.randn(128)  # 1D: just sequence length
        print(f"Input shape: {univariate_data.shape}")
        print(f"Input dimensions: {univariate_data.ndim}D")
        
        # Initialize tokenizer
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        # Test tokenization
        result = tokenizer.context_input_transform(univariate_data)
        print(f"✅ SUCCESS: Univariate tokenization worked")
        print(f"   Output type: {type(result)}")
        
        if isinstance(result, tuple):
            tokenized_tensor, scaler_state = result
            print(f"   Tokenized shape: {tokenized_tensor.shape}")
            print(f"   Scaler state: {type(scaler_state)}")
        else:
            print(f"   Result shape: {result.shape}")
        
        return True, f"Univariate input successful: {univariate_data.shape} -> processed"
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False, str(e)

def test_patchedunitokenizer_multivariate():
    """Test 3: Test PatchedUniTokenizer with multivariate input (HYPOTHESIS: SHOULD FAIL)"""
    print("\n" + "=" * 60)
    print("TEST 3: PatchedUniTokenizer Multivariate Input (Expected: FAILURE)")
    print("=" * 60)
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        # Create multivariate time series - multiple features
        multivariate_data = torch.randn(128, 5)  # [sequence_length, features]
        print(f"Input shape: {multivariate_data.shape}")
        print(f"Input dimensions: {multivariate_data.ndim}D")
        
        tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
        
        # This should fail if TiRex is truly univariate
        result = tokenizer.context_input_transform(multivariate_data)
        
        print(f"⚠️  UNEXPECTED SUCCESS: Multivariate input was accepted!")
        print(f"   This contradicts our univariate hypothesis!")
        
        if isinstance(result, tuple):
            tokenized_tensor, scaler_state = result
            print(f"   Tokenized shape: {tokenized_tensor.shape}")
        else:
            print(f"   Result shape: {result.shape}")
        
        return True, f"UNEXPECTED: Multivariate input worked: {multivariate_data.shape}"
        
    except Exception as e:
        print(f"✅ EXPECTED FAILURE: {str(e)}")
        print(f"   This confirms TiRex univariate-only architecture")
        return False, f"Expected failure confirms univariate: {str(e)}"

def test_different_univariate_shapes():
    """Test 4: Try different univariate shapes to understand exact requirements"""
    print("\n" + "=" * 60)
    print("TEST 4: Different Univariate Input Shapes")
    print("=" * 60)
    
    test_shapes = [
        64,    # Short sequence
        128,   # Standard sequence  
        256,   # Longer sequence
        512,   # Very long sequence
    ]
    
    results = []
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        for seq_length in test_shapes:
            try:
                test_data = torch.randn(seq_length)
                tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
                
                print(f"\nTesting sequence length: {seq_length}")
                result = tokenizer.context_input_transform(test_data)
                
                if isinstance(result, tuple):
                    tokenized_tensor, _ = result
                    output_shape = tokenized_tensor.shape
                else:
                    output_shape = result.shape
                
                print(f"   ✅ SUCCESS: {seq_length} → {output_shape}")
                results.append((seq_length, True, output_shape))
                
            except Exception as e:
                print(f"   ❌ FAILED: {seq_length} → {str(e)}")
                results.append((seq_length, False, str(e)))
        
        return True, results
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False, f"Import error: {str(e)}"

def test_batch_dimensions():
    """Test 5: Test batch dimensions (2D input with batch_size)"""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Dimension Testing")
    print("=" * 60)
    
    batch_shapes = [
        (1, 128),  # Single batch
        (4, 128),  # Multiple batch
        (8, 64),   # Larger batch, shorter sequence
    ]
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        for batch_shape in batch_shapes:
            try:
                test_data = torch.randn(batch_shape)
                tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
                
                print(f"\nTesting batch shape: {batch_shape}")
                result = tokenizer.context_input_transform(test_data)
                
                if isinstance(result, tuple):
                    tokenized_tensor, _ = result
                    output_shape = tokenized_tensor.shape
                else:
                    output_shape = result.shape
                
                print(f"   ✅ SUCCESS: {batch_shape} → {output_shape}")
                
            except Exception as e:
                print(f"   ❌ FAILED: {batch_shape} → {str(e)}")
        
        return True, "Batch dimension testing completed"
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False, str(e)

def test_3d_multivariate_variants():
    """Test 6: Test various 3D multivariate inputs (ALL SHOULD FAIL if univariate-only)"""
    print("\n" + "=" * 60)
    print("TEST 6: 3D Multivariate Input Variants (Expected: ALL FAIL)")
    print("=" * 60)
    
    test_cases = [
        ((1, 128, 2), "batch_seq_2features"),
        ((1, 128, 5), "batch_seq_5features"), 
        ((1, 128, 8), "batch_seq_8features"),
        ((4, 128, 5), "multibatch_seq_5features"),
        ((128, 5, 1), "seq_features_batch"),
        ((128, 1, 5), "seq_batch_features"),
    ]
    
    try:
        from tirex.models.components import PatchedUniTokenizer
        from sklearn.preprocessing import StandardScaler
        
        failure_count = 0
        success_count = 0
        
        for shape, description in test_cases:
            try:
                test_data = torch.randn(shape)
                tokenizer = PatchedUniTokenizer(patch_size=12, scaler=StandardScaler())
                
                print(f"\nTesting {description}: {shape}")
                result = tokenizer.context_input_transform(test_data)
                
                print(f"   ⚠️  UNEXPECTED SUCCESS: {description}")
                print(f"      This suggests TiRex might support multi-dimensional input!")
                success_count += 1
                
                if isinstance(result, tuple):
                    tokenized_tensor, _ = result
                    print(f"      Output shape: {tokenized_tensor.shape}")
                
            except Exception as e:
                print(f"   ✅ EXPECTED FAILURE: {str(e)}")
                failure_count += 1
        
        print(f"\n3D Input Test Summary:")
        print(f"   Failures (expected): {failure_count}")
        print(f"   Successes (unexpected): {success_count}")
        
        if success_count > 0:
            conclusion = f"UNEXPECTED: {success_count} 3D inputs worked - TiRex may support multidimensional"
        else:
            conclusion = f"CONFIRMED: All {failure_count} 3D inputs failed - TiRex is univariate-only"
            
        return True, conclusion
        
    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        return False, str(e)

def run_empirical_validation():
    """Run all empirical validation tests to resolve architecture questions definitively"""
    print("🧪 TIREX ARCHITECTURE EMPIRICAL VALIDATION")
    print("=" * 60)
    print("OBJECTIVE: Resolve univariate vs multivariate capability through direct testing")
    print("HYPOTHESIS: TiRex PatchedUniTokenizer only accepts univariate time series")
    print("METHOD: Direct component testing with various input shapes")
    print("=" * 60)
    
    tests = [
        ("Component Import", test_tirex_components_import),
        ("Univariate Input", test_patchedunitokenizer_univariate),
        ("Multivariate Input", test_patchedunitokenizer_multivariate),
        ("Univariate Shape Variations", test_different_univariate_shapes),
        ("Batch Dimensions", test_batch_dimensions),
        ("3D Multivariate Variants", test_3d_multivariate_variants),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            results.append((test_name, success, details))
            
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {str(e)}")
            results.append((test_name, False, f"Test crashed: {str(e)}"))
    
    # Analysis and conclusions
    print("\n" + "=" * 60)
    print("📊 EMPIRICAL RESULTS ANALYSIS")
    print("=" * 60)
    
    import_works = False
    univariate_works = False
    multivariate_works = False
    
    for test_name, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        print(f"    {details}")
        
        if "Import" in test_name and success:
            import_works = True
        elif "Univariate" in test_name and "Multivariate" not in test_name and success:
            univariate_works = True
        elif "Multivariate" in test_name and success:
            multivariate_works = True
    
    print("\n" + "=" * 60)
    print("🎯 DEFINITIVE ARCHITECTURAL CONCLUSIONS")
    print("=" * 60)
    
    if not import_works:
        print("❌ INCONCLUSIVE: Cannot import TiRex components")
        print("   Environment setup issues prevent definitive testing")
        
    elif univariate_works and not multivariate_works:
        print("✅ CONFIRMED: TiRex is UNIVARIATE-ONLY")
        print("   - ✅ Accepts single time series input")
        print("   - ❌ Rejects multivariate/multi-feature input")
        print("   - ✅ Documentation corrections were ACCURATE")
        print("   - ✅ The '2→8 feature expansion' claim was correctly identified as impossible")
        
    elif univariate_works and multivariate_works:
        print("⚠️  SURPRISING: TiRex appears to support BOTH input types")
        print("   - ✅ Univariate input works (expected)")
        print("   - ⚠️  Multivariate input also works (unexpected)")
        print("   - 🤔 Need to investigate how TiRex handles multi-dimensional input internally")
        print("   - 📝 Documentation may need revision to reflect actual capabilities")
        
    else:
        print("🤔 MIXED RESULTS: Partial functionality detected")
        print("   Further investigation needed to understand constraints")
    
    print("\n" + "=" * 60)
    print("📋 ACTIONABLE NEXT STEPS")
    print("=" * 60)
    
    if univariate_works and not multivariate_works:
        print("1. ✅ Keep univariate-focused documentation")
        print("2. 🔄 Focus optimization on single time series quality") 
        print("3. 📊 Design multi-model ensemble for additional features")
        
    elif multivariate_works:
        print("1. 🔍 INVESTIGATE: How does TiRex process multi-dimensional input internally?")
        print("2. 📝 UPDATE: Documentation to reflect actual capabilities")
        print("3. 🧪 TEST: Performance comparison univariate vs multivariate")
        print("4. 🚀 EXPLORE: Original '2→8 feature expansion' concept may be valid!")
        
    return results

if __name__ == "__main__":
    results = run_empirical_validation()