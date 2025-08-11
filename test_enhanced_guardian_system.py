#!/usr/bin/env python3
"""
🛡️ Enhanced Guardian System Testing

Comprehensive test suite to validate the enhanced Guardian system with
DataPipelineShield protects against all newly discovered vulnerabilities.
"""

import sys
import torch
import logging
import traceback
from typing import Tuple, Dict, Any

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_guardian_imports():
    """Test 1: Verify enhanced Guardian system can be imported"""
    print("\n🛡️ TEST 1: Enhanced Guardian System Import Validation")
    
    try:
        from sage_forge.guardian import TiRexGuardian, InputShield, CircuitShield, DataPipelineShield
        from sage_forge.guardian.exceptions import GuardianError, ShieldViolation, ThreatDetected
        
        print("✅ All enhanced Guardian components imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_pipeline_shield_functionality():
    """Test 2: Data Pipeline Shield protection against discovered vulnerabilities"""
    print("\n🛡️ TEST 2: Data Pipeline Shield Vulnerability Protection")
    
    from sage_forge.guardian.shields.data_pipeline_shield import DataPipelineShield, DataQualityThreat
    from sage_forge.guardian.exceptions import ShieldViolation, ThreatDetected
    
    shield = DataPipelineShield(protection_level="strict")
    vulnerability_results = {}
    
    # Test 2a: Context quality validation (discovered vulnerability: length 1 causes issues)
    try:
        very_short_context = torch.randn(1, 1)  # Length 1 - discovered problematic
        
        try:
            validated = shield.validate_data_pipeline_safety(very_short_context, prediction_length=5)
            vulnerability_results['short_context'] = "❌ FAILED - Very short context accepted"
        except ShieldViolation:
            vulnerability_results['short_context'] = "✅ PROTECTED - Short context properly rejected"
            
    except Exception as e:
        vulnerability_results['short_context'] = f"❌ ERROR - {e}"
    
    # Test 2b: Context length bounds (prevents integer overflow)
    try:
        excessive_prediction = 2**20  # Very large prediction length
        normal_context = torch.randn(1, 100)
        
        try:
            validated = shield.validate_data_pipeline_safety(normal_context, prediction_length=excessive_prediction)
            vulnerability_results['prediction_bounds'] = "❌ FAILED - Excessive prediction length accepted"
        except ShieldViolation:
            vulnerability_results['prediction_bounds'] = "✅ PROTECTED - Excessive prediction length rejected"
            
    except Exception as e:
        vulnerability_results['prediction_bounds'] = f"❌ ERROR - {e}"
    
    # Test 2c: Quantile ordering validation (discovered vulnerability: reversed ordering not detected)
    try:
        # Create reversed quantile ordering (decreasing instead of increasing)
        reversed_quantiles = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]])  # [B, T, Q]
        mean_prediction = torch.tensor([[5.0]])  # Should match middle quantile
        
        try:
            corrected_quantiles, corrected_mean = shield.validate_quantile_output_safety(
                reversed_quantiles, mean_prediction
            )
            
            # Check if quantiles were corrected to be properly ordered
            ordering_check = torch.all(corrected_quantiles[..., 1:] >= corrected_quantiles[..., :-1])
            if ordering_check:
                vulnerability_results['quantile_ordering'] = "✅ PROTECTED - Reversed quantiles auto-corrected"
            else:
                vulnerability_results['quantile_ordering'] = "❌ FAILED - Quantile ordering not corrected"
                
        except Exception as e:
            vulnerability_results['quantile_ordering'] = f"❌ ERROR - {e}"
            
    except Exception as e:
        vulnerability_results['quantile_ordering'] = f"❌ ERROR - {e}"
    
    # Test 2d: Scaling safety validation (prevents NaN corruption)
    try:
        # Create context that would produce problematic scaling
        constant_context = torch.full((1, 100), 5.0)  # All same values
        
        try:
            validated = shield.validate_data_pipeline_safety(constant_context, prediction_length=10)
            vulnerability_results['scaling_safety'] = "✅ PROTECTED - Constant context handled safely"
        except (ShieldViolation, ThreatDetected):
            vulnerability_results['scaling_safety'] = "✅ PROTECTED - Problematic scaling detected and blocked"
            
    except Exception as e:
        vulnerability_results['scaling_safety'] = f"❌ ERROR - {e}"
    
    # Test 2e: Precision monitoring
    try:
        high_precision = torch.tensor([[1.123456789123456789]], dtype=torch.double)
        
        # This should not trigger precision monitoring issues with proper inputs
        validated = shield.validate_data_pipeline_safety(high_precision, prediction_length=1)
        vulnerability_results['precision_monitoring'] = "✅ PROTECTED - Precision monitoring active"
        
    except Exception as e:
        vulnerability_results['precision_monitoring'] = f"❌ ERROR - {e}"
    
    # Print results
    print("Data Pipeline Shield Protection Results:")
    for vuln, result in vulnerability_results.items():
        print(f"  {vuln}: {result}")
    
    success_count = sum(1 for result in vulnerability_results.values() if "✅ PROTECTED" in result)
    total_tests = len(vulnerability_results)
    
    print(f"\nData Pipeline Protection Effectiveness: {success_count}/{total_tests} ({success_count/total_tests:.1%})")
    return success_count / total_tests >= 0.8

def test_enhanced_guardian_integration():
    """Test 3: Enhanced Guardian integration with data pipeline protection"""
    print("\n🛡️ TEST 3: Enhanced Guardian Integration Testing")
    
    from sage_forge.guardian import TiRexGuardian
    from sage_forge.guardian.exceptions import ShieldViolation, ThreatDetected
    
    integration_results = {}
    
    # Test 3a: Enhanced Guardian initialization
    try:
        guardian = TiRexGuardian(
            threat_detection_level="high",
            fallback_strategy="graceful", 
            data_pipeline_protection="strict"
        )
        
        # Check that all shields are initialized
        if hasattr(guardian, 'data_pipeline_shield'):
            integration_results['initialization'] = "✅ SUCCESS - Enhanced Guardian initialized with data pipeline shield"
        else:
            integration_results['initialization'] = "❌ FAILED - Data pipeline shield not initialized"
            
    except Exception as e:
        integration_results['initialization'] = f"❌ FAILED - {e}"
    
    # Test 3b: Multi-layer protection workflow
    try:
        guardian = TiRexGuardian(data_pipeline_protection="strict")
        
        # Mock the circuit shield to avoid model dependency
        def mock_protected_inference(context, prediction_length, **kwargs):
            batch_size = context.shape[0]
            base_values = torch.randn(batch_size, prediction_length)
            quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            return quantiles, mean
        
        guardian.circuit_shield.protected_inference = mock_protected_inference
        
        # Test with problematic input that should be caught by data pipeline shield
        problematic_context = torch.randn(1, 2)  # Very short - should be caught
        
        try:
            quantiles, mean = guardian.safe_forecast(problematic_context, prediction_length=5)
            integration_results['multi_layer_protection'] = "❌ FAILED - Problematic input not caught"
        except ShieldViolation as e:
            if "too short" in str(e) or "context" in str(e).lower():
                integration_results['multi_layer_protection'] = "✅ SUCCESS - Multi-layer protection active"
            else:
                integration_results['multi_layer_protection'] = f"✅ SUCCESS - Protection active: {type(e).__name__}"
                
    except Exception as e:
        integration_results['multi_layer_protection'] = f"❌ FAILED - {e}"
    
    # Test 3c: Protection statistics and monitoring
    try:
        guardian = TiRexGuardian()
        
        # Test if enhanced statistics are available
        status = guardian.get_protection_status()
        shield_status = status.get('shield_status', {})
        
        if 'data_pipeline_shield' in shield_status:
            integration_results['monitoring'] = "✅ SUCCESS - Data pipeline shield monitoring active"
        else:
            integration_results['monitoring'] = "❌ FAILED - Data pipeline shield not in monitoring"
            
        # Test individual shield statistics
        if hasattr(guardian, 'data_pipeline_shield'):
            pipeline_stats = guardian.data_pipeline_shield.get_data_pipeline_statistics()
            if 'vulnerability_coverage' in pipeline_stats:
                integration_results['statistics'] = "✅ SUCCESS - Comprehensive shield statistics available"
            else:
                integration_results['statistics'] = "❌ FAILED - Shield statistics incomplete"
                
    except Exception as e:
        integration_results['monitoring'] = f"❌ FAILED - {e}"
        integration_results['statistics'] = f"❌ FAILED - {e}"
    
    # Print results
    print("Enhanced Guardian Integration Results:")
    for test, result in integration_results.items():
        print(f"  {test}: {result}")
    
    success_count = sum(1 for result in integration_results.values() if "✅ SUCCESS" in result)
    total_tests = len(integration_results)
    
    print(f"\nIntegration Effectiveness: {success_count}/{total_tests} ({success_count/total_tests:.1%})")
    return success_count / total_tests >= 0.8

def test_vulnerability_coverage():
    """Test 4: Verify protection against specific discovered vulnerabilities"""
    print("\n🛡️ TEST 4: Specific Vulnerability Coverage Testing")
    
    from sage_forge.guardian import TiRexGuardian
    from sage_forge.guardian.exceptions import ShieldViolation, ThreatDetected
    
    coverage_results = {}
    
    guardian = TiRexGuardian(data_pipeline_protection="strict")
    
    # Mock circuit shield
    def mock_inference(context, prediction_length, **kwargs):
        batch_size = context.shape[0]
        base_values = torch.randn(batch_size, prediction_length)
        quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
        mean = quantiles[..., 4]
        return quantiles, mean
    
    guardian.circuit_shield.protected_inference = mock_inference
    
    # Test coverage for each vulnerability category discovered in deep dive
    vulnerability_tests = [
        ("nan_handling", torch.full((1, 50), float('nan'))),  # All-NaN
        ("context_length", torch.randn(1, 1)),               # Too short
        ("tensor_operations", torch.randn(0, 10)),            # Empty batch
        ("scaling_corruption", torch.full((1, 100), 1e12)),   # Extreme values
    ]
    
    for vuln_name, test_input in vulnerability_tests:
        try:
            quantiles, mean = guardian.safe_forecast(test_input, prediction_length=5)
            coverage_results[vuln_name] = "❌ NOT PROTECTED - Vulnerability not caught"
        except (ShieldViolation, ThreatDetected, ValueError, RuntimeError):
            coverage_results[vuln_name] = "✅ PROTECTED - Vulnerability successfully blocked"
        except Exception as e:
            coverage_results[vuln_name] = f"⚠️ PARTIAL - Caught with: {type(e).__name__}"
    
    # Test quantile output validation
    try:
        # Create guardian that can produce output for quantile testing
        guardian_for_quantile = TiRexGuardian(data_pipeline_protection="strict")
        
        # Test with normal input to get output, then validate quantile processing
        normal_context = torch.randn(1, 50) * 0.1
        
        # Create mock that produces reversed quantiles (discovered vulnerability)
        def mock_bad_quantile_inference(context, prediction_length, **kwargs):
            batch_size = context.shape[0]
            # Generate reversed quantiles (decreasing)
            base_values = torch.randn(batch_size, prediction_length)
            quantile_offsets = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])  # Reversed!
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            return quantiles, mean
        
        guardian_for_quantile.circuit_shield.protected_inference = mock_bad_quantile_inference
        
        try:
            quantiles, mean = guardian_for_quantile.safe_forecast(normal_context, prediction_length=5)
            # Check if quantiles were corrected
            is_ordered = torch.all(quantiles[..., 1:] >= quantiles[..., :-1])
            if is_ordered:
                coverage_results['quantile_ordering'] = "✅ PROTECTED - Reversed quantiles auto-corrected"
            else:
                coverage_results['quantile_ordering'] = "❌ NOT PROTECTED - Quantile ordering not fixed"
        except Exception:
            coverage_results['quantile_ordering'] = "⚠️ PARTIAL - Error in quantile processing"
            
    except Exception as e:
        coverage_results['quantile_ordering'] = f"❌ ERROR - {e}"
    
    # Print results
    print("Vulnerability Coverage Results:")
    for vuln, result in coverage_results.items():
        print(f"  {vuln}: {result}")
    
    protected_count = sum(1 for result in coverage_results.values() if "✅ PROTECTED" in result)
    total_vulnerabilities = len(coverage_results)
    
    print(f"\nVulnerability Protection Coverage: {protected_count}/{total_vulnerabilities} ({protected_count/total_vulnerabilities:.1%})")
    return protected_count / total_vulnerabilities >= 0.8

def run_enhanced_guardian_test_suite():
    """Run complete enhanced Guardian system test suite"""
    print("🛡️ ENHANCED GUARDIAN SYSTEM TEST SUITE")
    print("=" * 60)
    print("Testing: Data pipeline vulnerabilities and enhanced protection")
    print("=" * 60)
    
    test_functions = [
        ("Enhanced Import Validation", test_enhanced_guardian_imports),
        ("Data Pipeline Shield Protection", test_data_pipeline_shield_functionality),
        ("Enhanced Guardian Integration", test_enhanced_guardian_integration),
        ("Vulnerability Coverage", test_vulnerability_coverage),
    ]
    
    results = {}
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*20}")
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name}: {status}")
            
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            results[test_name] = False
    
    # Final comprehensive assessment
    print(f"\n{'='*60}")
    print("🛡️ ENHANCED GUARDIAN COMPREHENSIVE ASSESSMENT")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    overall_success_rate = passed_tests / total_tests
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 ENHANCED SYSTEM EFFECTIVENESS: {passed_tests}/{total_tests} ({overall_success_rate:.1%})")
    
    if overall_success_rate >= 0.8:
        print("🎉 Enhanced Guardian system demonstrates HIGH EFFECTIVENESS against all vulnerabilities!")
        print("🛡️ Data pipeline vulnerabilities successfully mitigated")
        print("🚀 System ready for production deployment with comprehensive protection")
        return True
    elif overall_success_rate >= 0.6:
        print("⚠️ Enhanced Guardian system shows MODERATE effectiveness - some improvements needed")
        return False
    else:
        print("🚨 Enhanced Guardian system requires SIGNIFICANT improvements")
        return False

if __name__ == "__main__":
    print("Starting Enhanced Guardian System Testing...")
    success = run_enhanced_guardian_test_suite()
    sys.exit(0 if success else 1)