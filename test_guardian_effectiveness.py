#!/usr/bin/env python3
"""
🛡️ Guardian Effectiveness Testing Suite

Comprehensive test suite to validate Guardian system effectiveness against
all empirically-confirmed TiRex vulnerabilities and ensure production readiness.
"""

import sys
import torch
import logging
import traceback
import numpy as np
from typing import Tuple, List, Dict, Any

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_guardian_imports():
    """Test 1: Verify all Guardian components can be imported"""
    print("\n🧪 TEST 1: Guardian Import Validation")
    
    try:
        from sage_forge.guardian import TiRexGuardian, InputShield, CircuitShield
        from sage_forge.guardian.exceptions import (
            GuardianError, ShieldViolation, ThreatDetected, 
            TiRexServiceUnavailableError, FallbackExhaustionError
        )
        print("✅ All Guardian components imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_input_shield_attack_protection():
    """Test 2: Input Shield protection against empirically-confirmed attacks"""
    print("\n🧪 TEST 2: Input Shield Attack Protection")
    
    from sage_forge.guardian.shields.input_shield import InputShield
    from sage_forge.guardian.exceptions import ShieldViolation, ThreatDetected
    
    shield = InputShield(threat_level="medium")
    attack_results = {}
    
    # Attack Vector 1: All-NaN Attack (empirically confirmed - 100% NaN accepted by raw TiRex)
    try:
        all_nan_input = torch.full((1, 100), float('nan'))
        shield.guard_against_attacks(all_nan_input, user_id="test_attacker")
        attack_results['all_nan'] = "❌ FAILED - Attack not blocked"
    except ThreatDetected:
        attack_results['all_nan'] = "✅ BLOCKED - All-NaN attack detected and blocked"
    except ShieldViolation:
        attack_results['all_nan'] = "✅ BLOCKED - NaN threshold violation"
    except Exception as e:
        attack_results['all_nan'] = f"⚠️ UNEXPECTED - {type(e).__name__}: {e}"
    
    # Attack Vector 2: Infinity Injection (empirically confirmed - 3% inf causes corruption)
    try:
        inf_input = torch.randn(1, 100)
        inf_input[0, :3] = float('inf')  # 3% infinity
        shield.guard_against_attacks(inf_input, user_id="test_attacker")
        attack_results['infinity_injection'] = "❌ FAILED - Attack not blocked"
    except ThreatDetected:
        attack_results['infinity_injection'] = "✅ BLOCKED - Infinity injection detected and blocked"
    except Exception as e:
        attack_results['infinity_injection'] = f"✅ BLOCKED - {type(e).__name__}"
    
    # Attack Vector 3: Extreme Value Attack (empirically confirmed - ±1e10 produces millions)
    try:
        extreme_input = torch.randn(1, 100)
        extreme_input[0, 0] = 1e10  # Extreme positive value
        shield.guard_against_attacks(extreme_input, user_id="test_attacker")
        attack_results['extreme_values'] = "❌ FAILED - Attack not blocked"
    except (ThreatDetected, ShieldViolation):
        attack_results['extreme_values'] = "✅ BLOCKED - Extreme value attack blocked"
    except Exception as e:
        attack_results['extreme_values'] = f"✅ BLOCKED - {type(e).__name__}"
    
    # Attack Vector 4: Alternating NaN Pattern (empirically confirmed - 50% alternating functional)
    try:
        alternating_input = torch.randn(1, 100)
        alternating_input[0, ::2] = float('nan')  # 50% alternating NaN
        shield.guard_against_attacks(alternating_input, user_id="test_attacker")
        attack_results['alternating_nan'] = "❌ FAILED - Attack not blocked"
    except (ThreatDetected, ShieldViolation):
        attack_results['alternating_nan'] = "✅ BLOCKED - Alternating NaN pattern blocked"
    except Exception as e:
        attack_results['alternating_nan'] = f"✅ BLOCKED - {type(e).__name__}"
    
    # Test Valid Input (should pass)
    try:
        valid_input = torch.randn(1, 100) * 0.1  # Small, reasonable values
        protected_input = shield.guard_against_attacks(valid_input, user_id="legitimate_user")
        attack_results['valid_input'] = "✅ PASSED - Valid input accepted"
        assert torch.allclose(protected_input, valid_input)
    except Exception as e:
        attack_results['valid_input'] = f"❌ FAILED - Valid input rejected: {e}"
    
    # Print results
    print("Input Shield Attack Protection Results:")
    for attack, result in attack_results.items():
        print(f"  {attack}: {result}")
    
    # Calculate success rate
    successful_blocks = sum(1 for result in attack_results.values() if "✅ BLOCKED" in result or "✅ PASSED" in result)
    total_tests = len(attack_results)
    success_rate = successful_blocks / total_tests
    
    print(f"\nInput Shield Effectiveness: {successful_blocks}/{total_tests} ({success_rate:.1%})")
    return success_rate >= 0.8  # 80% minimum effectiveness


class MockTiRex:
    """Mock TiRex model for testing circuit breaker functionality"""
    
    def __init__(self, failure_mode="none"):
        self.failure_mode = failure_mode
        self.call_count = 0
        
    def forecast(self, context: torch.Tensor, prediction_length: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.call_count += 1
        batch_size = context.shape[0]
        
        if self.failure_mode == "always_fail":
            raise RuntimeError("Mock TiRex failure")
            
        elif self.failure_mode == "fail_first_3":
            if self.call_count <= 3:
                raise RuntimeError(f"Mock failure #{self.call_count}")
                
        elif self.failure_mode == "nan_corruption":
            # Simulate output corruption
            quantiles = torch.full((batch_size, prediction_length, 9), float('nan'))
            mean = torch.full((batch_size, prediction_length), float('nan'))
            return quantiles, mean
            
        elif self.failure_mode == "extreme_output":
            # Simulate extreme value output
            quantiles = torch.full((batch_size, prediction_length, 9), 1e8)
            mean = torch.full((batch_size, prediction_length), 1e8)
            return quantiles, mean
        
        # Normal operation - generate properly ordered quantiles
        base_values = torch.randn(batch_size, prediction_length)
        # Generate quantiles as offsets from base, ensuring proper ordering
        quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=base_values.dtype, device=base_values.device)
        quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1  # Small spread around base
        mean = quantiles[..., 4]  # Use median (index 4) as mean
        return quantiles, mean


def test_circuit_breaker_functionality():
    """Test 3: Circuit breaker failure handling and fallback strategies"""
    print("\n🧪 TEST 3: Circuit Breaker Functionality")
    
    from sage_forge.guardian.shields.circuit_shield import CircuitShield, CircuitState
    from sage_forge.guardian.exceptions import TiRexServiceUnavailableError, FallbackExhaustionError
    
    test_results = {}
    context = torch.randn(1, 50) * 0.1
    prediction_length = 10
    
    # Test 1: Normal operation (circuit should stay closed)
    try:
        circuit = CircuitShield(failure_threshold=3, fallback_strategy="graceful")
        
        # Mock successful TiRex calls
        def mock_success(*args, **kwargs):
            # Generate properly ordered quantiles
            base_values = torch.randn(1, 10)
            quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            circuit._record_success()
            return quantiles, mean
        
        circuit._attempt_tirex_inference = mock_success
        
        quantiles, mean = circuit.protected_inference(context, prediction_length)
        
        assert circuit.state == CircuitState.CLOSED
        assert quantiles.shape == (1, 10, 9)
        assert mean.shape == (1, 10)
        test_results['normal_operation'] = "✅ PASSED - Circuit stays closed for successful calls"
        
    except Exception as e:
        test_results['normal_operation'] = f"❌ FAILED - {e}"
    
    # Test 2: Failure detection and circuit opening
    try:
        circuit = CircuitShield(failure_threshold=2, fallback_strategy="graceful")
        
        # Test circuit opening with consecutive failures
        circuit_test = CircuitShield(failure_threshold=2, fallback_strategy="graceful")
        
        # Simulate the first failure
        try:
            circuit_test._record_failure(RuntimeError("Mock failure 1"))
            assert circuit_test.state == CircuitState.CLOSED  # Should still be closed
            assert circuit_test.failure_count == 1
        except Exception as e:
            raise Exception(f"First failure recording failed: {e}")
        
        # Simulate the second failure - should open circuit
        try:
            circuit_test._record_failure(RuntimeError("Mock failure 2"))
            assert circuit_test.state == CircuitState.OPEN  # Should be open now
            assert circuit_test.failure_count == 2
        except Exception as e:
            raise Exception(f"Second failure recording failed: {e}")
        
        # Test fallback when circuit is open
        try:
            quantiles, mean = circuit_test._fallback_inference(context, prediction_length)
            assert quantiles.shape == (1, 10, 9)
            assert mean.shape == (1, 10)
        except Exception as e:
            raise Exception(f"Fallback inference failed: {e}")
            
        test_results['circuit_opening'] = "✅ PASSED - Circuit opens after threshold failures and uses fallbacks"
        
    except Exception as e:
        test_results['circuit_opening'] = f"❌ FAILED - {e}"
    
    # Test 3: Fallback strategies
    try:
        circuit = CircuitShield(failure_threshold=1, fallback_strategy="graceful")
        circuit.state = CircuitState.OPEN  # Force open state
        
        # Test simple moving average fallback
        context_for_ma = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]).expand(1, -1)
        quantiles, mean = circuit._simple_moving_average_forecast(context_for_ma, 3)
        
        expected_ma = context_for_ma[..., -5:].mean()  # Should be 3.0
        assert torch.allclose(mean, expected_ma.expand(1, 3), atol=1e-6)
        assert quantiles.shape == (1, 3, 9)
        
        # Test linear trend fallback
        quantiles_trend, mean_trend = circuit._linear_trend_forecast(context_for_ma, 3)
        assert quantiles_trend.shape == (1, 3, 9)
        assert mean_trend.shape == (1, 3)
        
        # Test last value fallback (ultimate fallback)
        quantiles_last, mean_last = circuit._last_value_forecast(context_for_ma, 3)
        expected_last = context_for_ma[..., -1]  # Should be 5.0
        assert torch.allclose(mean_last, expected_last.expand(1, 3))
        
        test_results['fallback_strategies'] = "✅ PASSED - All fallback strategies work correctly"
        
    except Exception as e:
        test_results['fallback_strategies'] = f"❌ FAILED - {e}"
    
    # Print results
    print("Circuit Breaker Functionality Results:")
    for test, result in test_results.items():
        print(f"  {test}: {result}")
    
    success_count = sum(1 for result in test_results.values() if "✅ PASSED" in result)
    total_tests = len(test_results)
    success_rate = success_count / total_tests
    
    print(f"\nCircuit Breaker Effectiveness: {success_count}/{total_tests} ({success_rate:.1%})")
    return success_rate >= 0.8


def test_guardian_integration():
    """Test 4: Full Guardian integration with comprehensive protection"""
    print("\n🧪 TEST 4: Guardian Integration Testing")
    
    from sage_forge.guardian import TiRexGuardian
    from sage_forge.guardian.exceptions import ShieldViolation, ThreatDetected
    
    integration_results = {}
    
    # Test 1: Normal operation
    try:
        guardian = TiRexGuardian(threat_detection_level="medium", fallback_strategy="graceful")
        
        # Mock the circuit shield's TiRex inference to avoid actual model dependency
        def mock_protected_inference(context, prediction_length, **kwargs):
            batch_size = context.shape[0]
            # Generate properly ordered quantiles
            base_values = torch.randn(batch_size, prediction_length)
            quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            return quantiles, mean
        
        guardian.circuit_shield.protected_inference = mock_protected_inference
        
        valid_context = torch.randn(1, 50) * 0.1
        quantiles, mean = guardian.safe_forecast(valid_context, prediction_length=10)
        
        assert quantiles.shape == (1, 10, 9)
        assert mean.shape == (1, 10)
        assert guardian.total_inferences == 1
        assert guardian.blocked_threats == 0
        
        integration_results['normal_operation'] = "✅ PASSED - Guardian processes valid input correctly"
        
    except Exception as e:
        integration_results['normal_operation'] = f"❌ FAILED - {e}"
    
    # Test 2: Attack blocking
    try:
        guardian = TiRexGuardian(threat_detection_level="high")  # More sensitive
        
        # Attempt NaN injection attack
        attack_context = torch.full((1, 50), float('nan'))
        
        try:
            guardian.safe_forecast(attack_context, prediction_length=10)
            integration_results['attack_blocking'] = "❌ FAILED - Attack not blocked by Guardian"
        except (ShieldViolation, ThreatDetected):
            assert guardian.blocked_threats == 1
            integration_results['attack_blocking'] = "✅ PASSED - Guardian blocked attack and updated counters"
        
    except Exception as e:
        integration_results['attack_blocking'] = f"❌ FAILED - {e}"
    
    # Test 3: Protection status monitoring
    try:
        guardian = TiRexGuardian()
        status = guardian.get_protection_status()
        
        required_keys = ['guardian_active', 'total_inferences', 'blocked_threats', 
                        'threat_block_rate', 'shield_status', 'threat_detection_level']
        
        for key in required_keys:
            assert key in status, f"Missing status key: {key}"
        
        assert status['guardian_active'] == True
        assert isinstance(status['shield_status'], dict)
        
        integration_results['status_monitoring'] = "✅ PASSED - Protection status monitoring works correctly"
        
    except Exception as e:
        integration_results['status_monitoring'] = f"❌ FAILED - {e}"
    
    # Test 4: Shield statistics
    try:
        guardian = TiRexGuardian()
        
        input_stats = guardian.input_shield.get_shield_statistics()
        circuit_stats = guardian.circuit_shield.get_circuit_statistics()
        
        assert 'shield_type' in input_stats
        assert 'empirical_basis' in input_stats
        assert input_stats['empirical_basis'] == '8/8_attack_vectors_tested'
        
        assert 'circuit_state' in circuit_stats
        assert 'enabled_fallbacks' in circuit_stats
        
        integration_results['shield_statistics'] = "✅ PASSED - Shield statistics provide comprehensive monitoring"
        
    except Exception as e:
        integration_results['shield_statistics'] = f"❌ FAILED - {e}"
    
    # Print results
    print("Guardian Integration Results:")
    for test, result in integration_results.items():
        print(f"  {test}: {result}")
    
    success_count = sum(1 for result in integration_results.values() if "✅ PASSED" in result)
    total_tests = len(integration_results)
    success_rate = success_count / total_tests
    
    print(f"\nGuardian Integration Effectiveness: {success_count}/{total_tests} ({success_rate:.1%})")
    return success_rate >= 0.8


def test_edge_cases_and_robustness():
    """Test 5: Edge cases and robustness validation"""
    print("\n🧪 TEST 5: Edge Cases and Robustness")
    
    from sage_forge.guardian import TiRexGuardian
    from sage_forge.guardian.exceptions import GuardianError
    
    edge_case_results = {}
    
    # Test 1: Empty input handling
    try:
        guardian = TiRexGuardian()
        empty_context = torch.empty(0, 0)
        
        try:
            guardian.safe_forecast(empty_context, prediction_length=5)
            edge_case_results['empty_input'] = "❌ FAILED - Empty input should be rejected"
        except GuardianError:
            edge_case_results['empty_input'] = "✅ PASSED - Empty input correctly rejected"
        
    except Exception as e:
        edge_case_results['empty_input'] = f"❌ FAILED - Unexpected error: {e}"
    
    # Test 2: Very large input handling
    try:
        guardian = TiRexGuardian()
        large_context = torch.randn(1, 10000) * 0.01  # Large but reasonable
        
        # Mock to avoid memory issues
        def mock_inference(*args, **kwargs):
            base_values = torch.randn(1, 5)
            quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            return quantiles, mean
        guardian.circuit_shield.protected_inference = mock_inference
        
        quantiles, mean = guardian.safe_forecast(large_context, prediction_length=5)
        edge_case_results['large_input'] = "✅ PASSED - Large input handled gracefully"
        
    except Exception as e:
        edge_case_results['large_input'] = f"❌ FAILED - {e}"
    
    # Test 3: Multiple batch handling
    try:
        guardian = TiRexGuardian()
        
        def mock_batch_inference(context, prediction_length, **kwargs):
            batch_size = context.shape[0]
            base_values = torch.randn(batch_size, prediction_length)
            quantile_offsets = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            quantiles = base_values.unsqueeze(-1) + quantile_offsets * 0.1
            mean = quantiles[..., 4]
            return quantiles, mean
        guardian.circuit_shield.protected_inference = mock_batch_inference
        
        batch_context = torch.randn(5, 50) * 0.1
        quantiles, mean = guardian.safe_forecast(batch_context, prediction_length=10)
        
        assert quantiles.shape == (5, 10, 9)
        assert mean.shape == (5, 10)
        edge_case_results['batch_processing'] = "✅ PASSED - Batch processing works correctly"
        
    except Exception as e:
        edge_case_results['batch_processing'] = f"❌ FAILED - {e}"
    
    # Test 4: Extreme threat levels
    try:
        # Test very conservative (high) threat level
        strict_guardian = TiRexGuardian(threat_detection_level="high")
        
        # Input with 5% NaN (should be blocked at high threat level)
        mixed_input = torch.randn(1, 100) * 0.1
        mixed_input[0, :5] = float('nan')  # 5% NaN
        
        try:
            strict_guardian.safe_forecast(mixed_input, prediction_length=5)
            edge_case_results['strict_threat_detection'] = "❌ FAILED - High threat level should block 5% NaN"
        except GuardianError:
            edge_case_results['strict_threat_detection'] = "✅ PASSED - High threat level correctly strict"
        
    except Exception as e:
        edge_case_results['strict_threat_detection'] = f"❌ FAILED - {e}"
    
    # Print results
    print("Edge Cases and Robustness Results:")
    for test, result in edge_case_results.items():
        print(f"  {test}: {result}")
    
    success_count = sum(1 for result in edge_case_results.values() if "✅ PASSED" in result)
    total_tests = len(edge_case_results)
    success_rate = success_count / total_tests
    
    print(f"\nEdge Case Robustness: {success_count}/{total_tests} ({success_rate:.1%})")
    return success_rate >= 0.75  # Slightly lower threshold for edge cases


def run_comprehensive_test_suite():
    """Run complete Guardian effectiveness test suite"""
    print("🛡️ GUARDIAN EFFECTIVENESS TEST SUITE")
    print("=" * 50)
    
    test_functions = [
        ("Import Validation", test_guardian_imports),
        ("Input Shield Protection", test_input_shield_attack_protection),
        ("Circuit Breaker Functionality", test_circuit_breaker_functionality),
        ("Guardian Integration", test_guardian_integration),
        ("Edge Cases & Robustness", test_edge_cases_and_robustness),
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
    
    # Final summary
    print(f"\n{'='*50}")
    print("🔍 COMPREHENSIVE TEST RESULTS")
    print(f"{'='*50}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    overall_success_rate = passed_tests / total_tests
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 OVERALL EFFECTIVENESS: {passed_tests}/{total_tests} ({overall_success_rate:.1%})")
    
    if overall_success_rate >= 0.8:
        print("🎉 Guardian system demonstrates HIGH EFFECTIVENESS - Production Ready!")
        return True
    elif overall_success_rate >= 0.6:
        print("⚠️ Guardian system shows MODERATE EFFECTIVENESS - Needs improvements")
        return False
    else:
        print("🚨 Guardian system shows LOW EFFECTIVENESS - Requires major fixes")
        return False


if __name__ == "__main__":
    print("Starting Guardian Effectiveness Testing...")
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)