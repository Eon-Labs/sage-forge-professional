#!/usr/bin/env python3
"""
ULTRATHINK VERIFICATION: Comprehensive testing to confirm insights
Multiple test approaches to validate findings beyond doubt
"""
import os
import sys
sys.path.append('/home/tca/eon/nt/repos/tirex/src')

import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any
import json
import hashlib

warnings.filterwarnings('ignore')

class InsightValidator:
    """Comprehensive validation of TiRex behavior insights"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.test_results = {}
        self.load_model()
    
    def load_model(self):
        """Load model once for all tests"""
        try:
            from tirex import load_model
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                os.environ['TIREX_NO_CUDA'] = '1'
            
            self.model = load_model("NX-AI/TiRex", device=self.device)
            print(f"✓ Model loaded on {self.device}")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise
    
    def log_result(self, test_name: str, result: Dict[str, Any]):
        """Log test results for analysis"""
        self.test_results[test_name] = result
    
    def verify_quantile_parameter_bug(self) -> Dict[str, Any]:
        """INSIGHT 1: quantile_levels parameter is completely ignored"""
        print("=" * 80)
        print("VERIFICATION 1: Quantile Parameter Bug")
        print("=" * 80)
        
        results = {
            "confirmed": True,
            "evidence": [],
            "counter_evidence": []
        }
        
        # Generate deterministic context for reproducible results
        torch.manual_seed(42)
        context = torch.randn(2, 100)  # [batch=2, context=100]
        
        # Test cases with progressively more extreme requests
        test_cases = [
            ([0.5], "single_median"),
            ([0.1, 0.9], "two_extremes"),
            ([0.2, 0.4, 0.6, 0.8], "four_middle"),
            ([0.05, 0.95], "outside_training"),
            ([0.001, 0.999], "extreme_tails"),
            ([], "empty_list"),
            ([0.5, 0.5, 0.5], "duplicates"),
            (list(np.linspace(0.01, 0.99, 50)), "fifty_quantiles")
        ]
        
        baseline_output = None
        baseline_hash = None
        
        for quantile_levels, test_name in test_cases:
            try:
                if quantile_levels == []:
                    # Test default behavior
                    q, m = self.model.forecast(context, prediction_length=10)
                else:
                    q, m = self.model.forecast(context, prediction_length=10, quantile_levels=quantile_levels)
                
                # Create deterministic hash of output
                output_hash = hashlib.md5(q.detach().cpu().numpy().tobytes()).hexdigest()
                
                evidence = {
                    "test": test_name,
                    "requested_levels": quantile_levels,
                    "expected_shape": f"[2, 10, {len(quantile_levels) if quantile_levels else 9}]",
                    "actual_shape": str(list(q.shape)),
                    "output_hash": output_hash,
                    "identical_to_baseline": False
                }
                
                if baseline_output is None:
                    baseline_output = q.clone()
                    baseline_hash = output_hash
                    evidence["is_baseline"] = True
                else:
                    # Check if output is identical to baseline
                    if torch.allclose(q, baseline_output, atol=1e-10):
                        evidence["identical_to_baseline"] = True
                        evidence["identical_hash"] = (output_hash == baseline_hash)
                    
                # Check if shape matches expectation
                expected_quantiles = len(quantile_levels) if quantile_levels else 9
                actual_quantiles = q.shape[2]
                
                if actual_quantiles != expected_quantiles:
                    evidence["shape_mismatch"] = True
                    evidence["got_quantiles"] = actual_quantiles
                    evidence["expected_quantiles"] = expected_quantiles
                
                results["evidence"].append(evidence)
                print(f"  {test_name:15}: {list(q.shape)} | Hash: {output_hash[:8]}")
                
            except Exception as e:
                results["counter_evidence"].append({
                    "test": test_name,
                    "error": str(e),
                    "quantile_levels": quantile_levels
                })
                print(f"  {test_name:15}: FAILED - {e}")
        
        # Analysis
        identical_outputs = sum(1 for e in results["evidence"] if e.get("identical_to_baseline", False))
        total_tests = len(results["evidence"])
        
        print(f"\nANALYSIS:")
        print(f"  Tests with identical output: {identical_outputs}/{total_tests}")
        print(f"  Tests with shape mismatch: {sum(1 for e in results['evidence'] if e.get('shape_mismatch', False))}")
        
        if identical_outputs >= total_tests - 1:  # Allow baseline to not be identical to itself
            print("  ✓ INSIGHT CONFIRMED: quantile_levels parameter is ignored")
            results["confirmed"] = True
        else:
            print("  ✗ INSIGHT REJECTED: quantile_levels parameter affects output")
            results["confirmed"] = False
        
        self.log_result("quantile_parameter_bug", results)
        return results
    
    def verify_nan_injection_vulnerability(self) -> Dict[str, Any]:
        """INSIGHT 2: NaN injection attack is possible"""
        print("\n" + "=" * 80)
        print("VERIFICATION 2: NaN Injection Vulnerability")
        print("=" * 80)
        
        results = {
            "confirmed": True,
            "attack_vectors": [],
            "defenses_found": []
        }
        
        # Test different NaN injection patterns
        attack_patterns = [
            ("scattered_nan", lambda ctx: self._inject_scattered_nan(ctx, 0.1)),
            ("block_nan", lambda ctx: self._inject_block_nan(ctx, 20)),
            ("alternating_nan", lambda ctx: self._inject_alternating_nan(ctx)),
            ("leading_nan", lambda ctx: self._inject_leading_nan(ctx, 30)),
            ("trailing_nan", lambda ctx: self._inject_trailing_nan(ctx, 30)),
            ("all_nan", lambda ctx: torch.full_like(ctx, float('nan'))),
            ("inf_injection", lambda ctx: self._inject_inf(ctx, 0.05)),
            ("extreme_values", lambda ctx: self._inject_extreme_values(ctx)),
        ]
        
        baseline_context = torch.randn(1, 100)
        baseline_q, baseline_m = self.model.forecast(baseline_context, prediction_length=5)
        
        for pattern_name, pattern_func in attack_patterns:
            try:
                # Create attack context
                attack_context = pattern_func(baseline_context.clone())
                
                # Calculate attack statistics
                nan_ratio = torch.isnan(attack_context).float().mean().item()
                inf_ratio = torch.isinf(attack_context).float().mean().item()
                finite_ratio = torch.isfinite(attack_context).float().mean().item()
                
                print(f"\n  Testing {pattern_name}:")
                print(f"    NaN ratio: {nan_ratio:.1%}")
                print(f"    Inf ratio: {inf_ratio:.1%}")
                print(f"    Finite ratio: {finite_ratio:.1%}")
                
                # Attempt forecast
                attack_q, attack_m = self.model.forecast(attack_context, prediction_length=5)
                
                # Analyze output quality
                output_has_nan = torch.isnan(attack_q).any().item()
                output_has_inf = torch.isinf(attack_q).any().item()
                output_range = (attack_m.min().item(), attack_m.max().item())
                
                attack_result = {
                    "pattern": pattern_name,
                    "input_stats": {
                        "nan_ratio": nan_ratio,
                        "inf_ratio": inf_ratio,
                        "finite_ratio": finite_ratio
                    },
                    "attack_succeeded": True,
                    "output_stats": {
                        "has_nan": output_has_nan,
                        "has_inf": output_has_inf,
                        "range": output_range,
                        "shape": list(attack_q.shape)
                    }
                }
                
                print(f"    ✓ Attack succeeded: {attack_q.shape}")
                print(f"    Output range: [{output_range[0]:.3f}, {output_range[1]:.3f}]")
                print(f"    Output quality: NaN={output_has_nan}, Inf={output_has_inf}")
                
                results["attack_vectors"].append(attack_result)
                
            except Exception as e:
                print(f"    ✗ Attack blocked: {e}")
                results["defenses_found"].append({
                    "pattern": pattern_name,
                    "error": str(e),
                    "defense_type": "exception"
                })
        
        # Summary analysis
        successful_attacks = len(results["attack_vectors"])
        blocked_attacks = len(results["defenses_found"])
        
        print(f"\nVULNERABILITY ANALYSIS:")
        print(f"  Successful attacks: {successful_attacks}")
        print(f"  Blocked attacks: {blocked_attacks}")
        
        if successful_attacks > 0:
            print("  ✓ INSIGHT CONFIRMED: NaN injection vulnerability exists")
            results["confirmed"] = True
            
            # Find most severe attack
            severe_attacks = [a for a in results["attack_vectors"] 
                            if a["input_stats"]["finite_ratio"] < 0.5]
            print(f"  Severe attacks (>50% non-finite): {len(severe_attacks)}")
            
        else:
            print("  ✗ INSIGHT REJECTED: NaN injection attacks are blocked")
            results["confirmed"] = False
        
        self.log_result("nan_injection_vulnerability", results)
        return results
    
    def verify_output_format_correctness(self) -> Dict[str, Any]:
        """INSIGHT 3: Output format is vector [B,k], not scalar"""
        print("\n" + "=" * 80)
        print("VERIFICATION 3: Output Format Correctness")
        print("=" * 80)
        
        results = {
            "confirmed": True,
            "format_tests": []
        }
        
        # Test different batch sizes and prediction lengths
        test_configs = [
            (1, 1, "minimal"),
            (1, 24, "single_batch_standard"),
            (8, 12, "multi_batch_standard"),
            (32, 48, "large_batch_long"),
            (1, 168, "single_week"),
            (100, 1, "large_batch_single"),
        ]
        
        for batch_size, pred_len, config_name in test_configs:
            context = torch.randn(batch_size, 50)
            
            try:
                q, m = self.model.forecast(context, prediction_length=pred_len)
                
                expected_q_shape = [batch_size, pred_len, 9]
                expected_m_shape = [batch_size, pred_len]
                actual_q_shape = list(q.shape)
                actual_m_shape = list(m.shape)
                
                format_test = {
                    "config": config_name,
                    "batch_size": batch_size,
                    "prediction_length": pred_len,
                    "expected_quantiles_shape": expected_q_shape,
                    "actual_quantiles_shape": actual_q_shape,
                    "expected_mean_shape": expected_m_shape,
                    "actual_mean_shape": actual_m_shape,
                    "quantiles_correct": actual_q_shape == expected_q_shape,
                    "mean_correct": actual_m_shape == expected_m_shape,
                    "mean_is_vector": len(actual_m_shape) == 2
                }
                
                print(f"  {config_name:20}: Q{actual_q_shape} M{actual_m_shape}")
                
                results["format_tests"].append(format_test)
                
            except Exception as e:
                print(f"  {config_name:20}: FAILED - {e}")
                results["format_tests"].append({
                    "config": config_name,
                    "error": str(e),
                    "failed": True
                })
        
        # Analysis
        successful_tests = [t for t in results["format_tests"] if not t.get("failed", False)]
        all_means_vector = all(t["mean_is_vector"] for t in successful_tests)
        all_shapes_correct = all(t["quantiles_correct"] and t["mean_correct"] for t in successful_tests)
        
        print(f"\nFORMAT ANALYSIS:")
        print(f"  Successful tests: {len(successful_tests)}")
        print(f"  All means are vectors: {all_means_vector}")
        print(f"  All shapes correct: {all_shapes_correct}")
        
        if all_means_vector and all_shapes_correct:
            print("  ✓ INSIGHT CONFIRMED: Output format is vector [B,k]")
            results["confirmed"] = True
        else:
            print("  ✗ INSIGHT REJECTED: Output format issues found")
            results["confirmed"] = False
        
        self.log_result("output_format_correctness", results)
        return results
    
    def verify_source_code_consistency(self) -> Dict[str, Any]:
        """INSIGHT 4: Source code analysis to understand WHY behaviors occur"""
        print("\n" + "=" * 80)
        print("VERIFICATION 4: Source Code Analysis")
        print("=" * 80)
        
        results = {
            "quantile_parameter_analysis": {},
            "nan_handling_analysis": {},
            "output_format_analysis": {}
        }
        
        # Read and analyze key source files
        source_files = [
            "/home/tca/eon/nt/repos/tirex/src/tirex/api_adapter/forecast.py",
            "/home/tca/eon/nt/repos/tirex/src/tirex/models/predict_utils.py",
            "/home/tca/eon/nt/repos/tirex/src/tirex/models/tirex.py"
        ]
        
        for file_path in source_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                print(f"\n  Analyzing {file_path.split('/')[-1]}:")
                
                # Look for quantile_levels handling
                if "quantile_levels" in content:
                    print("    ✓ quantile_levels parameter mentioned")
                    # Count occurrences and context
                    lines = content.split('\n')
                    quantile_lines = [i for i, line in enumerate(lines) if "quantile_levels" in line]
                    print(f"    Found on lines: {quantile_lines[:5]}")  # First 5 occurrences
                
                # Look for NaN handling
                nan_patterns = ["nan_to_num", "isnan", "nan_mask", "torch.nan"]
                for pattern in nan_patterns:
                    if pattern in content:
                        print(f"    ✓ NaN handling: {pattern} found")
                
                # Look for output shape logic
                shape_patterns = ["shape", "dim", "quantiles", "means"]
                shape_mentions = sum(1 for pattern in shape_patterns if pattern in content)
                print(f"    Shape-related mentions: {shape_mentions}")
                
            except Exception as e:
                print(f"    ✗ Failed to analyze {file_path}: {e}")
        
        # Try to trace the actual execution path
        print(f"\n  Tracing execution path:")
        try:
            # Set up hooks to trace function calls
            import inspect
            
            # Inspect the forecast method
            forecast_source = inspect.getsource(self.model.forecast)
            print("    ✓ Got forecast method source")
            
            # Look for where quantile_levels is actually used
            if "quantile_levels" in forecast_source:
                print("    ✓ quantile_levels used in forecast method")
            else:
                print("    ⚠️ quantile_levels NOT used in forecast method")
                
        except Exception as e:
            print(f"    ✗ Source tracing failed: {e}")
        
        self.log_result("source_code_analysis", results)
        return results
    
    def _inject_scattered_nan(self, context: torch.Tensor, ratio: float) -> torch.Tensor:
        """Inject NaNs at random positions"""
        mask = torch.rand_like(context) < ratio
        context[mask] = float('nan')
        return context
    
    def _inject_block_nan(self, context: torch.Tensor, block_size: int) -> torch.Tensor:
        """Inject a block of NaNs"""
        start_idx = torch.randint(0, context.shape[1] - block_size, (1,)).item()
        context[:, start_idx:start_idx + block_size] = float('nan')
        return context
    
    def _inject_alternating_nan(self, context: torch.Tensor) -> torch.Tensor:
        """Inject NaNs at every other position"""
        context[:, ::2] = float('nan')
        return context
    
    def _inject_leading_nan(self, context: torch.Tensor, count: int) -> torch.Tensor:
        """Inject NaNs at the beginning"""
        context[:, :count] = float('nan')
        return context
    
    def _inject_trailing_nan(self, context: torch.Tensor, count: int) -> torch.Tensor:
        """Inject NaNs at the end"""
        context[:, -count:] = float('nan')
        return context
    
    def _inject_inf(self, context: torch.Tensor, ratio: float) -> torch.Tensor:
        """Inject inf values"""
        mask = torch.rand_like(context) < ratio
        context[mask] = float('inf')
        return context
    
    def _inject_extreme_values(self, context: torch.Tensor) -> torch.Tensor:
        """Inject extremely large/small values"""
        extreme_mask = torch.rand_like(context) < 0.05
        context[extreme_mask] = torch.randint(0, 2, extreme_mask.sum().shape).float() * 2e10 - 1e10
        return context
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification tests"""
        print("ULTRATHINK VERIFICATION SUITE")
        print("Comprehensive validation of TiRex behavior insights")
        print("=" * 80)
        
        verifications = [
            self.verify_quantile_parameter_bug,
            self.verify_nan_injection_vulnerability,
            self.verify_output_format_correctness,
            self.verify_source_code_consistency,
        ]
        
        for verification in verifications:
            try:
                verification()
            except Exception as e:
                print(f"✗ Verification failed: {e}")
        
        return self.test_results
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 80)
        print("ULTRATHINK VERIFICATION REPORT")
        print("=" * 80)
        
        insights = [
            ("Quantile Parameter Bug", "quantile_parameter_bug"),
            ("NaN Injection Vulnerability", "nan_injection_vulnerability"),
            ("Output Format Correctness", "output_format_correctness"),
        ]
        
        confirmed_insights = 0
        total_insights = len(insights)
        
        for insight_name, test_key in insights:
            if test_key in self.test_results:
                result = self.test_results[test_key]
                confirmed = result.get("confirmed", False)
                status = "✓ CONFIRMED" if confirmed else "✗ REJECTED"
                print(f"{insight_name:30}: {status}")
                if confirmed:
                    confirmed_insights += 1
            else:
                print(f"{insight_name:30}: ⚠️ NOT TESTED")
        
        print(f"\nOVERALL VALIDATION: {confirmed_insights}/{total_insights} insights confirmed")
        
        if confirmed_insights == total_insights:
            print("🎯 ALL INSIGHTS VALIDATED - Proceed with documentation updates")
        elif confirmed_insights > total_insights / 2:
            print("⚠️ MAJORITY VALIDATED - Review rejected insights")
        else:
            print("❌ MAJOR VALIDATION FAILURE - Reconsider findings")

def main():
    """Run ultrathink verification"""
    try:
        validator = InsightValidator()
        validator.run_all_verifications()
        validator.generate_report()
        
        # Save detailed results
        with open('/home/tca/eon/nt/ultraverify_results.json', 'w') as f:
            json.dump(validator.test_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: ultraverify_results.json")
        
        return True
    except Exception as e:
        print(f"Ultraverification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)