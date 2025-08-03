#!/usr/bin/env python3
"""
HOSTILE SENSIBILITY AUDIT - Critical Analysis of SAGE Model Outputs
PURPOSE: Aggressively challenge the "sensible" results claim with real financial logic
APPROACH: Apply actual financial domain knowledge to expose meaningless outputs
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sage.models.alphaforge_wrapper import AlphaForgeWrapper
from sage.models.catch22_wrapper import Catch22Wrapper
from sage.models.tsfresh_wrapper import TSFreshWrapper

console = Console()

class HostileSensibilityAuditor:
    """
    Hostile auditor that applies real financial domain knowledge
    to expose whether model outputs are actually meaningful.
    """
    
    def __init__(self):
        self.console = console
        self.audit_failures = []
        self.critical_findings = []
        
    def create_extreme_market_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Create extreme scenarios that should break poorly designed models"""
        scenarios = {}
        
        # Scenario 1: Flash Crash (10% drop in 5 minutes)
        scenarios['flash_crash'] = self._create_flash_crash()
        
        # Scenario 2: Flat Line (no price movement for hours)
        scenarios['flat_line'] = self._create_flat_line()
        
        # Scenario 3: Random Walk (purely random, no signal)
        scenarios['random_walk'] = self._create_random_walk()
        
        # Scenario 4: Single Spike (one massive outlier)
        scenarios['single_spike'] = self._create_single_spike()
        
        # Scenario 5: Constant Increments (artificial pattern)
        scenarios['constant_increments'] = self._create_constant_increments()
        
        return scenarios
    
    def _create_flash_crash(self) -> pd.DataFrame:
        """Flash crash: 10% drop in 5 minutes, then recovery"""
        n_points = 1440
        base_price = 45000.0
        
        prices = np.full(n_points, base_price)
        
        # Flash crash at minute 300-305
        crash_start, crash_end = 300, 305
        crash_prices = np.linspace(base_price, base_price * 0.9, crash_end - crash_start)
        prices[crash_start:crash_end] = crash_prices
        
        # Gradual recovery
        recovery_end = 400
        recovery_prices = np.linspace(base_price * 0.9, base_price * 0.98, recovery_end - crash_end)
        prices[crash_end:recovery_end] = recovery_prices
        
        # Rest stays at 98% of original
        prices[recovery_end:] = base_price * 0.98
        
        return self._create_ohlcv_from_prices(prices, "Flash Crash")
    
    def _create_flat_line(self) -> pd.DataFrame:
        """Completely flat price for entire period"""
        n_points = 1440
        base_price = 45000.0
        
        # Add tiny noise to avoid division by zero, but essentially flat
        prices = base_price + np.random.normal(0, 0.001, n_points)  # 0.1 cent noise
        
        return self._create_ohlcv_from_prices(prices, "Flat Line")
    
    def _create_random_walk(self) -> pd.DataFrame:
        """Pure random walk with no predictable pattern"""
        n_points = 1440
        base_price = 45000.0
        
        # Pure random walk
        random_returns = np.random.normal(0, 0.001, n_points)
        prices = base_price * np.exp(np.cumsum(random_returns))
        
        return self._create_ohlcv_from_prices(prices, "Random Walk")
    
    def _create_single_spike(self) -> pd.DataFrame:
        """Single massive spike that should be ignored"""
        n_points = 1440
        base_price = 45000.0
        
        prices = np.full(n_points, base_price)
        
        # Single 100x spike at minute 720
        prices[720] = base_price * 100
        
        return self._create_ohlcv_from_prices(prices, "Single Spike")
    
    def _create_constant_increments(self) -> pd.DataFrame:
        """Artificial constant increments (unrealistic but predictable)"""
        n_points = 1440
        base_price = 45000.0
        
        # Constant $1 increment every minute
        prices = base_price + np.arange(n_points)
        
        return self._create_ohlcv_from_prices(prices, "Constant Increments")
    
    def _create_ohlcv_from_prices(self, prices: np.ndarray, label: str) -> pd.DataFrame:
        """Create OHLCV data from price array"""
        timestamps = pd.date_range(start="2024-01-01", periods=len(prices), freq="1min")
        
        # For extreme scenarios, OHLC should be close to price
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        highs = prices * 1.0001  # Tiny spread
        lows = prices * 0.9999
        closes = prices
        volumes = np.random.lognormal(10, 0.1, len(prices))  # Low variance volume
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'scenario': label
        }, index=timestamps)
    
    def audit_alphaforge_sensibility(self, model, scenarios: Dict) -> Dict:
        """Hostile audit of AlphaForge - check if factors make financial sense"""
        audit_results = {
            'model': 'AlphaForge',
            'critical_failures': [],
            'financial_logic_violations': [],
            'detailed_analysis': {}
        }
        
        console.print("🔍 [red]HOSTILE AUDIT: AlphaForge Alpha Factors[/red]")
        
        for scenario_name, data in scenarios.items():
            console.print(f"   🎯 Testing {scenario_name}...")
            
            try:
                factors = model.generate_factors(data, num_factors=5)
                
                # CRITICAL TEST 1: Flat line should produce near-zero alpha factors
                if scenario_name == 'flat_line':
                    self._audit_flat_line_response(factors, audit_results, scenario_name)
                
                # CRITICAL TEST 2: Single spike should be robust (not dominated by outlier)
                elif scenario_name == 'single_spike':
                    self._audit_spike_robustness(factors, data, audit_results, scenario_name)
                
                # CRITICAL TEST 3: Flash crash should show directional signal
                elif scenario_name == 'flash_crash':
                    self._audit_crash_detection(factors, data, audit_results, scenario_name)
                
                # CRITICAL TEST 4: Random walk should show uncertainty/low confidence
                elif scenario_name == 'random_walk':
                    self._audit_random_walk_response(factors, audit_results, scenario_name)
                
                # CRITICAL TEST 5: Constant increments should detect artificial pattern
                elif scenario_name == 'constant_increments':
                    self._audit_artificial_pattern_detection(factors, audit_results, scenario_name)
                
                audit_results['detailed_analysis'][scenario_name] = {
                    'factors_generated': len(factors),
                    'sample_values': {k: self._safe_sample(v) for k, v in factors.items()}
                }
                
            except Exception as e:
                audit_results['critical_failures'].append(f"{scenario_name}: {str(e)}")
        
        return audit_results
    
    def audit_catch22_sensibility(self, model, scenarios: Dict) -> Dict:
        """Hostile audit of catch22 - check if features reflect actual market properties"""
        audit_results = {
            'model': 'catch22',
            'critical_failures': [],
            'financial_logic_violations': [],
            'detailed_analysis': {}
        }
        
        console.print("🔍 [red]HOSTILE AUDIT: catch22 Canonical Features[/red]")
        
        for scenario_name, data in scenarios.items():
            console.print(f"   🎯 Testing {scenario_name}...")
            
            try:
                features = model.extract_features(data['close'])
                
                # CRITICAL TEST 1: Flat line features should reflect low complexity
                if scenario_name == 'flat_line':
                    self._audit_catch22_flat_line(features, audit_results, scenario_name)
                
                # CRITICAL TEST 2: Single spike should show outlier detection
                elif scenario_name == 'single_spike':
                    self._audit_catch22_outlier_detection(features, data, audit_results, scenario_name)
                
                # CRITICAL TEST 3: Random walk should show random characteristics
                elif scenario_name == 'random_walk':
                    self._audit_catch22_randomness(features, audit_results, scenario_name)
                
                # CRITICAL TEST 4: Constant increments should show high predictability
                elif scenario_name == 'constant_increments':
                    self._audit_catch22_predictability(features, audit_results, scenario_name)
                
                audit_results['detailed_analysis'][scenario_name] = {
                    'features_count': len(features),
                    'sample_features': dict(list(features.items())[:5])
                }
                
            except Exception as e:
                audit_results['critical_failures'].append(f"{scenario_name}: {str(e)}")
        
        return audit_results
    
    def audit_tsfresh_sensibility(self, model, scenarios: Dict) -> Dict:
        """Hostile audit of tsfresh - check if features are statistically meaningful"""
        audit_results = {
            'model': 'tsfresh',
            'critical_failures': [],
            'financial_logic_violations': [],
            'detailed_analysis': {}
        }
        
        console.print("🔍 [red]HOSTILE AUDIT: tsfresh Statistical Features[/red]")
        
        for scenario_name, data in scenarios.items():
            console.print(f"   🎯 Testing {scenario_name}...")
            
            try:
                features_df = model.extract_features(data['close'], feature_set='efficient')
                features = features_df.iloc[0].to_dict()
                
                # CRITICAL TEST 1: Statistical features should match basic statistics
                if scenario_name == 'flat_line':
                    self._audit_tsfresh_statistical_consistency(features, data, audit_results, scenario_name)
                
                # CRITICAL TEST 2: Single spike should be handled robustly
                elif scenario_name == 'single_spike':
                    self._audit_tsfresh_outlier_robustness(features, data, audit_results, scenario_name)
                
                # CRITICAL TEST 3: Features should differentiate between scenarios
                self._audit_tsfresh_discriminative_power(features, data, audit_results, scenario_name)
                
                audit_results['detailed_analysis'][scenario_name] = {
                    'features_count': len(features),
                    'sample_features': dict(list(features.items())[:5])
                }
                
            except Exception as e:
                audit_results['critical_failures'].append(f"{scenario_name}: {str(e)}")
        
        return audit_results
    
    def _audit_flat_line_response(self, factors: Dict, audit_results: Dict, scenario: str):
        """Audit AlphaForge response to flat line - should produce minimal signals"""
        violations = []
        
        for factor_name, factor_values in factors.items():
            if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                values = np.array(factor_values)
                
                # Flat line should produce low-variance factors
                factor_std = np.std(values)
                if factor_std > 0.1:  # Arbitrary threshold for "low variance"
                    violations.append(f"Factor {factor_name} shows high variance ({factor_std:.4f}) on flat data")
                
                # Should not have extreme values
                if np.any(np.abs(values) > 10):
                    violations.append(f"Factor {factor_name} has extreme values on flat data: {np.max(np.abs(values)):.4f}")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_spike_robustness(self, factors: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit robustness to single outlier spike"""
        violations = []
        
        # Find the spike location
        spike_idx = np.argmax(data['close'].values)
        spike_value = data['close'].iloc[spike_idx]
        normal_value = data['close'].iloc[0]  # All other values are similar
        
        console.print(f"      🔍 Spike detected: {spike_value:,.2f} vs normal {normal_value:,.2f}")
        
        for factor_name, factor_values in factors.items():
            if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                values = np.array(factor_values)
                
                # Factors should not be completely dominated by the single spike
                # This is a sophisticated test - factors should be robust
                if len(values) > 1:
                    factor_at_spike = values[spike_idx] if spike_idx < len(values) else values[-1]
                    other_factors = np.delete(values, spike_idx) if spike_idx < len(values) else values[:-1]
                    
                    if len(other_factors) > 0:
                        typical_factor = np.median(other_factors)
                        
                        # If spike factor is more than 100x typical, it's probably not robust
                        if abs(factor_at_spike) > 100 * abs(typical_factor) and abs(typical_factor) > 1e-6:
                            violations.append(f"Factor {factor_name} not robust to outlier: spike={factor_at_spike:.4f}, typical={typical_factor:.4f}")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_crash_detection(self, factors: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit if flash crash is properly detected"""
        violations = []
        
        # Flash crash should produce some directional signal
        crash_detected = False
        
        for factor_name, factor_values in factors.items():
            if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                values = np.array(factor_values)
                
                # Look for significant negative signal during crash period (minutes 300-305)
                if len(values) >= 305:
                    crash_period_factors = values[300:305]
                    pre_crash_factors = values[250:300]
                    
                    if len(pre_crash_factors) > 0 and len(crash_period_factors) > 0:
                        crash_signal = np.mean(crash_period_factors)
                        normal_signal = np.mean(pre_crash_factors)
                        
                        # Should show some response to 10% price drop
                        signal_change = abs(crash_signal - normal_signal)
                        if signal_change > 0.01:  # Some threshold for meaningful response
                            crash_detected = True
                            break
        
        if not crash_detected:
            violations.append("No factors detected the flash crash (10% drop in 5 minutes)")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_random_walk_response(self, factors: Dict, audit_results: Dict, scenario: str):
        """Audit response to pure random walk"""
        violations = []
        
        # Random walk should not produce highly confident directional signals
        for factor_name, factor_values in factors.items():
            if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                values = np.array(factor_values)
                
                # Check if factors show unrealistic confidence in random data
                if len(values) > 100:
                    # Look for persistent directional bias
                    positive_signals = np.sum(values > 0.1)  # Strong positive
                    negative_signals = np.sum(values < -0.1)  # Strong negative
                    total_signals = len(values)
                    
                    # More than 80% in one direction is suspicious for random data
                    if positive_signals > 0.8 * total_signals or negative_signals > 0.8 * total_signals:
                        violations.append(f"Factor {factor_name} shows unrealistic directional bias on random data")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_artificial_pattern_detection(self, factors: Dict, audit_results: Dict, scenario: str):
        """Audit detection of artificial constant increment pattern"""
        violations = []
        
        # Constant increments should be easily detected by any reasonable alpha model
        pattern_detected = False
        
        for factor_name, factor_values in factors.items():
            if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                values = np.array(factor_values)
                
                # Should show strong trend signal for constant increments
                if len(values) > 10:
                    trend_strength = abs(np.polyfit(range(len(values)), values, 1)[0])
                    if trend_strength > 0.01:  # Some threshold for trend detection
                        pattern_detected = True
                        break
        
        if not pattern_detected:
            violations.append("Failed to detect obvious artificial pattern (constant increments)")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_catch22_flat_line(self, features: Dict, audit_results: Dict, scenario: str):
        """Audit catch22 response to flat line"""
        violations = []
        
        # Flat data should have low complexity measures
        complexity_features = [k for k in features.keys() if any(word in k.lower() for word in ['entropy', 'complexity', 'embed'])]
        
        for feature_name in complexity_features:
            if feature_name in features:
                value = features[feature_name]
                if not np.isfinite(value):
                    violations.append(f"Complexity feature {feature_name} is not finite on flat data")
                elif value > 1.0:  # Expect low complexity
                    violations.append(f"Complexity feature {feature_name}={value:.4f} unexpectedly high on flat data")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_catch22_outlier_detection(self, features: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit catch22 outlier detection"""
        violations = []
        
        # Should detect the massive outlier
        outlier_features = [k for k in features.keys() if any(word in k.lower() for word in ['outlier', 'extreme', 'max', 'range'])]
        
        outlier_detected = False
        for feature_name in outlier_features:
            if feature_name in features:
                value = features[feature_name]
                if np.isfinite(value) and abs(value) > 10:  # Some threshold for outlier detection
                    outlier_detected = True
                    break
        
        if not outlier_detected:
            violations.append("catch22 failed to detect massive outlier (100x spike)")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_catch22_randomness(self, features: Dict, audit_results: Dict, scenario: str):
        """Audit catch22 randomness detection"""
        violations = []
        
        # Random walk should show appropriate randomness measures
        randomness_features = [k for k in features.keys() if any(word in k.lower() for word in ['random', 'entropy', 'predict'])]
        
        for feature_name in randomness_features:
            if feature_name in features:
                value = features[feature_name]
                if not np.isfinite(value):
                    violations.append(f"Randomness feature {feature_name} is not finite on random data")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_catch22_predictability(self, features: Dict, audit_results: Dict, scenario: str):
        """Audit catch22 predictability detection"""
        violations = []
        
        # Constant increments should show high predictability
        predictability_features = [k for k in features.keys() if any(word in k.lower() for word in ['predict', 'linear', 'trend'])]
        
        high_predictability_detected = False
        for feature_name in predictability_features:
            if feature_name in features:
                value = features[feature_name]
                if np.isfinite(value) and abs(value) > 0.1:  # Some threshold
                    high_predictability_detected = True
                    break
        
        if not high_predictability_detected:
            violations.append("catch22 failed to detect high predictability in constant increment pattern")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_tsfresh_statistical_consistency(self, features: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit tsfresh statistical consistency"""
        violations = []
        
        # Check if basic statistics match
        actual_mean = data['close'].mean()
        actual_std = data['close'].std()
        actual_min = data['close'].min()
        actual_max = data['close'].max()
        
        # Look for statistical features
        for feature_name, feature_value in features.items():
            if 'mean' in feature_name.lower():
                if abs(feature_value - actual_mean) > 0.01 * actual_mean:  # 1% tolerance
                    violations.append(f"Mean feature {feature_name}={feature_value:.4f} doesn't match actual mean {actual_mean:.4f}")
            
            elif 'std' in feature_name.lower() or 'standard_deviation' in feature_name.lower():
                if actual_std < 1.0 and feature_value > 10:  # Flat data shouldn't have high std features
                    violations.append(f"Std feature {feature_name}={feature_value:.4f} unrealistic for flat data (actual std={actual_std:.4f})")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_tsfresh_outlier_robustness(self, features: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit tsfresh robustness to outliers"""
        violations = []
        
        # Single spike should not completely dominate all features
        spike_idx = np.argmax(data['close'].values)
        spike_value = data['close'].iloc[spike_idx]
        normal_value = data['close'].iloc[0]
        
        dominated_features = 0
        total_features = len(features)
        
        for feature_name, feature_value in features.items():
            if np.isfinite(feature_value) and abs(feature_value) > 1000 * normal_value:
                dominated_features += 1
        
        if dominated_features > 0.5 * total_features:
            violations.append(f"Too many features ({dominated_features}/{total_features}) dominated by single outlier")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _audit_tsfresh_discriminative_power(self, features: Dict, data: pd.DataFrame, audit_results: Dict, scenario: str):
        """Audit if tsfresh features can discriminate between scenarios"""
        # This would require comparing across scenarios - simplified for now
        violations = []
        
        # Check if all features are identical (no discriminative power)
        unique_values = set()
        for feature_value in features.values():
            if np.isfinite(feature_value):
                unique_values.add(round(feature_value, 6))
        
        if len(unique_values) < 2:
            violations.append("All features have identical values - no discriminative power")
        
        if violations:
            audit_results['financial_logic_violations'].extend(violations)
    
    def _safe_sample(self, values, n=3):
        """Safely sample values for display"""
        try:
            if isinstance(values, (list, np.ndarray, pd.Series)):
                arr = np.array(values)
                if len(arr) <= n:
                    return arr.tolist()
                else:
                    indices = np.linspace(0, len(arr)-1, n, dtype=int)
                    return arr[indices].tolist()
            else:
                return [values]
        except:
            return ["Error sampling values"]
    
    def run_hostile_audit(self) -> Dict:
        """Run the complete hostile sensibility audit"""
        console.print(Panel.fit(
            "🔥 HOSTILE SENSIBILITY AUDIT 🔥\n"
            "Aggressively challenging SAGE model 'sensibility' claims\n"
            "Using extreme scenarios and real financial domain knowledge",
            title="CRITICAL FINANCIAL VALIDATION",
            border_style="red"
        ))
        
        # Create extreme test scenarios
        console.print("💀 Creating extreme market scenarios to break poorly designed models...")
        scenarios = self.create_extreme_market_scenarios()
        
        # Display scenarios
        scenario_table = Table(title="💀 Hostile Test Scenarios")
        scenario_table.add_column("Scenario", style="red")
        scenario_table.add_column("Description", style="yellow")
        scenario_table.add_column("Expected Model Behavior", style="cyan")
        
        scenario_table.add_row("Flash Crash", "10% drop in 5 minutes", "Should detect crash signal")
        scenario_table.add_row("Flat Line", "No price movement", "Should show low variance/complexity")
        scenario_table.add_row("Random Walk", "Pure noise", "Should show uncertainty, not false confidence")
        scenario_table.add_row("Single Spike", "One 100x outlier", "Should be robust, not dominated")
        scenario_table.add_row("Constant Increments", "Artificial $1/min pattern", "Should detect obvious trend")
        
        console.print(scenario_table)
        
        # Initialize models
        console.print("\n🧠 Initializing models for hostile testing...")
        alphaforge = AlphaForgeWrapper()
        catch22 = Catch22Wrapper()
        tsfresh = TSFreshWrapper()
        tsfresh.initialize_model()
        
        # Run hostile audits
        audit_results = {}
        
        # Audit AlphaForge
        audit_results['alphaforge'] = self.audit_alphaforge_sensibility(alphaforge, scenarios)
        
        # Audit catch22
        audit_results['catch22'] = self.audit_catch22_sensibility(catch22, scenarios)
        
        # Audit tsfresh
        audit_results['tsfresh'] = self.audit_tsfresh_sensibility(tsfresh, scenarios)
        
        # Generate hostile audit report
        self._generate_hostile_report(audit_results)
        
        return audit_results
    
    def _generate_hostile_report(self, audit_results: Dict):
        """Generate hostile audit report"""
        console.print("\n" + "="*80)
        console.print("💀 HOSTILE AUDIT RESULTS - FINANCIAL SENSIBILITY EXPOSED")
        console.print("="*80)
        
        total_violations = 0
        total_failures = 0
        
        for model_name, results in audit_results.items():
            console.print(f"\n🔍 {model_name.upper()} HOSTILE AUDIT:")
            
            violations = len(results['financial_logic_violations'])
            failures = len(results['critical_failures'])
            
            total_violations += violations
            total_failures += failures
            
            # Violations table
            if violations > 0:
                violation_table = Table(title=f"⚠️ {model_name.upper()} Financial Logic Violations")
                violation_table.add_column("Violation", style="red")
                
                for violation in results['financial_logic_violations']:
                    violation_table.add_row(violation)
                
                console.print(violation_table)
            else:
                console.print(f"✅ No financial logic violations found for {model_name.upper()}")
            
            # Technical failures
            if failures > 0:
                console.print(f"❌ Technical failures: {failures}")
                for failure in results['critical_failures']:
                    console.print(f"   • {failure}")
            
            # Sample outputs for inspection
            console.print(f"\n📊 Sample {model_name.upper()} outputs for manual inspection:")
            for scenario, analysis in results['detailed_analysis'].items():
                console.print(f"   {scenario}: {analysis}")
        
        # Final hostile verdict
        console.print("\n" + "="*80)
        console.print("⚖️ HOSTILE AUDIT VERDICT")
        console.print("="*80)
        
        if total_violations == 0 and total_failures == 0:
            verdict = "🎉 SURPRISINGLY ROBUST - Models pass hostile financial sensibility tests"
            color = "green"
        elif total_violations <= 2 and total_failures == 0:
            verdict = "⚠️ MOSTLY SENSIBLE - Minor issues but generally financially sound"
            color = "yellow"
        elif total_violations <= 5 or total_failures <= 2:
            verdict = "🔶 CONCERNING - Multiple financial logic violations detected"
            color = "bright_yellow"
        else:
            verdict = "💀 FINANCIALLY MEANINGLESS - Models fail basic financial sensibility"
            color = "red"
        
        console.print(Panel.fit(
            f"Total Financial Logic Violations: {total_violations}\n"
            f"Total Technical Failures: {total_failures}\n"
            f"Verdict: {verdict}",
            title="HOSTILE AUDIT FINAL VERDICT",
            border_style=color
        ))
        
        # Specific actionable recommendations
        if total_violations > 0 or total_failures > 0:
            console.print("\n🔧 ACTIONABLE RECOMMENDATIONS:")
            console.print("1. Review model parameters for financial realism")
            console.print("2. Add outlier robustness mechanisms")
            console.print("3. Implement sanity checks for extreme scenarios")
            console.print("4. Validate against known financial patterns")
            console.print("5. Add uncertainty quantification for low-signal environments")

def main():
    """Run hostile sensibility audit"""
    auditor = HostileSensibilityAuditor()
    results = auditor.run_hostile_audit()
    return results

if __name__ == "__main__":
    main()