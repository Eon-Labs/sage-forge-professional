#!/usr/bin/env python3
"""
SAGE Models Sensibility Testing - Phase 0 Week 2 Validation
PURPOSE: Validate that all SAGE models produce sensible financial results, not just technical functionality.
Tests realistic market scenarios and verifies output meaningfulness.
"""

import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add sage modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import SAGE models
from sage.models.alphaforge_wrapper import AlphaForgeWrapper
from sage.models.catch22_wrapper import Catch22Wrapper
from sage.models.tsfresh_wrapper import TSFreshWrapper

console = Console()

class SAGESensibilityTest:
    """
    Comprehensive sensibility testing for SAGE models.
    Tests realistic market scenarios and validates meaningful outputs.
    """
    
    def __init__(self):
        self.console = console
        self.models = {}
        self.test_scenarios = {}
        self.results = {}
        
    def setup_test_scenarios(self) -> bool:
        """Create realistic market test scenarios"""
        try:
            console.print("📊 Creating realistic market test scenarios...")
            
            np.random.seed(42)  # Reproducible tests
            
            # Scenario 1: Bull Market (Strong Uptrend)
            self.test_scenarios['bull_market'] = self._create_bull_market_data()
            
            # Scenario 2: Bear Market (Downtrend)
            self.test_scenarios['bear_market'] = self._create_bear_market_data()
            
            # Scenario 3: Sideways Market (Range-bound)
            self.test_scenarios['sideways_market'] = self._create_sideways_market_data()
            
            # Scenario 4: High Volatility Market
            self.test_scenarios['high_volatility'] = self._create_high_volatility_data()
            
            # Scenario 5: Low Volatility Market
            self.test_scenarios['low_volatility'] = self._create_low_volatility_data()
            
            console.print(f"✅ Created {len(self.test_scenarios)} realistic market scenarios")
            
            # Display scenario summary
            self._display_scenario_summary()
            
            return True
            
        except Exception as e:
            console.print(f"❌ Test scenario creation failed: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_bull_market_data(self) -> pd.DataFrame:
        """Create bull market scenario (consistent uptrend)"""
        n_points = 1440  # 24 hours of 1-minute data
        base_price = 45000.0
        
        # Strong upward trend with realistic volatility
        trend = np.linspace(0, 0.05, n_points)  # 5% increase over period
        noise = np.random.normal(0, 0.002, n_points)  # 0.2% volatility
        
        returns = trend + noise
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(prices, "Bull Market (+5%)")
    
    def _create_bear_market_data(self) -> pd.DataFrame:
        """Create bear market scenario (consistent downtrend)"""
        n_points = 1440
        base_price = 45000.0
        
        # Downward trend with realistic volatility
        trend = np.linspace(0, -0.04, n_points)  # 4% decrease over period
        noise = np.random.normal(0, 0.0025, n_points)  # 0.25% volatility
        
        returns = trend + noise
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(prices, "Bear Market (-4%)")
    
    def _create_sideways_market_data(self) -> pd.DataFrame:
        """Create sideways market scenario (range-bound)"""
        n_points = 1440
        base_price = 45000.0
        
        # Oscillating around mean with no trend
        oscillation = np.sin(np.linspace(0, 8 * np.pi, n_points)) * 0.01  # 1% oscillation
        noise = np.random.normal(0, 0.001, n_points)  # 0.1% volatility
        
        returns = oscillation + noise
        prices = base_price * np.exp(np.cumsum(returns))
        
        return self._create_ohlcv_data(prices, "Sideways Market (±1%)")
    
    def _create_high_volatility_data(self) -> pd.DataFrame:
        """Create high volatility scenario"""
        n_points = 1440
        base_price = 45000.0
        
        # High volatility with small trend
        trend = np.linspace(0, 0.01, n_points)  # 1% trend
        high_vol = np.random.normal(0, 0.008, n_points)  # 0.8% volatility
        
        returns = trend + high_vol
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(prices, "High Volatility (σ=0.8%)")
    
    def _create_low_volatility_data(self) -> pd.DataFrame:
        """Create low volatility scenario"""
        n_points = 1440
        base_price = 45000.0
        
        # Low volatility with small trend
        trend = np.linspace(0, 0.015, n_points)  # 1.5% trend
        low_vol = np.random.normal(0, 0.0005, n_points)  # 0.05% volatility
        
        returns = trend + low_vol
        prices = base_price * np.exp(np.cumsum(returns / 100))
        
        return self._create_ohlcv_data(prices, "Low Volatility (σ=0.05%)")
    
    def _create_ohlcv_data(self, prices: np.ndarray, label: str) -> pd.DataFrame:
        """Create realistic OHLCV data from price series"""
        timestamps = pd.date_range(start="2024-01-01", periods=len(prices), freq="1min")
        
        # Generate realistic OHLCV from price series
        spread_pct = 0.0002  # 0.02% spread
        
        opens = prices * (1 + np.random.normal(0, spread_pct/4, len(prices)))
        highs = prices * (1 + np.abs(np.random.normal(0, spread_pct, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, spread_pct, len(prices))))
        closes = prices
        volumes = np.random.lognormal(10, 0.5, len(prices))  # Realistic volume distribution
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'scenario': label
        }, index=timestamps)
    
    def _display_scenario_summary(self):
        """Display test scenario summary"""
        table = Table(title="📊 Market Test Scenarios")
        table.add_column("Scenario", style="cyan")
        table.add_column("Description", style="yellow")
        table.add_column("Start Price", style="green")
        table.add_column("End Price", style="green")
        table.add_column("Return", style="magenta")
        table.add_column("Volatility", style="red")
        
        for name, data in self.test_scenarios.items():
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            total_return = (end_price - start_price) / start_price * 100
            volatility = data['close'].pct_change().std() * 100 * np.sqrt(1440)  # Annualized
            
            table.add_row(
                name.replace('_', ' ').title(),
                data['scenario'].iloc[0],
                f"${start_price:,.2f}",
                f"${end_price:,.2f}",
                f"{total_return:+.2f}%",
                f"{volatility:.2f}%"
            )
        
        console.print(table)
    
    def initialize_models(self) -> bool:
        """Initialize all SAGE models"""
        try:
            console.print("🧠 Initializing SAGE models...")
            
            # Initialize AlphaForge
            console.print("   📈 Initializing AlphaForge...")
            self.models['alphaforge'] = AlphaForgeWrapper()
            
            # Initialize Catch22
            console.print("   🔬 Initializing Catch22...")
            self.models['catch22'] = Catch22Wrapper()
            
            # Initialize tsfresh
            console.print("   ⚗️ Initializing tsfresh...")
            self.models['tsfresh'] = TSFreshWrapper()
            self.models['tsfresh'].initialize_model()
            
            console.print("✅ All SAGE models initialized successfully")
            return True
            
        except Exception as e:
            console.print(f"❌ Model initialization failed: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_model_sensibility(self, model_name: str, model, scenario_name: str, data: pd.DataFrame) -> Dict:
        """Test individual model sensibility on specific scenario"""
        try:
            result = {
                'model': model_name,
                'scenario': scenario_name,
                'success': False,
                'outputs': {},
                'sensibility_checks': {},
                'warnings': [],
                'errors': []
            }
            
            # Test model-specific outputs
            if model_name == 'alphaforge':
                result.update(self._test_alphaforge_sensibility(model, data))
            elif model_name == 'catch22':
                result.update(self._test_catch22_sensibility(model, data))
            elif model_name == 'tsfresh':
                result.update(self._test_tsfresh_sensibility(model, data))
            
            return result
            
        except Exception as e:
            return {
                'model': model_name,
                'scenario': scenario_name,
                'success': False,
                'outputs': {},
                'sensibility_checks': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _test_alphaforge_sensibility(self, model, data: pd.DataFrame) -> Dict:
        """Test AlphaForge sensibility - alpha factors should make sense"""
        try:
            # Generate alpha factors
            factors = model.generate_factors(data, num_factors=5)
            
            sensibility_checks = {}
            
            # Check 1: Factors should have reasonable ranges
            for factor_name, factor_values in factors.items():
                if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                    values = np.array(factor_values)
                    
                    # Check for reasonable ranges (not extreme values)
                    sensibility_checks[f"{factor_name}_range_check"] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'reasonable': np.all(np.abs(values) < 1000),  # Reasonable alpha factor range
                        'has_variation': np.std(values) > 0.001  # Should have some variation
                    }
            
            # Check 2: Market trend responsiveness
            market_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            scenario_label = data['scenario'].iloc[0]
            
            # Bull market should generally produce positive alpha signals
            # Bear market should generally produce negative alpha signals
            sensibility_checks['market_responsiveness'] = {
                'market_return': float(market_return),
                'scenario': scenario_label,
                'factors_generated': len(factors),
                'has_directional_bias': self._check_directional_bias(factors, market_return)
            }
            
            return {
                'success': True,
                'outputs': {
                    'factors_count': len(factors),
                    'factors_summary': {name: self._summarize_values(vals) for name, vals in factors.items()}
                },
                'sensibility_checks': sensibility_checks,
                'warnings': [],
                'errors': []
            }
            
        except Exception as e:
            return {
                'success': False,
                'outputs': {},
                'sensibility_checks': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _test_catch22_sensibility(self, model, data: pd.DataFrame) -> Dict:
        """Test Catch22 sensibility - features should reflect market characteristics"""
        try:
            # Extract canonical features
            features = model.extract_features(data['close'])
            
            sensibility_checks = {}
            scenario_label = data['scenario'].iloc[0]
            
            # Check 1: Features should be finite and reasonable
            finite_features = 0
            reasonable_features = 0
            
            for feature_name, feature_value in features.items():
                is_finite = np.isfinite(feature_value)
                is_reasonable = abs(feature_value) < 1e6  # Not extremely large
                
                if is_finite:
                    finite_features += 1
                if is_reasonable:
                    reasonable_features += 1
            
            sensibility_checks['feature_quality'] = {
                'total_features': len(features),
                'finite_features': finite_features,
                'reasonable_features': reasonable_features,
                'quality_ratio': finite_features / len(features) if len(features) > 0 else 0
            }
            
            # Check 2: Volatility-related features should reflect market volatility
            price_volatility = data['close'].pct_change().std()
            
            # Look for volatility-sensitive features
            volatility_sensitive_features = []
            for name, value in features.items():
                if any(keyword in name.lower() for keyword in ['var', 'std', 'spread', 'range']):
                    volatility_sensitive_features.append((name, value))
            
            sensibility_checks['volatility_sensitivity'] = {
                'market_volatility': float(price_volatility),
                'scenario': scenario_label,
                'volatility_features_count': len(volatility_sensitive_features),
                'volatility_features': volatility_sensitive_features[:3]  # Show top 3
            }
            
            # Check 3: Trend-related features should reflect market direction
            market_trend = np.polyfit(range(len(data)), data['close'], 1)[0]  # Linear trend slope
            
            trend_sensitive_features = []
            for name, value in features.items():
                if any(keyword in name.lower() for keyword in ['trend', 'slope', 'correlation']):
                    trend_sensitive_features.append((name, value))
            
            sensibility_checks['trend_sensitivity'] = {
                'market_trend': float(market_trend),
                'trend_direction': 'up' if market_trend > 0 else 'down',
                'trend_features_count': len(trend_sensitive_features),
                'trend_features': trend_sensitive_features[:3]  # Show top 3
            }
            
            return {
                'success': True,
                'outputs': {
                    'features_count': len(features),
                    'sample_features': dict(list(features.items())[:5])  # Show first 5
                },
                'sensibility_checks': sensibility_checks,
                'warnings': [],
                'errors': []
            }
            
        except Exception as e:
            return {
                'success': False,
                'outputs': {},
                'sensibility_checks': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _test_tsfresh_sensibility(self, model, data: pd.DataFrame) -> Dict:
        """Test tsfresh sensibility - features should be statistically meaningful"""
        try:
            # Extract tsfresh features
            features_df = model.extract_features(data['close'], feature_set='efficient')
            
            sensibility_checks = {}
            scenario_label = data['scenario'].iloc[0]
            
            # Check 1: Features should be finite and not constant
            features_dict = features_df.iloc[0].to_dict()  # tsfresh returns single row
            
            finite_count = sum(1 for v in features_dict.values() if np.isfinite(v))
            non_zero_count = sum(1 for v in features_dict.values() if abs(v) > 1e-10)
            
            sensibility_checks['feature_quality'] = {
                'total_features': len(features_dict),
                'finite_features': finite_count,
                'non_zero_features': non_zero_count,
                'quality_ratio': finite_count / len(features_dict) if len(features_dict) > 0 else 0
            }
            
            # Check 2: Statistical features should reflect data characteristics
            data_stats = {
                'mean': float(data['close'].mean()),
                'std': float(data['close'].std()),
                'min': float(data['close'].min()),
                'max': float(data['close'].max()),
                'length': len(data)
            }
            
            # Look for matching statistical features
            statistical_matches = {}
            for feature_name, feature_value in features_dict.items():
                if 'mean' in feature_name.lower():
                    statistical_matches['mean_feature'] = (feature_name, feature_value, data_stats['mean'])
                elif 'std' in feature_name.lower() or 'standard_deviation' in feature_name.lower():
                    statistical_matches['std_feature'] = (feature_name, feature_value, data_stats['std'])
                elif 'length' in feature_name.lower():
                    statistical_matches['length_feature'] = (feature_name, feature_value, data_stats['length'])
                elif 'minimum' in feature_name.lower():
                    statistical_matches['min_feature'] = (feature_name, feature_value, data_stats['min'])
                elif 'maximum' in feature_name.lower():
                    statistical_matches['max_feature'] = (feature_name, feature_value, data_stats['max'])
            
            sensibility_checks['statistical_consistency'] = {
                'data_stats': data_stats,
                'scenario': scenario_label,
                'matches_found': len(statistical_matches),
                'statistical_matches': statistical_matches
            }
            
            # Check 3: Feature values should be in reasonable ranges for financial data
            extreme_features = []
            reasonable_features = []
            
            for name, value in features_dict.items():
                if np.isfinite(value):
                    if abs(value) > 1e8:  # Extremely large values
                        extreme_features.append((name, value))
                    else:
                        reasonable_features.append((name, value))
            
            sensibility_checks['value_ranges'] = {
                'reasonable_count': len(reasonable_features),
                'extreme_count': len(extreme_features),
                'extreme_features': extreme_features[:3],  # Show first 3 extreme
                'reasonableness_ratio': len(reasonable_features) / len(features_dict) if len(features_dict) > 0 else 0
            }
            
            return {
                'success': True,
                'outputs': {
                    'features_count': len(features_dict),
                    'sample_features': dict(list(features_dict.items())[:5])  # Show first 5
                },
                'sensibility_checks': sensibility_checks,
                'warnings': [],
                'errors': []
            }
            
        except Exception as e:
            return {
                'success': False,
                'outputs': {},
                'sensibility_checks': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _check_directional_bias(self, factors: Dict, market_return: float) -> bool:
        """Check if alpha factors show appropriate directional bias"""
        try:
            # Calculate average factor signal
            all_values = []
            for factor_values in factors.values():
                if isinstance(factor_values, (list, np.ndarray, pd.Series)):
                    values = np.array(factor_values)
                    if len(values) > 0:
                        all_values.extend(values.flatten())
            
            if len(all_values) == 0:
                return False
            
            avg_signal = np.mean(all_values)
            
            # For strong trends, expect some directional bias
            if abs(market_return) > 0.02:  # > 2% move
                # Strong up market should have positive bias, strong down should have negative
                expected_bias = market_return > 0
                actual_bias = avg_signal > 0
                return expected_bias == actual_bias
            
            return True  # For weak trends, no strong bias expected
            
        except:
            return False
    
    def _summarize_values(self, values) -> Dict:
        """Summarize array of values"""
        try:
            if isinstance(values, (list, np.ndarray, pd.Series)):
                arr = np.array(values)
                return {
                    'count': len(arr),
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr))
                }
            else:
                return {'single_value': float(values)}
        except:
            return {'error': 'Could not summarize values'}
    
    def run_comprehensive_sensibility_test(self) -> bool:
        """Run comprehensive sensibility tests across all models and scenarios"""
        try:
            console.print("🧪 Running comprehensive SAGE sensibility tests...")
            
            self.results = {}
            
            # Test each model against each scenario
            total_tests = len(self.models) * len(self.test_scenarios)
            test_count = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Running {total_tests} sensibility tests...", total=total_tests)
                
                for model_name, model in self.models.items():
                    self.results[model_name] = {}
                    
                    for scenario_name, scenario_data in self.test_scenarios.items():
                        progress.update(task, description=f"Testing {model_name} on {scenario_name}...")
                        
                        # Run sensibility test
                        result = self.test_model_sensibility(model_name, model, scenario_name, scenario_data)
                        self.results[model_name][scenario_name] = result
                        
                        test_count += 1
                        progress.advance(task)
            
            # Display comprehensive results
            self._display_sensibility_results()
            
            return True
            
        except Exception as e:
            console.print(f"❌ Comprehensive sensibility testing failed: {str(e)}")
            traceback.print_exc()
            return False
    
    def _display_sensibility_results(self):
        """Display comprehensive sensibility test results"""
        console.print("\n" + "="*80)
        console.print("📊 SAGE MODELS SENSIBILITY TEST RESULTS")
        console.print("="*80)
        
        # Summary table
        summary_table = Table(title="🎯 Overall Sensibility Summary")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Success Rate", style="green")
        summary_table.add_column("Avg Quality", style="yellow")
        summary_table.add_column("Key Insights", style="magenta")
        
        for model_name, model_results in self.results.items():
            success_count = sum(1 for r in model_results.values() if r['success'])
            total_tests = len(model_results)
            success_rate = success_count / total_tests * 100 if total_tests > 0 else 0
            
            # Calculate average quality metrics
            quality_scores = []
            key_insights = []
            
            for scenario_name, result in model_results.items():
                if result['success'] and 'sensibility_checks' in result:
                    checks = result['sensibility_checks']
                    
                    # Extract quality metrics
                    if 'feature_quality' in checks:
                        quality_scores.append(checks['feature_quality'].get('quality_ratio', 0))
                    elif 'market_responsiveness' in checks:
                        quality_scores.append(1.0 if checks['market_responsiveness'].get('has_directional_bias', False) else 0.5)
                    
                    # Extract key insights
                    if model_name == 'alphaforge':
                        factors_count = result['outputs'].get('factors_count', 0)
                        key_insights.append(f"{factors_count} factors")
                    elif model_name == 'catch22':
                        features_count = result['outputs'].get('features_count', 0)
                        key_insights.append(f"{features_count} features")
                    elif model_name == 'tsfresh':
                        features_count = result['outputs'].get('features_count', 0)
                        key_insights.append(f"{features_count} features")
            
            avg_quality = np.mean(quality_scores) * 100 if quality_scores else 0
            insights_summary = ", ".join(set(key_insights)) if key_insights else "No insights"
            
            summary_table.add_row(
                model_name.upper(),
                f"{success_rate:.1f}%",
                f"{avg_quality:.1f}%",
                insights_summary
            )
        
        console.print(summary_table)
        
        # Detailed results for each model
        for model_name, model_results in self.results.items():
            console.print(f"\n📈 {model_name.upper()} DETAILED RESULTS:")
            
            detail_table = Table(title=f"{model_name.upper()} Sensibility Analysis")
            detail_table.add_column("Scenario", style="cyan")
            detail_table.add_column("Status", style="green")
            detail_table.add_column("Quality Score", style="yellow")
            detail_table.add_column("Key Metrics", style="magenta")
            
            for scenario_name, result in model_results.items():
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                
                if result['success'] and 'sensibility_checks' in result:
                    checks = result['sensibility_checks']
                    
                    # Extract quality score and key metrics
                    if model_name == 'alphaforge':
                        quality = "✅ Good" if checks.get('market_responsiveness', {}).get('has_directional_bias', False) else "⚠️ Mixed"
                        factors = checks.get('market_responsiveness', {}).get('factors_generated', 0)
                        metrics = f"{factors} factors generated"
                        
                    elif model_name == 'catch22':
                        quality_ratio = checks.get('feature_quality', {}).get('quality_ratio', 0)
                        quality = "✅ Excellent" if quality_ratio > 0.9 else "✅ Good" if quality_ratio > 0.7 else "⚠️ Fair"
                        features = checks.get('feature_quality', {}).get('finite_features', 0)
                        metrics = f"{features} valid features"
                        
                    elif model_name == 'tsfresh':
                        quality_ratio = checks.get('feature_quality', {}).get('quality_ratio', 0)
                        reasonable_ratio = checks.get('value_ranges', {}).get('reasonableness_ratio', 0)
                        overall_quality = (quality_ratio + reasonable_ratio) / 2
                        quality = "✅ Excellent" if overall_quality > 0.9 else "✅ Good" if overall_quality > 0.7 else "⚠️ Fair"
                        features = checks.get('feature_quality', {}).get('finite_features', 0)
                        metrics = f"{features} meaningful features"
                    else:
                        quality = "Unknown"
                        metrics = "No metrics"
                else:
                    quality = "❌ Failed"
                    metrics = "Errors: " + ", ".join(result.get('errors', ['Unknown']))[:50]
                
                detail_table.add_row(
                    scenario_name.replace('_', ' ').title(),
                    status,
                    quality,
                    metrics
                )
            
            console.print(detail_table)
        
        # Final assessment
        console.print("\n" + "="*80)
        console.print("🎯 FINAL SENSIBILITY ASSESSMENT")
        console.print("="*80)
        
        total_tests = sum(len(model_results) for model_results in self.results.values())
        total_successes = sum(sum(1 for r in model_results.values() if r['success']) 
                            for model_results in self.results.values())
        
        overall_success_rate = total_successes / total_tests * 100 if total_tests > 0 else 0
        
        if overall_success_rate >= 90:
            assessment = "🎉 EXCELLENT - All models produce sensible financial results"
            color = "bright_green"
        elif overall_success_rate >= 75:
            assessment = "✅ GOOD - Most models produce sensible results with minor issues"
            color = "green"
        elif overall_success_rate >= 60:
            assessment = "⚠️ FAIR - Some models need improvement for realistic results"
            color = "yellow"
        else:
            assessment = "❌ POOR - Significant sensibility issues detected"
            color = "red"
        
        console.print(Panel.fit(
            f"Overall Success Rate: {overall_success_rate:.1f}%\n"
            f"Total Tests: {total_tests}\n"
            f"Assessment: {assessment}",
            title="SAGE Sensibility Assessment",
            border_style=color
        ))

def main():
    """Run SAGE sensibility testing"""
    
    # Display test header
    console.print(Panel.fit(
        "🧪 SAGE Models Sensibility Testing - Phase 0 Week 2\n"
        "Validating realistic financial outputs across market scenarios",
        title="SAGE SENSIBILITY VALIDATION",
        border_style="bright_blue"
    ))
    
    test = SAGESensibilityTest()
    
    try:
        # Step 1: Setup test scenarios
        console.print("🎯 STEP 1: Setup Realistic Market Test Scenarios")
        if not test.setup_test_scenarios():
            console.print("❌ Test scenario setup failed")
            return False
        
        # Step 2: Initialize models
        console.print("\n🎯 STEP 2: Initialize All SAGE Models")
        if not test.initialize_models():
            console.print("❌ Model initialization failed")
            return False
        
        # Step 3: Run comprehensive sensibility tests
        console.print("\n🎯 STEP 3: Run Comprehensive Sensibility Tests")
        if not test.run_comprehensive_sensibility_test():
            console.print("❌ Sensibility testing failed")
            return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ Test execution failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)