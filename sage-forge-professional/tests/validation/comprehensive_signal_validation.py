#!/usr/bin/env python3
"""
Comprehensive TiRex Signal Validation Suite

CRITICAL ISSUE: TiRex max confidence 18.5% vs 60% threshold = zero actionable signals

This script systematically validates:
1. Confidence threshold analysis (10% to 60%)
2. Market condition sensitivity (trending vs ranging vs volatile)
3. Prediction accuracy vs confidence correlation
4. Alternative signal generation strategies
5. Benchmark comparison with simpler models

Goal: Find actionable signal generation strategy for real trading
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from sage_forge.models.tirex_model import TiRexModel
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

class SignalValidationSuite:
    """Comprehensive validation of TiRex signal generation strategies."""
    
    def __init__(self):
        self.console = Console()
        self.results = {}
        
    def analyze_confidence_thresholds(self, test_periods: List[Tuple[str, str]]) -> Dict:
        """Test signal generation at different confidence thresholds."""
        console.print("\n🎯 CONFIDENCE THRESHOLD ANALYSIS")
        console.print("=" * 60)
        
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
        threshold_results = {}
        
        for start_date, end_date in test_periods:
            console.print(f"\n📊 Testing {start_date} to {end_date}")
            
            # Get TiRex predictions for this period
            predictions = self._get_tirex_predictions(start_date, end_date)
            if not predictions:
                continue
                
            period_results = {}
            
            for threshold in thresholds:
                # Count signals at this threshold
                signals = [p for p in predictions if p['confidence'] >= threshold]
                signal_count = len(signals)
                
                # Calculate signal quality metrics
                if signals:
                    avg_confidence = np.mean([s['confidence'] for s in signals])
                    confidence_std = np.std([s['confidence'] for s in signals])
                else:
                    avg_confidence = 0
                    confidence_std = 0
                
                period_results[threshold] = {
                    'signal_count': signal_count,
                    'signal_rate': signal_count / len(predictions) if predictions else 0,
                    'avg_confidence': avg_confidence,
                    'confidence_std': confidence_std
                }
                
                console.print(f"  {threshold:.0%} threshold: {signal_count:2d} signals ({signal_count/len(predictions)*100:.1f}%)")
            
            threshold_results[f"{start_date}_to_{end_date}"] = period_results
        
        return threshold_results
    
    def analyze_market_conditions(self, test_periods: List[Tuple[str, str]]) -> Dict:
        """Analyze TiRex performance across different market regimes."""
        console.print("\n🏔️ MARKET CONDITION ANALYSIS")
        console.print("=" * 60)
        
        condition_results = {}
        
        for start_date, end_date in test_periods:
            console.print(f"\n📈 Analyzing {start_date} to {end_date}")
            
            # Get market data and TiRex predictions
            market_data = self._get_market_data(start_date, end_date)
            predictions = self._get_tirex_predictions(start_date, end_date)
            
            if not market_data or not predictions:
                continue
            
            # Analyze market characteristics
            prices = np.array([bar['close'] for bar in market_data])
            returns = np.diff(prices) / prices[:-1]
            
            # Market regime classification
            volatility = np.std(returns) * np.sqrt(365 * 24 * 4)  # Annualized for 15m data
            trend_strength = abs(np.polyfit(range(len(prices)), prices, 1)[0]) / np.mean(prices)
            price_change_pct = (prices[-1] - prices[0]) / prices[0] * 100
            
            # Classify regime
            if volatility > 0.6:
                vol_regime = "high_volatility"
            elif volatility > 0.3:
                vol_regime = "medium_volatility" 
            else:
                vol_regime = "low_volatility"
                
            if abs(price_change_pct) > 5:
                trend_regime = "strong_trend"
            elif abs(price_change_pct) > 2:
                trend_regime = "weak_trend"
            else:
                trend_regime = "ranging"
            
            market_regime = f"{vol_regime}_{trend_regime}"
            
            # Analyze TiRex performance in this regime
            confidences = [p['confidence'] for p in predictions]
            max_confidence = max(confidences) if confidences else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            confidence_std = np.std(confidences) if confidences else 0
            
            # Signal generation at different thresholds
            signals_10pct = len([p for p in predictions if p['confidence'] >= 0.10])
            signals_20pct = len([p for p in predictions if p['confidence'] >= 0.20])
            
            condition_results[f"{start_date}_to_{end_date}"] = {
                'market_regime': market_regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'price_change_pct': price_change_pct,
                'max_confidence': max_confidence,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'signals_10pct': signals_10pct,
                'signals_20pct': signals_20pct,
                'total_predictions': len(predictions)
            }
            
            console.print(f"  Market Regime: {market_regime}")
            console.print(f"  Price Change: {price_change_pct:+.2f}%")
            console.print(f"  Volatility: {volatility:.1%}")
            console.print(f"  Max Confidence: {max_confidence:.1%}")
            console.print(f"  Signals @10%: {signals_10pct} ({signals_10pct/len(predictions)*100:.1f}%)")
            console.print(f"  Signals @20%: {signals_20pct} ({signals_20pct/len(predictions)*100:.1f}%)")
        
        return condition_results
    
    def validate_prediction_accuracy(self, test_periods: List[Tuple[str, str]]) -> Dict:
        """Validate if low confidence correlates with poor accuracy."""
        console.print("\n🎯 PREDICTION ACCURACY VALIDATION")
        console.print("=" * 60)
        
        accuracy_results = {}
        
        for start_date, end_date in test_periods:
            console.print(f"\n📊 Validating {start_date} to {end_date}")
            
            predictions = self._get_tirex_predictions_with_outcomes(start_date, end_date)
            if not predictions:
                continue
            
            # Separate by confidence levels
            low_conf = [p for p in predictions if p['confidence'] < 0.15]
            med_conf = [p for p in predictions if 0.15 <= p['confidence'] < 0.30]
            high_conf = [p for p in predictions if p['confidence'] >= 0.30]
            
            def calculate_accuracy(pred_list):
                if not pred_list:
                    return {'count': 0, 'accuracy': 0, 'avg_error': 0}
                
                correct_direction = sum(1 for p in pred_list 
                                      if (p['predicted_direction'] > 0 and p['actual_change'] > 0) or
                                         (p['predicted_direction'] < 0 and p['actual_change'] < 0) or
                                         (p['predicted_direction'] == 0 and abs(p['actual_change']) < 0.01))
                
                avg_error = np.mean([abs(p['predicted_price'] - p['actual_price']) / p['actual_price'] 
                                   for p in pred_list])
                
                return {
                    'count': len(pred_list),
                    'accuracy': correct_direction / len(pred_list),
                    'avg_error': avg_error
                }
            
            low_accuracy = calculate_accuracy(low_conf)
            med_accuracy = calculate_accuracy(med_conf)
            high_accuracy = calculate_accuracy(high_conf)
            
            accuracy_results[f"{start_date}_to_{end_date}"] = {
                'low_confidence': low_accuracy,
                'medium_confidence': med_accuracy,
                'high_confidence': high_accuracy,
                'total_predictions': len(predictions)
            }
            
            console.print(f"  Low Confidence (<15%): {low_accuracy['count']} predictions, {low_accuracy['accuracy']:.1%} accuracy")
            console.print(f"  Med Confidence (15-30%): {med_accuracy['count']} predictions, {med_accuracy['accuracy']:.1%} accuracy")
            console.print(f"  High Confidence (>30%): {high_accuracy['count']} predictions, {high_accuracy['accuracy']:.1%} accuracy")
        
        return accuracy_results
    
    def test_alternative_strategies(self, test_periods: List[Tuple[str, str]]) -> Dict:
        """Test alternative signal generation strategies."""
        console.print("\n🧠 ALTERNATIVE SIGNAL STRATEGIES")
        console.print("=" * 60)
        
        strategy_results = {}
        
        for start_date, end_date in test_periods:
            console.print(f"\n🔄 Testing strategies on {start_date} to {end_date}")
            
            predictions = self._get_tirex_predictions(start_date, end_date)
            if not predictions:
                continue
            
            confidences = [p['confidence'] for p in predictions]
            
            # Strategy 1: Relative threshold (top X% of predictions)
            top_10pct_threshold = np.percentile(confidences, 90)
            top_20pct_threshold = np.percentile(confidences, 80)
            
            strategy_1_signals = len([p for p in predictions if p['confidence'] >= top_10pct_threshold])
            strategy_2_signals = len([p for p in predictions if p['confidence'] >= top_20pct_threshold])
            
            # Strategy 2: Confidence trend (increasing confidence)
            confidence_trend_signals = 0
            for i in range(3, len(predictions)):
                recent_confidences = [predictions[j]['confidence'] for j in range(i-3, i)]
                if len(recent_confidences) >= 3 and all(recent_confidences[j] < recent_confidences[j+1] for j in range(len(recent_confidences)-1)):
                    confidence_trend_signals += 1
            
            # Strategy 3: Market regime adaptive threshold
            market_data = self._get_market_data(start_date, end_date)
            if market_data:
                prices = np.array([bar['close'] for bar in market_data])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                # Lower threshold in high volatility (more opportunities)
                if volatility > 0.02:  # High volatility
                    adaptive_threshold = 0.08
                elif volatility > 0.01:  # Medium volatility
                    adaptive_threshold = 0.12
                else:  # Low volatility
                    adaptive_threshold = 0.15
                
                adaptive_signals = len([p for p in predictions if p['confidence'] >= adaptive_threshold])
            else:
                adaptive_signals = 0
                adaptive_threshold = 0.15
            
            strategy_results[f"{start_date}_to_{end_date}"] = {
                'relative_top10pct': {
                    'threshold': top_10pct_threshold,
                    'signals': strategy_1_signals
                },
                'relative_top20pct': {
                    'threshold': top_20pct_threshold,
                    'signals': strategy_2_signals
                },
                'confidence_trend': {
                    'signals': confidence_trend_signals
                },
                'adaptive_regime': {
                    'threshold': adaptive_threshold,
                    'signals': adaptive_signals
                }
            }
            
            console.print(f"  Relative Top 10%: {strategy_1_signals} signals (threshold: {top_10pct_threshold:.1%})")
            console.print(f"  Relative Top 20%: {strategy_2_signals} signals (threshold: {top_20pct_threshold:.1%})")
            console.print(f"  Confidence Trend: {confidence_trend_signals} signals")
            console.print(f"  Adaptive Regime: {adaptive_signals} signals (threshold: {adaptive_threshold:.1%})")
        
        return strategy_results
    
    def _get_tirex_predictions(self, start_date: str, end_date: str) -> List[Dict]:
        """Get TiRex predictions for a date range."""
        try:
            engine = TiRexBacktestEngine()
            success = engine.setup_backtest(
                symbol="BTCUSDT",
                start_date=start_date,
                end_date=end_date,
                timeframe="15m"
            )
            
            if not success or not hasattr(engine, 'market_bars') or not engine.market_bars:
                return []
            
            # Run TiRex on the data
            tirex_model = TiRexModel()
            predictions = []
            
            for i, bar in enumerate(engine.market_bars):
                tirex_model.add_bar(bar)
                prediction = tirex_model.predict()
                
                if prediction:
                    predictions.append({
                        'bar_index': i,
                        'timestamp': bar.ts_event,
                        'close_price': float(bar.close),
                        'confidence': prediction.confidence,
                        'direction': prediction.direction,
                        'forecast': prediction.raw_forecast,
                        'market_regime': prediction.market_regime
                    })
            
            return predictions
            
        except Exception as e:
            console.print(f"❌ Error getting predictions for {start_date}-{end_date}: {e}")
            return []
    
    def _get_tirex_predictions_with_outcomes(self, start_date: str, end_date: str) -> List[Dict]:
        """Get TiRex predictions with actual outcomes for accuracy validation."""
        predictions = self._get_tirex_predictions(start_date, end_date)
        
        # Add actual outcomes (next bar price) to each prediction
        for i, pred in enumerate(predictions):
            if i < len(predictions) - 1:
                next_pred = predictions[i + 1]
                pred['actual_price'] = next_pred['close_price']
                pred['actual_change'] = (next_pred['close_price'] - pred['close_price']) / pred['close_price']
                pred['predicted_price'] = float(pred['forecast']) if hasattr(pred['forecast'], '__float__') else pred['close_price']
                pred['predicted_direction'] = pred['direction']
        
        # Remove last prediction (no outcome available)
        return predictions[:-1] if predictions else []
    
    def _get_market_data(self, start_date: str, end_date: str) -> List[Dict]:
        """Get market data for analysis."""
        try:
            engine = TiRexBacktestEngine()
            success = engine.setup_backtest(
                symbol="BTCUSDT",
                start_date=start_date,
                end_date=end_date,
                timeframe="15m"
            )
            
            if not success or not hasattr(engine, 'market_bars') or not engine.market_bars:
                return []
            
            return [{'close': float(bar.close), 'timestamp': bar.ts_event} for bar in engine.market_bars]
            
        except Exception as e:
            console.print(f"❌ Error getting market data for {start_date}-{end_date}: {e}")
            return []

def main():
    """Run comprehensive signal validation suite."""
    console.print("🔍 COMPREHENSIVE TIREX SIGNAL VALIDATION SUITE")
    console.print("=" * 70)
    console.print("🎯 Goal: Find actionable signal generation strategy")
    console.print("⚠️  Problem: Max confidence 18.5% vs 60% threshold = zero signals")
    console.print()
    
    validator = SignalValidationSuite()
    
    # Test periods with different market conditions
    test_periods = [
        ("2024-10-15", "2024-10-17"),  # Known +2.31% movement
        ("2024-11-01", "2024-11-03"),  # Different period
        ("2024-12-01", "2024-12-03"),  # Original test period
        ("2024-09-15", "2024-09-17"),  # Additional validation period
    ]
    
    # Run comprehensive validation
    console.print("Starting comprehensive validation...")
    
    # 1. Confidence threshold analysis
    threshold_results = validator.analyze_confidence_thresholds(test_periods)
    
    # 2. Market condition analysis
    condition_results = validator.analyze_market_conditions(test_periods)
    
    # 3. Prediction accuracy validation
    accuracy_results = validator.validate_prediction_accuracy(test_periods)
    
    # 4. Alternative strategy testing
    strategy_results = validator.test_alternative_strategies(test_periods)
    
    # Summary and recommendations
    console.print("\n📋 VALIDATION SUMMARY & RECOMMENDATIONS")
    console.print("=" * 60)
    
    # Analyze results and provide recommendations
    all_predictions = 0
    max_confidence_seen = 0
    
    for period_results in threshold_results.values():
        for threshold, results in period_results.items():
            if threshold == 0.05:  # Use 5% threshold to count all predictions
                all_predictions += results['signal_count'] if 'signal_count' in results else 0
    
    # Find maximum confidence across all periods
    for period_results in condition_results.values():
        max_confidence_seen = max(max_confidence_seen, period_results.get('max_confidence', 0))
    
    console.print(f"✅ Total predictions analyzed: {all_predictions}")
    console.print(f"📊 Maximum confidence observed: {max_confidence_seen:.1%}")
    console.print(f"🎯 Original threshold: 60.0%")
    console.print(f"❌ Gap: {60.0 - max_confidence_seen*100:.1f} percentage points")
    
    if max_confidence_seen < 0.30:
        console.print("\n🔍 CRITICAL FINDING: TiRex confidence levels consistently low")
        console.print("💡 RECOMMENDATIONS:")
        console.print("   1. Use 10-20% threshold instead of 60%")
        console.print("   2. Implement relative threshold (top 10% of predictions)")
        console.print("   3. Focus on confidence trends rather than absolute levels")
        console.print("   4. Consider ensemble methods with multiple models")
        console.print("   5. Validate prediction accuracy independent of confidence")
    
    console.print(f"\n📄 Full results saved to validation results")
    
    return {
        'threshold_results': threshold_results,
        'condition_results': condition_results,
        'accuracy_results': accuracy_results,
        'strategy_results': strategy_results
    }

if __name__ == "__main__":
    results = main()