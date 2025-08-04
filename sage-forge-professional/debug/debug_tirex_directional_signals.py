#!/usr/bin/env python3
"""
Debug TiRex Directional Signal Generation

CRITICAL ISSUE: TiRex generates predictions but all result in HOLD signals (direction=0)
OBJECTIVE: Find why forecast interpretation isn't producing BUY/SELL signals

Investigation areas:
1. What are the actual TiRex forecast values vs current prices?
2. Are the directional thresholds too strict?
3. How sensitive is the threshold to getting BUY/SELL signals?
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from sage_forge.models.tirex_model import TiRexModel
from rich.console import Console
from rich.table import Table

console = Console()

def debug_tirex_directional_signals():
    """Debug why TiRex only generates HOLD signals."""
    console.print("🔍 DEBUGGING TIREX DIRECTIONAL SIGNALS")
    console.print("=" * 60)
    console.print("❌ Problem: All TiRex signals are HOLD (direction=0)")
    console.print("🎯 Goal: Find why no BUY/SELL signals are generated")
    console.print()
    
    # Load market data and TiRex model
    engine = TiRexBacktestEngine()
    success = engine.setup_backtest("BTCUSDT", "2024-10-15", "2024-10-17", timeframe="15m")
    
    if not success or not hasattr(engine, 'market_bars'):
        console.print("❌ Failed to load market data")
        return
    
    bars = engine.market_bars
    console.print(f"📊 Loaded {len(bars)} bars for analysis")
    
    # Initialize TiRex model
    tirex_model = TiRexModel()
    if not tirex_model.is_loaded:
        console.print("❌ TiRex model failed to load")
        return
    
    console.print("✅ TiRex model loaded, starting analysis...")
    console.print()
    
    # Collect detailed prediction data
    detailed_predictions = []
    
    for i, bar in enumerate(bars):
        # Add bar to model
        tirex_model.add_bar(bar)
        
        # Get prediction
        prediction = tirex_model.predict()
        if prediction is None:
            continue
        
        current_price = float(bar.close)
        
        # Get raw forecast value - handle different TiRex output formats
        try:
            if isinstance(prediction.raw_forecast, np.ndarray) and prediction.raw_forecast.shape == ():
                # Scalar case
                forecast_value = float(prediction.raw_forecast)
            elif hasattr(prediction.raw_forecast, '__len__') and len(prediction.raw_forecast) > 0:
                # Array case
                forecast_value = float(prediction.raw_forecast[0])
            else:
                # Fallback
                forecast_value = float(prediction.raw_forecast)
        except Exception as e:
            console.print(f"⚠️  Error extracting forecast value: {e}")
            console.print(f"   Raw forecast type: {type(prediction.raw_forecast)}")
            console.print(f"   Raw forecast: {prediction.raw_forecast}")
            continue
        
        # Calculate price change details
        price_change = forecast_value - current_price
        relative_change = price_change / current_price
        relative_change_pct = relative_change * 100
        
        # Test different thresholds
        thresholds_to_test = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]  # 0.01% to 1%
        threshold_results = {}
        
        for threshold in thresholds_to_test:
            if relative_change > threshold:
                direction = 1  # BUY
            elif relative_change < -threshold:
                direction = -1  # SELL
            else:
                direction = 0  # HOLD
            threshold_results[threshold] = direction
        
        detailed_predictions.append({
            'bar_index': i,
            'current_price': current_price,
            'forecast_value': forecast_value,
            'price_change': price_change,
            'relative_change_pct': relative_change_pct,
            'actual_direction': prediction.direction,
            'confidence': prediction.confidence,
            'threshold_results': threshold_results
        })
        
        # Show first few predictions in detail
        if len(detailed_predictions) <= 5:
            console.print(f"🔍 Prediction {len(detailed_predictions)}:")
            console.print(f"   Current Price: ${current_price:.2f}")
            console.print(f"   TiRex Forecast: ${forecast_value:.2f}")
            console.print(f"   Price Change: ${price_change:.4f} ({relative_change_pct:+.4f}%)")
            console.print(f"   Direction: {prediction.direction} ({'BUY' if prediction.direction == 1 else 'SELL' if prediction.direction == -1 else 'HOLD'})")
            console.print(f"   Confidence: {prediction.confidence:.1%}")
            console.print()
    
    if not detailed_predictions:
        console.print("❌ No predictions generated")
        return
    
    console.print(f"📊 Analysis of {len(detailed_predictions)} predictions:")
    console.print()
    
    # Analyze prediction characteristics
    price_changes = [p['price_change'] for p in detailed_predictions]
    relative_changes = [p['relative_change_pct'] for p in detailed_predictions]
    
    console.print("📈 FORECAST CHARACTERISTICS:")
    console.print(f"   Price change range: ${min(price_changes):.4f} to ${max(price_changes):.4f}")
    console.print(f"   Relative change range: {min(relative_changes):+.4f}% to {max(relative_changes):+.4f}%")
    console.print(f"   Average change: {np.mean(relative_changes):+.4f}%")
    console.print(f"   Std deviation: {np.std(relative_changes):.4f}%")
    console.print()
    
    # Test threshold sensitivity
    console.print("🎯 THRESHOLD SENSITIVITY ANALYSIS:")
    
    table = Table(title="Signal Generation by Threshold", show_header=True, header_style="bold cyan")
    table.add_column("Threshold", style="white")
    table.add_column("BUY Signals", style="green")
    table.add_column("SELL Signals", style="red")
    table.add_column("HOLD Signals", style="yellow")
    table.add_column("Total Actionable", style="blue")
    
    for threshold in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for prediction in detailed_predictions:
            direction = prediction['threshold_results'][threshold]
            if direction == 1:
                buy_count += 1
            elif direction == -1:
                sell_count += 1
            else:
                hold_count += 1
        
        actionable = buy_count + sell_count
        threshold_pct = threshold * 100
        
        table.add_row(
            f"{threshold_pct:.3f}%",
            str(buy_count),
            str(sell_count),
            str(hold_count),
            str(actionable)
        )
    
    console.print(table)
    console.print()
    
    # Recommendations
    console.print("💡 RECOMMENDATIONS:")
    
    # Find optimal threshold
    best_threshold = None
    max_actionable = 0
    
    for threshold in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        actionable = sum(1 for p in detailed_predictions 
                        if p['threshold_results'][threshold] != 0)
        if actionable > max_actionable:
            max_actionable = actionable
            best_threshold = threshold
    
    if best_threshold:
        console.print(f"🎯 Optimal threshold: {best_threshold*100:.3f}% ({max_actionable} actionable signals)")
        console.print(f"   Current threshold: 0.100% (0 actionable signals)")
        console.print(f"   Improvement: {max_actionable} vs 0 signals")
    
    # Check if TiRex forecasts are too conservative
    avg_abs_change = np.mean([abs(c) for c in relative_changes])
    console.print(f"📊 Average absolute forecast change: {avg_abs_change:.4f}%")
    
    if avg_abs_change < 0.1:
        console.print("⚠️  TiRex forecasts are very small movements (< 0.1%)")
        console.print("💡 Consider lowering directional threshold to 0.001-0.005%")
    elif avg_abs_change < 0.01:
        console.print("⚠️  TiRex forecasts are extremely small (< 0.01%)")
        console.print("💡 Consider threshold of 0.0001-0.001%")
    
    console.print()
    console.print("🔧 NEXT STEPS:")
    console.print("1. Lower directional threshold in TiRexModel._interpret_forecast()")
    console.print("2. Test with optimized threshold")
    console.print("3. Validate signal generation and profitability")

def main():
    """Run TiRex directional signal debugging."""
    debug_tirex_directional_signals()

if __name__ == "__main__":
    main()