#!/usr/bin/env python3
"""
🔍 Deep Dive Analysis: Why Extended Version Works Better Despite Architectural Violations

Investigation into why original script produces only BUY signals vs extended producing mixed signals.
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import warnings
import numpy as np

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sage_src = current_dir / "src"
sys.path.append(str(sage_src))

warnings.filterwarnings('ignore')
console = Console()

def analyze_original_bias_issue():
    """Deep dive into why original script produces only BUY signals."""
    console.print(Panel("🔍 ANALYZING ORIGINAL SCRIPT BUY BIAS ISSUE", style="red bold"))
    
    try:
        from sage_forge.models.tirex_model import TiRexModel
        from sage_forge.data.manager import ArrowDataManager
        from nautilus_trader.model.data import Bar, BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        
        # Load model and data
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        data_manager = ArrowDataManager()
        
        end_time = datetime(2024, 10, 17, 0, 0, 0)
        start_time_data = datetime(2024, 10, 1, 0, 0, 0)
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            start_time=start_time_data,
            end_time=end_time,
            timeframe="15m"
        )
        market_data = df.to_pandas()
        
        # NT objects setup
        instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
        bar_spec = BarSpecification(step=15, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
        bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)
        
        console.print("🔍 HYPOTHESIS 1: Model gets stuck in final market trend")
        
        # Analyze market trend in the final 128 bars (what model sees)
        final_bars = market_data.tail(128)
        price_changes = final_bars['close'].diff().dropna()
        
        positive_changes = (price_changes > 0).sum()
        negative_changes = (price_changes < 0).sum()
        
        console.print(f"📊 Final 128 bars analysis:")
        console.print(f"  Positive price changes: {positive_changes}")
        console.print(f"  Negative price changes: {negative_changes}")
        console.print(f"  Trend bias: {'BULLISH' if positive_changes > negative_changes else 'BEARISH'}")
        
        # Check overall price trend
        start_price = final_bars.iloc[0]['close'] 
        end_price = final_bars.iloc[-1]['close']
        trend = "UP" if end_price > start_price else "DOWN"
        
        console.print(f"  Overall trend: {start_price:.2f} → {end_price:.2f} ({trend})")
        
        if positive_changes > negative_changes * 1.5:
            console.print("🚨 STRONG BULLISH BIAS in final 128 bars - explains BUY-only predictions!")
        
        console.print("\n🔍 HYPOTHESIS 2: Model state gets biased by sequential feeding")
        
        # Simulate original approach with state tracking
        predictions_by_position = []
        
        # Feed all data sequentially (like original)
        for i, (_, row) in enumerate(market_data.iterrows()):
            ts_ns = dt_to_unix_nanos(row['timestamp'])
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{float(row['open']):.2f}"),
                high=Price.from_str(f"{float(row['high']):.2f}"),
                low=Price.from_str(f"{float(row['low']):.2f}"),
                close=Price.from_str(f"{float(row['close']):.2f}"),
                volume=Quantity.from_str(f"{float(row.get('volume', 1000)):.0f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            tirex.add_bar(bar)
            
            # Make predictions at key points to see when bias starts
            if i in [128, 256, 512, 1024, len(market_data)-1]:
                pred = tirex.predict()
                if pred:
                    predictions_by_position.append({
                        'position': i,
                        'direction': pred.direction,
                        'confidence': pred.confidence,
                        'price': float(row['close']),
                        'timestamp': row['timestamp']
                    })
                    console.print(f"  Position {i}: direction={pred.direction}, confidence={pred.confidence:.1%}, price=${row['close']:.2f}")
        
        # Analyze prediction evolution
        if predictions_by_position:
            directions = [p['direction'] for p in predictions_by_position]
            if all(d > 0 for d in directions):
                console.print("🚨 CONFIRMED: Sequential feeding leads to persistent BUY bias")
            elif all(d < 0 for d in directions):
                console.print("🚨 CONFIRMED: Sequential feeding leads to persistent SELL bias")
            else:
                console.print("✅ Sequential feeding produces mixed predictions")
        
        return {
            'final_trend_bias': 'BULLISH' if positive_changes > negative_changes else 'BEARISH',
            'predictions_by_position': predictions_by_position,
            'trend_ratio': positive_changes / max(negative_changes, 1)
        }
        
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return None

def analyze_extended_version_benefits():
    """Analyze why extended version produces better results."""
    console.print(Panel("🔍 ANALYZING EXTENDED VERSION BENEFITS", style="green bold"))
    
    try:
        from sage_forge.models.tirex_model import TiRexModel
        from sage_forge.data.manager import ArrowDataManager
        from nautilus_trader.model.data import Bar, BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        
        # Load model and data
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        data_manager = ArrowDataManager()
        
        end_time = datetime(2024, 10, 17, 0, 0, 0)
        start_time_data = datetime(2024, 10, 1, 0, 0, 0)
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            start_time=start_time_data,
            end_time=end_time,
            timeframe="15m"
        )
        market_data = df.to_pandas()
        
        # NT objects setup
        instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
        bar_spec = BarSpecification(step=15, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
        bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)
        
        console.print("🔍 EXTENDED APPROACH: Analyzing different time windows")
        
        # Simulate extended approach with different windows
        min_context_window = 128  # Use correct sequence length
        available_data = len(market_data)
        
        if available_data < min_context_window + 1:
            console.print("❌ Insufficient data")
            return None
        
        # Test 5 different windows across the dataset
        stride = (available_data - min_context_window) // 5
        sample_points = [i * stride for i in range(5)]
        
        window_analyses = []
        
        for i, start_idx in enumerate(sample_points):
            end_idx = start_idx + min_context_window
            context_data = market_data.iloc[start_idx:end_idx]
            
            # Analyze this window's characteristics
            price_changes = context_data['close'].diff().dropna()
            positive_changes = (price_changes > 0).sum()
            negative_changes = (price_changes < 0).sum()
            
            window_start_price = context_data.iloc[0]['close']
            window_end_price = context_data.iloc[-1]['close']
            window_trend = "UP" if window_end_price > window_start_price else "DOWN"
            
            # Clear state and feed this window
            tirex.input_processor.price_buffer.clear()
            tirex.input_processor.timestamp_buffer.clear()
            tirex.input_processor.last_timestamp = None
            
            for _, row in context_data.iterrows():
                bar = Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{float(row['open']):.2f}"),
                    high=Price.from_str(f"{float(row['high']):.2f}"),
                    low=Price.from_str(f"{float(row['low']):.2f}"),
                    close=Price.from_str(f"{float(row['close']):.2f}"),
                    volume=Quantity.from_str(f"{float(row.get('volume', 1000)):.0f}"),
                    ts_event=dt_to_unix_nanos(row['timestamp']),
                    ts_init=dt_to_unix_nanos(row['timestamp']),
                )
                tirex.add_bar(bar)
            
            # Make prediction
            prediction = tirex.predict()
            
            window_analysis = {
                'window': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': context_data.iloc[0]['timestamp'],
                'end_time': context_data.iloc[-1]['timestamp'],
                'positive_changes': positive_changes,
                'negative_changes': negative_changes,
                'window_trend': window_trend,
                'start_price': window_start_price,
                'end_price': window_end_price,
                'prediction': prediction.direction if prediction else None,
                'confidence': prediction.confidence if prediction else None
            }
            
            window_analyses.append(window_analysis)
            
            console.print(f"Window {i}: {context_data.iloc[0]['timestamp'].strftime('%m-%d %H:%M')} - {context_data.iloc[-1]['timestamp'].strftime('%m-%d %H:%M')}")
            console.print(f"  Trend: {window_trend} ({window_start_price:.2f} → {window_end_price:.2f})")
            console.print(f"  Changes: +{positive_changes} / -{negative_changes}")
            console.print(f"  Prediction: {prediction.direction if prediction else None} ({prediction.confidence:.1%} confidence)" if prediction else "  Prediction: None")
        
        console.print("\n🎯 KEY INSIGHT: Extended approach captures different market regimes!")
        
        # Analyze diversity of predictions
        predictions = [w['prediction'] for w in window_analyses if w['prediction'] is not None]
        if predictions:
            buy_count = sum(1 for p in predictions if p > 0)
            sell_count = sum(1 for p in predictions if p < 0)
            console.print(f"Prediction diversity: {buy_count} BUY, {sell_count} SELL")
            
            if buy_count > 0 and sell_count > 0:
                console.print("✅ MIXED PREDICTIONS - captures different market conditions")
            else:
                console.print("⚠️ Still biased predictions")
        
        return window_analyses
        
    except Exception as e:
        console.print(f"❌ Extended analysis failed: {e}")
        traceback.print_exc()
        return None

def propose_optimal_solution():
    """Propose the optimal solution combining benefits without violations."""
    console.print(Panel("💡 OPTIMAL SOLUTION DESIGN", style="blue bold"))
    
    console.print("🎯 UNDERSTANDING THE REAL PROBLEM:")
    console.print("1. Original approach gets biased by sequential feeding of trending data")
    console.print("2. Extended approach works by resetting state between different market periods")
    console.print("3. Extended approach accidentally violates architecture (512→128 truncation)")
    console.print("4. The benefit comes from DIVERSITY of market contexts, not buffer violations")
    
    console.print("\n💡 OPTIMAL SOLUTION:")
    console.print("1. Keep the sliding window approach for diverse predictions")
    console.print("2. Use CORRECT sequence length (128 bars, not 512)")
    console.print("3. Keep state clearing between windows (this is the real benefit)")
    console.print("4. Maintain temporal ordering validation for security")
    
    optimal_approach = """
def generate_diverse_tirex_signals(tirex_model, market_data):
    \"\"\"Optimal approach: Diverse windows with correct sequence length.\"\"\"
    
    # Use CORRECT sequence length from model architecture
    min_context_window = tirex_model.input_processor.sequence_length  # 128, not 512!
    available_data = len(market_data)
    
    if available_data < min_context_window + 1:
        return []
    
    # Calculate stride for diverse sampling
    num_windows = min(10, (available_data - min_context_window) // 10)  # Reasonable number
    stride = (available_data - min_context_window) // num_windows if num_windows > 0 else 1
    sample_points = range(0, available_data - min_context_window, stride)
    
    signals = []
    
    for i, start_idx in enumerate(sample_points):
        end_idx = start_idx + min_context_window
        context_data = market_data.iloc[start_idx:end_idx]
        
        # Clear state between windows (THIS IS THE KEY BENEFIT)
        tirex_model.input_processor.price_buffer.clear()
        tirex_model.input_processor.timestamp_buffer.clear()
        # Keep timestamp validation - just reset for new window
        tirex_model.input_processor.last_timestamp = None
        
        # Feed exactly the right amount of data
        for _, row in context_data.iterrows():
            # ... create and add bar ...
            tirex_model.add_bar(bar)
        
        # Generate prediction from this specific market context
        prediction = tirex_model.predict()
        if prediction and prediction.direction != 0:
            signals.append(prediction_to_signal(prediction, context_data.iloc[-1]))
    
    return signals
"""
    
    console.print(optimal_approach)
    
    console.print("\n🔍 WHY THIS WORKS:")
    console.print("✅ Uses correct model architecture (128 bars)")
    console.print("✅ Captures diverse market conditions through windowing")
    console.print("✅ Prevents bias accumulation via state clearing")
    console.print("✅ Maintains temporal ordering validation")
    console.print("✅ Efficient - no wasted computation")
    
    console.print("\n🚨 WHY EXTENDED VERSION WORKS BY ACCIDENT:")
    console.print("• Feeds 512 bars but model only uses last 128 (deque truncation)")
    console.print("• State clearing prevents bias accumulation (real benefit)")
    console.print("• Different windows capture different market regimes (real benefit)")
    console.print("• Works despite architectural violation, not because of it")

def main():
    """Run comprehensive deep dive analysis."""
    console.print(Panel("🔍 DEEP DIVE: Why Extended Version Works Despite Violations", style="bold magenta"))
    console.print("🎯 Investigating the root cause of original BUY bias and extended version benefits")
    console.print()
    
    # Analyze original bias issue
    original_analysis = analyze_original_bias_issue()
    console.print()
    
    # Analyze extended version benefits  
    extended_analysis = analyze_extended_version_benefits()
    console.print()
    
    # Propose optimal solution
    propose_optimal_solution()
    
    console.print(Panel("🎯 CONCLUSION: Extended version works by ACCIDENT, not design", style="bold red"))
    console.print("The real benefits are: 1) State clearing, 2) Diverse windows")
    console.print("The architectural violation (512 bars) is unnecessary and wasteful")
    console.print("Optimal solution: Keep benefits, fix violations")

if __name__ == "__main__":
    main()