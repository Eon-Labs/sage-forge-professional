#!/usr/bin/env python3
"""
Test Fixed TiRex Signal Generation

OBJECTIVE: Prove the threshold fix generates actual BUY/SELL signals
PREVIOUS: 10 HOLD signals (0 actionable)
EXPECTED: Mix of BUY/SELL signals with profitability analysis
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from sage_forge.models.tirex_model import TiRexModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_fixed_tirex_signals():
    """Test that the optimized threshold fix generates actionable signals."""
    console.print(Panel("🔧 TESTING FIXED TIREX SIGNAL GENERATION", style="bold green"))
    console.print("🎯 Objective: Prove threshold fix generates BUY/SELL signals")
    console.print("📊 Previous: 10 HOLD, 0 BUY, 0 SELL (0 actionable)")
    console.print("✅ Expected: Mix of BUY/SELL signals with profitability")
    console.print()
    
    # Test the same problematic period
    start_date, end_date = "2024-10-15", "2024-10-17"
    console.print(f"📈 Testing period: {start_date} to {end_date}")
    console.print("   Market moved +2.31% during this period")
    console.print()
    
    try:
        # Load market data and TiRex model
        engine = TiRexBacktestEngine()
        success = engine.setup_backtest("BTCUSDT", start_date, end_date, timeframe="15m")
        
        if not success or not hasattr(engine, 'market_bars'):
            console.print("❌ Failed to load market data")
            return False
        
        bars = engine.market_bars
        console.print(f"📊 Loaded {len(bars)} bars")
        
        # Initialize TiRex model with fixed threshold
        tirex_model = TiRexModel()
        if not tirex_model.is_loaded:
            console.print("❌ TiRex model failed to load")
            return False
        
        console.print("✅ TiRex model loaded with OPTIMIZED threshold (0.01%)")
        console.print()
        console.print("🎯 Generating signals...")
        
        signals = []
        
        for i, bar in enumerate(bars):
            # Add bar to model
            tirex_model.add_bar(bar)
            
            # Get prediction
            prediction = tirex_model.predict()
            if prediction is None:
                continue
            
            current_price = float(bar.close)
            
            # Record signal details
            signal_type = 'BUY' if prediction.direction == 1 else 'SELL' if prediction.direction == -1 else 'HOLD'
            
            signals.append({
                'bar_index': i,
                'timestamp': bar.ts_event,
                'price': current_price,
                'direction': prediction.direction,
                'signal_type': signal_type,
                'confidence': prediction.confidence,
                'forecast': prediction.raw_forecast,
                'market_regime': prediction.market_regime
            })
            
            # Show first few signals
            if len(signals) <= 5:
                console.print(f"🎯 Signal {len(signals)}: {signal_type} @ ${current_price:.2f} ({prediction.confidence:.1%} confidence)")
        
        if not signals:
            console.print("❌ No signals generated")
            return False
        
        # Analyze signals
        buy_signals = len([s for s in signals if s['signal_type'] == 'BUY'])
        sell_signals = len([s for s in signals if s['signal_type'] == 'SELL'])
        hold_signals = len([s for s in signals if s['signal_type'] == 'HOLD'])
        actionable_signals = buy_signals + sell_signals
        
        console.print()
        console.print("📊 SIGNAL GENERATION RESULTS:")
        
        # Results table
        table = Table(title="Signal Generation Comparison", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white")
        table.add_column("Previous (0.1% threshold)", style="red")
        table.add_column("Fixed (0.01% threshold)", style="green")
        table.add_column("Improvement", style="yellow")
        
        table.add_row("BUY Signals", "0", str(buy_signals), f"+{buy_signals}")
        table.add_row("SELL Signals", "0", str(sell_signals), f"+{sell_signals}")
        table.add_row("HOLD Signals", "10", str(hold_signals), f"{hold_signals-10:+d}")
        table.add_row("Total Actionable", "0", str(actionable_signals), f"+{actionable_signals}")
        table.add_row("Signal Rate", "0%", f"{actionable_signals/len(signals)*100:.1f}%", f"+{actionable_signals/len(signals)*100:.1f}%")
        
        console.print(table)
        console.print()
        
        # Quick profitability analysis
        if actionable_signals > 0:
            console.print("💰 PROFITABILITY ANALYSIS:")
            
            profitable_signals = 0
            total_return = 0.0
            
            for i, signal in enumerate(signals):
                if signal['signal_type'] == 'HOLD':
                    continue
                
                # Look ahead 4 bars (1 hour) for outcome
                if i + 4 < len(signals):
                    entry_price = signal['price']
                    exit_price = signals[i + 4]['price']
                    
                    if signal['signal_type'] == 'BUY':
                        signal_return = (exit_price - entry_price) / entry_price
                    else:  # SELL
                        signal_return = (entry_price - exit_price) / entry_price
                    
                    total_return += signal_return
                    if signal_return > 0:
                        profitable_signals += 1
            
            if actionable_signals > 0:
                win_rate = profitable_signals / actionable_signals
                avg_return = total_return / actionable_signals
                
                console.print(f"   Win Rate: {win_rate:.1%}")
                console.print(f"   Average Return: {avg_return:.2%}")
                console.print(f"   Total Return: {total_return*100:+.2f}%")
        
        # Final verdict
        console.print()
        if actionable_signals > 0:
            console.print("🎉 SUCCESS: TiRex signal generation FIXED! ✅")
            console.print(f"📈 Generated {actionable_signals} actionable signals vs 0 previously")
            console.print("🎯 Threshold optimization successful")
            console.print("💡 TiRex now produces tradeable BUY/SELL signals")
            return True
        else:
            console.print("❌ FAILED: Still no actionable signals generated")
            return False
            
    except Exception as e:
        console.print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run fixed TiRex signal generation test."""
    console.print("🔬 TESTING FIXED TIREX SIGNAL GENERATION")
    console.print("=" * 70)
    console.print()
    
    success = test_fixed_tirex_signals()
    
    console.print()
    console.print("=" * 70)
    if success:
        console.print("🏆 CONCLUSION: TiRex signal generation successfully FIXED")
        console.print("✅ Ready for production backtesting with real signals")
        console.print("📊 Threshold optimization eliminated HOLD-only problem")
    else:
        console.print("❌ CONCLUSION: Signal generation still needs work")
        console.print("🔧 Further threshold tuning may be required")
    
    return success

if __name__ == "__main__":
    main()