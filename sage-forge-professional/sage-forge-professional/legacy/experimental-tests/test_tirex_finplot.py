#!/usr/bin/env python3
"""
Quick TiRex Signal Test with FinPlot Visualization
==================================================

Simple script to test TiRex signal generation and finplot visualization.
"""

import sys
from pathlib import Path
import os

# Add SAGE-Forge to path
current_dir = Path(__file__).parent
sage_src = current_dir / "src"
sys.path.append(str(sage_src))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console

# Set environment for headless operation
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

try:
    import finplot as fplt
    finplot_available = True
    print("✅ FinPlot imported successfully")
except ImportError as e:
    print(f"❌ FinPlot import failed: {e}")
    finplot_available = False

from sage_forge.models.tirex_model import TiRexModel

console = Console()

def create_sample_data(days=2):
    """Create sample OHLCV data for testing."""
    console.print(f"📊 Creating {days} days of sample OHLCV data...")
    
    # Generate realistic Bitcoin-like price data
    base_price = 67000
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='1min'
    )
    
    # Random walk with Bitcoin-like characteristics
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, len(timestamps))  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Simple OHLC generation
        high = close * (1 + abs(np.random.normal(0, 0.0005)))
        low = close * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': max(open_price, close, high),
            'low': min(open_price, close, low),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    console.print(f"✅ Created {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def test_tirex_signals(df):
    """Test TiRex model signal generation."""
    console.print("🤖 Testing TiRex signal generation...")
    
    try:
        # Initialize TiRex model (correct parameters)
        tirex = TiRexModel(
            model_name="NX-AI/TiRex",
            prediction_length=1
        )
        
        console.print("✅ TiRex model initialized")
        
        # Process bars and collect signals
        signals = []
        long_signals = []
        short_signals = []
        
        for idx, row in df.iterrows():
            # Create NautilusTrader Bar-compatible object
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.identifiers import InstrumentId
            from nautilus_trader.model.objects import Price, Quantity
            from decimal import Decimal
            
            # Create minimal Bar object for testing
            try:
                # Add bar to TiRex model
                tirex.add_bar(None)  # We'll create a simple test
                
                # Try to get prediction (simplified)
                prediction = tirex.predict()
                if prediction:
                    # Determine signal direction from prediction
                    # This is simplified - real logic would be in strategy
                    direction = 1 if prediction.median > row['close'] else -1
                    
                    signals.append({
                        'timestamp': row['timestamp'],
                        'price': row['close'],
                        'signal': direction,
                        'confidence': prediction.confidence
                    })
                    
                    if direction > 0:
                        long_signals.append((row['timestamp'], row['close']))
                    else:
                        short_signals.append((row['timestamp'], row['close']))
                        
            except Exception as bar_error:
                # Skip this bar if processing fails
                continue
        
        console.print(f"✅ Generated {len(signals)} total signals")
        console.print(f"   📈 LONG signals: {len(long_signals)}")
        console.print(f"   📉 SHORT signals: {len(short_signals)}")
        
        return signals, long_signals, short_signals
        
    except Exception as e:
        console.print(f"❌ TiRex signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

def create_finplot_chart(df, long_signals, short_signals):
    """Create finplot chart with signals."""
    if not finplot_available:
        console.print("❌ FinPlot not available, skipping visualization")
        return
    
    console.print("📊 Creating FinPlot visualization...")
    
    try:
        # Prepare data for finplot
        df_plot = df.set_index('timestamp')
        
        # Create main price chart
        ax = fplt.create_plot('TiRex Signal Test', rows=1)
        
        # Plot candlesticks
        fplt.candlestick_ochl(df_plot[['open', 'close', 'high', 'low']], ax=ax)
        
        # Plot LONG signals (green up arrows)
        if long_signals:
            long_times, long_prices = zip(*long_signals)
            fplt.plot(long_times, long_prices, ax=ax, style='^', color='green', legend='LONG')
        
        # Plot SHORT signals (red down arrows)  
        if short_signals:
            short_times, short_prices = zip(*short_signals)
            fplt.plot(short_times, short_prices, ax=ax, style='v', color='red', legend='SHORT')
        
        console.print(f"✅ FinPlot chart created with {len(long_signals)} LONG and {len(short_signals)} SHORT signals")
        
        # Try to show (will fail in headless mode but chart is created)
        try:
            fplt.show()
            console.print("✅ FinPlot window opened")
        except Exception as show_error:
            console.print(f"⚠️ FinPlot show failed (expected in headless mode): {show_error}")
            console.print("📊 Chart created successfully but cannot display in headless environment")
            
    except Exception as e:
        console.print(f"❌ FinPlot chart creation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    console.print("🚀 TiRex + FinPlot Test Starting...\n")
    
    # Step 1: Create test data
    df = create_sample_data(days=2)
    
    # Step 2: Test TiRex signals
    signals, long_signals, short_signals = test_tirex_signals(df)
    
    # Step 3: Create finplot visualization
    if signals:
        create_finplot_chart(df, long_signals, short_signals)
        
        # Summary
        console.print("\n📊 Test Summary:")
        console.print(f"   • Data points: {len(df)}")
        console.print(f"   • Total signals: {len(signals)}")
        console.print(f"   • LONG signals: {len(long_signals)}")
        console.print(f"   • SHORT signals: {len(short_signals)}")
        
        if signals:
            avg_confidence = np.mean([s['confidence'] for s in signals])
            console.print(f"   • Average confidence: {avg_confidence:.3f}")
            
        console.print("✅ TiRex + FinPlot test completed successfully!")
    else:
        console.print("❌ No signals generated - check TiRex configuration")

if __name__ == "__main__":
    main()