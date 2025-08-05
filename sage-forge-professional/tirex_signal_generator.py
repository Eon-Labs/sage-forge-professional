#!/usr/bin/env python3
"""
🦖 TiRex Signal Generator - Evolutionary Implementation

This implementation represents the current evolutionary state of TiRex signal generation,
incorporating lessons learned from previous approaches while maintaining architectural integrity.

Evolutionary improvements:
- Native sequence length compliance (128 bars)
- Strategic state management between market contexts
- Diverse market regime sampling
- Computational efficiency through proper resource utilization
- Balanced signal generation across market conditions

Legacy reference: See legacy/tirex-evolution/ for historical implementations
"""

import sys
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

def load_tirex_model():
    """Load TiRex model with current evolutionary configuration."""
    try:
        from sage_forge.models.tirex_model import TiRexModel
        console.print("🤖 Loading TiRex model (evolutionary configuration)...")
        
        # Initialize with standard configuration
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        
        console.print("✅ TiRex model loaded successfully")
        console.print(f"   Device: {tirex.device}")
        console.print(f"   Native sequence length: {tirex.input_processor.sequence_length}")
        console.print(f"   Buffer maxlen: {tirex.input_processor.price_buffer.maxlen}")
        console.print(f"   Architecture: Native xLSTM compliance")
        
        return tirex
        
    except Exception as e:
        console.print(f"❌ Failed to load TiRex model: {e}")
        return None

def load_market_data():
    """Load market data for signal generation."""
    try:
        from sage_forge.data.manager import ArrowDataManager
        
        console.print("📊 Loading market data...")
        data_manager = ArrowDataManager()
        
        # Use extended period for diverse market condition sampling
        end_time = datetime(2024, 10, 17, 0, 0, 0)
        start_time = datetime(2024, 10, 1, 0, 0, 0)
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            start_time=start_time,
            end_time=end_time,
            timeframe="15m"
        )
        
        if df is None or df.height == 0:
            console.print("❌ No market data available")
            return None
            
        console.print(f"✅ Loaded {df.height} BTCUSDT market bars")
        return df.to_pandas()
        
    except Exception as e:
        console.print(f"❌ Failed to load market data: {e}")
        return None

def generate_tirex_signals(tirex_model, market_data):
    """
    Generate TiRex signals using evolutionary approach.
    
    This implementation combines insights from previous iterations:
    - Uses native sequence length (128 bars) for architectural compliance
    - Clears model state between market contexts to prevent bias accumulation
    - Samples diverse market regimes for balanced signal generation
    - Maintains temporal ordering validation for security
    - Achieves computational efficiency through proper resource utilization
    
    Args:
        tirex_model: Loaded TiRex model instance
        market_data: DataFrame containing OHLCV market data
        
    Returns:
        List of signal dictionaries with prediction metadata
    """
    console.print("🦖 Generating TiRex signals (evolutionary approach)...")
    
    if tirex_model is None or market_data is None:
        console.print("❌ Cannot generate signals without model or data")
        return []
    
    try:
        # Import NT objects for proper Bar creation
        from nautilus_trader.model.data import Bar, BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        
        instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
        bar_spec = BarSpecification(step=15, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
        bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)
        
        # Use native sequence length from model architecture
        context_window_size = tirex_model.input_processor.sequence_length
        available_data = len(market_data)
        
        console.print(f"📊 Using native context window: {context_window_size} bars")
        
        if available_data < context_window_size + 1:
            console.print(f"⚠️ Insufficient data: {available_data} bars < {context_window_size + 1} required")
            return []
        
        # Calculate strategic sampling for diverse market conditions
        # Balance between comprehensive coverage and computational efficiency
        num_desired_windows = 20  # Reasonable number for diverse market regime coverage
        max_possible_windows = available_data - context_window_size
        num_windows = min(num_desired_windows, max_possible_windows)
        
        if num_windows <= 1:
            stride = 1
            sample_points = [0]
        else:
            stride = max_possible_windows // (num_windows - 1)
            sample_points = [i * stride for i in range(num_windows)]
            # Ensure we don't exceed data bounds
            sample_points = [sp for sp in sample_points if sp + context_window_size <= available_data]
        
        console.print(f"📊 Sampling {len(sample_points)} diverse market contexts (stride={stride})")
        console.print(f"📈 Market regime coverage across {len(sample_points)} different periods")
        
        signals = []
        prediction_count = 0
        
        for i, start_idx in enumerate(sample_points):
            if i % 5 == 0:
                console.print(f"   Processing context {i+1}/{len(sample_points)}...")
            
            end_idx = start_idx + context_window_size
            context_data = market_data.iloc[start_idx:end_idx]
            
            # Strategic state management between market contexts
            # This prevents bias accumulation while maintaining temporal validation
            tirex_model.input_processor.price_buffer.clear()
            tirex_model.input_processor.timestamp_buffer.clear()
            # Reset timestamp for new context while preserving validation capability
            tirex_model.input_processor.last_timestamp = None
            
            # Feed market context data to model
            for _, row in context_data.iterrows():
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
                tirex_model.add_bar(bar)
            
            # Generate prediction from current market context
            prediction = tirex_model.predict()
            prediction_count += 1
            
            if prediction is not None and prediction.direction != 0:
                # Use next bar after context window for signal placement
                signal_bar_index = end_idx
                if signal_bar_index < len(market_data):
                    signal_bar = market_data.iloc[signal_bar_index]
                    signal_type = "BUY" if prediction.direction > 0 else "SELL"
                    
                    signals.append({
                        'timestamp': signal_bar['timestamp'],
                        'price': float(signal_bar['close']),
                        'signal': signal_type,
                        'confidence': prediction.confidence,
                        'volatility_forecast': prediction.volatility_forecast,
                        'raw_forecast': prediction.raw_forecast.tolist() if hasattr(prediction.raw_forecast, 'tolist') else float(prediction.raw_forecast),
                        'bar_index': signal_bar_index,
                        'context_window_start': start_idx,
                        'context_window_end': end_idx,
                        'context_period': f"{context_data.iloc[0]['timestamp'].strftime('%m-%d %H:%M')} - {context_data.iloc[-1]['timestamp'].strftime('%m-%d %H:%M')}",
                        'model_source': 'TIREX_EVOLUTIONARY',
                        'architecture_compliance': 'NATIVE_XLSTM'
                    })
        
        console.print(f"✅ Generated {len(signals)} TiRex signals")
        console.print(f"   Total predictions: {prediction_count}")
        console.print(f"   Signal rate: {len(signals)/prediction_count*100:.1f}%")
        console.print(f"   Architecture: Native xLSTM compliance")
        console.print(f"   Efficiency: No computational waste")
        
        return signals
        
    except Exception as e:
        console.print(f"❌ TiRex signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_signal_results(signals):
    """Analyze the generated TiRex signals."""
    console.print(Panel("📊 TiRex Signal Analysis", style="green bold"))
    
    if not signals:
        console.print("❌ No signals generated")
        return
    
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']
    
    # Results table
    table = Table(title="🦖 TiRex Model Results (Evolutionary Implementation)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Notes", style="yellow")
    
    table.add_row("Implementation", "Evolutionary", "Current development stage")
    table.add_row("Architecture", "Native xLSTM", "128-bar sequence length compliance")
    table.add_row("BUY Signals", str(len(buy_signals)), "Long predictions")
    table.add_row("SELL Signals", str(len(sell_signals)), "Short predictions")
    table.add_row("Signal Balance", f"{len(buy_signals)} BUY / {len(sell_signals)} SELL", "Market regime diversity")
    
    # Signal diversity assessment
    if len(buy_signals) > 0 and len(sell_signals) > 0:
        table.add_row("Market Coverage", "Diverse", "Captures different market conditions")
        diversity_score = min(len(buy_signals), len(sell_signals)) / max(len(buy_signals), len(sell_signals))
        table.add_row("Diversity Score", f"{diversity_score:.2f}", "Balance between signal types")
    elif len(buy_signals) > 0:
        table.add_row("Market Coverage", "Bullish Period", "Market trend indication")
    else:
        table.add_row("Market Coverage", "Bearish Period", "Market trend indication")
    
    if signals:
        confidences = [s['confidence'] for s in signals]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        table.add_row("Avg Confidence", f"{avg_confidence:.1%}", "Model certainty level")
        table.add_row("Confidence Range", f"{min_confidence:.1%} - {max_confidence:.1%}", "Prediction quality variation")
    
    console.print(table)
    
    # Implementation benefits
    console.print("\n🔧 Evolutionary Implementation Benefits:")
    console.print("• Native TiRex architecture compliance")
    console.print("• Strategic state management prevents bias accumulation")
    console.print("• Diverse market regime sampling")
    console.print("• Computational efficiency through proper resource utilization")
    console.print("• Temporal ordering validation maintained")
    console.print("• Balanced signal generation capability")
    
    # Context diversity analysis
    if signals:
        console.print(f"\n📊 Market Context Analysis:")
        console.print(f"   Analyzed {len(set(s['context_period'] for s in signals))} different market periods")
        
        # Show sample of different periods analyzed
        periods = list(set(s['context_period'] for s in signals))[:5]
        for period in periods:
            period_signals = [s for s in signals if s['context_period'] == period]
            signal_types = [s['signal'] for s in period_signals]
            console.print(f"   • {period}: {', '.join(signal_types)}")

def setup_finplot_theme():
    """Setup professional theme for visualization."""
    try:
        import finplot as fplt
        import pyqtgraph as pg
        
        console.print("🎨 Setting up visualization theme...")
        
        # Professional dark theme
        fplt.foreground = "#f0f6fc"
        fplt.background = "#0d1117"
        
        pg.setConfigOptions(
            foreground=fplt.foreground,
            background=fplt.background,
            antialias=True,
        )
        
        # Chart styling
        fplt.odd_plot_background = fplt.background
        fplt.candle_bull_color = "#26d0ce"
        fplt.candle_bear_color = "#f85149" 
        fplt.candle_bull_body_color = "#238636"
        fplt.candle_bear_body_color = "#da3633"
        fplt.volume_bull_color = "#26d0ce40"
        fplt.volume_bear_color = "#f8514940"
        fplt.cross_hair_color = "#58a6ff"
        
        return True
        
    except ImportError:
        console.print("⚠️ FinPlot not available - analysis only mode")
        return False

def visualize_signals(market_data, signals):
    """Create visualization of TiRex signals."""
    if not setup_finplot_theme():
        return None
        
    try:
        import finplot as fplt
        
        console.print("📈 Creating signal visualization...")
        
        df_indexed = market_data.set_index('timestamp')
        
        # Create plot
        ax, ax2 = fplt.create_plot('🦖 TiRex Signals - Evolutionary Implementation', rows=2, maximize=True)
        
        # Plot OHLC data
        fplt.candlestick_ochl(df_indexed[['open', 'close', 'high', 'low']], ax=ax)
        
        # Plot volume if available
        if 'volume' in df_indexed.columns:
            fplt.volume_ocv(df_indexed[['open', 'close', 'volume']], ax=ax2)
        
        # Separate signals by type
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        
        console.print(f"🎯 Plotting {len(buy_signals)} BUY and {len(sell_signals)} SELL signals")
        
        # Calculate positioning offsets
        price_range = df_indexed['high'].max() - df_indexed['low'].min()
        bar_ranges = df_indexed['high'] - df_indexed['low']
        avg_bar_range = bar_ranges.mean()
        
        triangle_offset_ratio = 0.02  # Reasonable positioning
        
        # Plot BUY signals (green triangles below bars)
        if buy_signals:
            buy_times = []
            buy_prices_offset = []
            
            for signal in buy_signals:
                signal_timestamp = signal['timestamp']
                
                # Find matching bar
                matching_bars = df_indexed[df_indexed.index == signal_timestamp]
                if len(matching_bars) > 0:
                    bar_data = matching_bars.iloc[0]
                    exact_bar_time = matching_bars.index[0]
                else:
                    time_diffs = abs(df_indexed.index - signal_timestamp)
                    nearest_idx = time_diffs.argmin()
                    bar_data = df_indexed.iloc[nearest_idx]
                    exact_bar_time = df_indexed.index[nearest_idx]
                
                # Position triangle below bar
                low_price = bar_data['low']
                bar_range = bar_data['high'] - bar_data['low']
                offset_price = low_price - bar_range * triangle_offset_ratio
                
                buy_times.append(exact_bar_time)
                buy_prices_offset.append(offset_price)
            
            fplt.plot(buy_times, buy_prices_offset, ax=ax, color='#00ff00', style='^', width=4, legend='TiRex BUY')
        
        # Plot SELL signals (red triangles above bars)
        if sell_signals:
            sell_times = []
            sell_prices_offset = []
            
            for signal in sell_signals:
                signal_timestamp = signal['timestamp']
                
                # Find matching bar
                matching_bars = df_indexed[df_indexed.index == signal_timestamp]
                if len(matching_bars) > 0:
                    bar_data = matching_bars.iloc[0]
                    exact_bar_time = matching_bars.index[0]
                else:
                    time_diffs = abs(df_indexed.index - signal_timestamp)
                    nearest_idx = time_diffs.argmin()
                    bar_data = df_indexed.iloc[nearest_idx]
                    exact_bar_time = df_indexed.index[nearest_idx]
                
                # Position triangle above bar
                high_price = bar_data['high']
                bar_range = bar_data['high'] - bar_data['low']
                offset_price = high_price + bar_range * triangle_offset_ratio
                
                sell_times.append(exact_bar_time)
                sell_prices_offset.append(offset_price)
            
            fplt.plot(sell_times, sell_prices_offset, ax=ax, color='#ff0000', style='v', width=4, legend='TiRex SELL')
        
        # Add confidence labels
        for signal in buy_signals + sell_signals:
            signal_timestamp = signal['timestamp']
            matching_bars = df_indexed[df_indexed.index == signal_timestamp]
            
            if len(matching_bars) > 0:
                exact_bar_time = matching_bars.index[0]
                bar_data = matching_bars.iloc[0]
            else:
                time_diffs = abs(df_indexed.index - signal_timestamp)
                nearest_idx = time_diffs.argmin()
                exact_bar_time = df_indexed.index[nearest_idx]
                bar_data = df_indexed.iloc[nearest_idx]
            
            # Position confidence label
            conf_text = f"{signal['confidence']:.0%}"
            bar_range = bar_data['high'] - bar_data['low']
            
            if signal['signal'] == 'BUY':
                text_price = bar_data['low'] - bar_range * (triangle_offset_ratio + 0.01)
            else:
                text_price = bar_data['high'] + bar_range * (triangle_offset_ratio + 0.01)
            
            fplt.add_text((exact_bar_time, text_price), conf_text, ax=ax, color='#cccccc')
        
        # Set title
        ax.setTitle('🦖 TiRex Signals - Evolutionary Implementation (Native xLSTM Architecture)')
        
        return ax, ax2
        
    except Exception as e:
        console.print(f"❌ Visualization failed: {e}")
        return None

def main():
    """Main function for TiRex signal generation."""
    console.print(Panel("🦖 TiRex Signal Generator - Evolutionary Implementation", style="bold green"))
    console.print("🔬 Current evolutionary state incorporating architectural compliance and strategic state management")
    console.print("📚 Legacy implementations archived in legacy/tirex-evolution/ for reference")
    console.print()
    
    try:
        # Load TiRex model
        tirex_model = load_tirex_model()
        if tirex_model is None:
            console.print("❌ Cannot proceed without TiRex model")
            return
        
        # Load market data
        market_data = load_market_data()
        if market_data is None:
            console.print("❌ Cannot proceed without market data")
            return
        
        # Generate signals using evolutionary approach
        signals = generate_tirex_signals(tirex_model, market_data)
        
        # Analyze results
        analyze_signal_results(signals)
        
        # Create visualization if signals exist
        if signals:
            console.print("\n🎨 Creating signal visualization...")
            visualization = visualize_signals(market_data, signals)
            
            if visualization:
                ax, ax2 = visualization
                console.print("✅ TiRex signal visualization created successfully")
                console.print("🖱️ Explore evolutionary TiRex predictions on market data")
                
                # Show the plot
                try:
                    import finplot as fplt
                    fplt.show()
                except ImportError:
                    console.print("⚠️ FinPlot not available for display")
            else:
                console.print("⚠️ Visualization creation failed")
        else:
            console.print("\n⚠️ No signals to visualize")
            
    except Exception as e:
        console.print(f"❌ TiRex signal generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()