#!/usr/bin/env python3
"""
🧪 DEBUG COMPARISON TEST: Original vs Extended TiRex Scripts
Test both approaches with detailed logging to validate adversarial audit findings.
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

def test_original_approach():
    """Test the original TiRex approach with debug logging."""
    console.print(Panel("🔄 TESTING ORIGINAL APPROACH", style="blue bold"))
    
    results = {
        'success': False,
        'predictions': 0,
        'signals': 0,
        'errors': [],
        'execution_time': 0,
        'buffer_behavior': {},
        'sequence_length_used': None
    }
    
    start_time = time.time()
    
    try:
        # Load model
        from sage_forge.models.tirex_model import TiRexModel
        from sage_forge.data.manager import ArrowDataManager
        from nautilus_trader.model.data import Bar, BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        
        console.print("🔄 Loading TiRex model...")
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        
        # Record model architecture details
        results['sequence_length_used'] = tirex.input_processor.sequence_length
        results['buffer_maxlen'] = tirex.input_processor.price_buffer.maxlen
        
        console.print(f"✅ Model loaded - sequence_length: {results['sequence_length_used']}")
        console.print(f"📊 Buffer maxlen: {results['buffer_maxlen']}")
        
        # Load market data
        console.print("📊 Loading market data...")
        data_manager = ArrowDataManager()
        end_time = datetime(2024, 10, 17, 0, 0, 0)
        start_time_data = datetime(2024, 10, 1, 0, 0, 0)
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            start_time=start_time_data,
            end_time=end_time,
            timeframe="15m"
        )
        
        if df is None or df.height == 0:
            results['errors'].append("No market data available")
            return results
            
        market_data = df.to_pandas()
        console.print(f"📈 Loaded {len(market_data)} bars")
        
        # Set up NT objects
        instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
        bar_spec = BarSpecification(step=15, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
        bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)
        
        # ORIGINAL APPROACH: Feed all data sequentially (like original script)
        console.print("🔄 Feeding ALL data to model sequentially...")
        buffer_states = []
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Record buffer state before adding
            buffer_size_before = len(tirex.input_processor.price_buffer)
            
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
            
            # Record buffer state after adding
            buffer_size_after = len(tirex.input_processor.price_buffer)
            
            # Log key buffer state changes
            if i % 100 == 0 or i < 10 or i >= len(market_data) - 10:
                buffer_states.append({
                    'bar_index': i,
                    'buffer_before': buffer_size_before,
                    'buffer_after': buffer_size_after,
                    'timestamp': row['timestamp'],
                    'price': float(row['close'])
                })
                console.print(f"  Bar {i}: buffer {buffer_size_before}→{buffer_size_after}, price={row['close']}")
        
        results['buffer_behavior'] = {
            'final_buffer_size': len(tirex.input_processor.price_buffer),
            'buffer_states_sampled': buffer_states,
            'overflow_occurred': len(market_data) > tirex.input_processor.price_buffer.maxlen
        }
        
        # Make predictions (limited to 3 for testing)
        console.print("🦖 Making predictions from final model state...")
        predictions = []
        
        for i in range(min(3, 10)):  # Test with 3 predictions
            prediction = tirex.predict()
            results['predictions'] += 1
            
            if prediction is not None:
                console.print(f"  Prediction {i+1}: direction={prediction.direction}, confidence={prediction.confidence:.1%}")
                if prediction.direction != 0:
                    results['signals'] += 1
                    predictions.append({
                        'direction': prediction.direction,
                        'confidence': prediction.confidence,
                        'volatility': prediction.volatility_forecast
                    })
            else:
                console.print(f"  Prediction {i+1}: None returned")
        
        results['predictions_data'] = predictions
        results['success'] = True
        
    except Exception as e:
        console.print(f"❌ Original approach failed: {e}")
        results['errors'].append(str(e))
        traceback.print_exc()
    
    results['execution_time'] = time.time() - start_time
    return results

def test_extended_approach():
    """Test the extended TiRex approach with debug logging."""
    console.print(Panel("🔄 TESTING EXTENDED APPROACH", style="yellow bold"))
    
    results = {
        'success': False,
        'predictions': 0,
        'signals': 0,
        'errors': [],
        'execution_time': 0,
        'buffer_behavior': {},
        'sequence_length_used': None,
        'windows_processed': 0
    }
    
    start_time = time.time()
    
    try:
        # Load model
        from sage_forge.models.tirex_model import TiRexModel
        from sage_forge.data.manager import ArrowDataManager
        from nautilus_trader.model.data import Bar, BarType, BarSpecification
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        
        console.print("🔄 Loading TiRex model...")
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        
        # Record model architecture details
        results['sequence_length_used'] = tirex.input_processor.sequence_length
        results['buffer_maxlen'] = tirex.input_processor.price_buffer.maxlen
        
        console.print(f"✅ Model loaded - sequence_length: {results['sequence_length_used']}")
        console.print(f"📊 Buffer maxlen: {results['buffer_maxlen']}")
        
        # Load market data
        console.print("📊 Loading market data...")
        data_manager = ArrowDataManager()
        end_time = datetime(2024, 10, 17, 0, 0, 0)
        start_time_data = datetime(2024, 10, 1, 0, 0, 0)
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            start_time=start_time_data,
            end_time=end_time,
            timeframe="15m"
        )
        
        if df is None or df.height == 0:
            results['errors'].append("No market data available")
            return results
            
        market_data = df.to_pandas()
        console.print(f"📈 Loaded {len(market_data)} bars")
        
        # Set up NT objects
        instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
        bar_spec = BarSpecification(step=15, aggregation=BarAggregation.MINUTE, price_type=PriceType.LAST)
        bar_type = BarType(instrument_id=instrument_id, bar_spec=bar_spec)
        
        # EXTENDED APPROACH: Manual windowing with 512 bars (CRITICAL BUG TEST)
        min_context_window = 512  # This is the bug! Model expects 128
        available_data = len(market_data)
        
        console.print(f"🚨 TESTING BUG: Using {min_context_window} bars when model expects {results['sequence_length_used']}")
        
        if available_data < min_context_window + 1:
            results['errors'].append(f"Insufficient data: {available_data} < {min_context_window + 1}")
            return results
        
        # Calculate stride (limit to 3 windows for testing)
        stride = max(1, (available_data - min_context_window) // 3)  # Test with 3 windows only
        sample_points = list(range(0, available_data - min_context_window, stride))[:3]
        
        console.print(f"📊 Testing {len(sample_points)} windows (stride={stride})")
        
        predictions = []
        buffer_clearing_logs = []
        
        for i, start_idx in enumerate(sample_points):
            console.print(f"\n🔄 Processing window {i+1}/{len(sample_points)} (start_idx={start_idx})")
            
            end_idx = start_idx + min_context_window
            context_data = market_data.iloc[start_idx:end_idx]
            
            # EXTENDED APPROACH: Clear buffers (test if this causes issues)
            buffer_before_clear = {
                'price_buffer_len': len(tirex.input_processor.price_buffer),
                'timestamp_buffer_len': len(tirex.input_processor.timestamp_buffer),
                'last_timestamp': tirex.input_processor.last_timestamp
            }
            
            console.print(f"  📊 Before clear: price_buffer={buffer_before_clear['price_buffer_len']}, timestamp_buffer={buffer_before_clear['timestamp_buffer_len']}")
            
            # Clear state (test for issues)
            tirex.input_processor.price_buffer.clear()
            tirex.input_processor.timestamp_buffer.clear()
            tirex.input_processor.last_timestamp = None
            
            buffer_after_clear = {
                'price_buffer_len': len(tirex.input_processor.price_buffer),
                'timestamp_buffer_len': len(tirex.input_processor.timestamp_buffer),
                'last_timestamp': tirex.input_processor.last_timestamp
            }
            
            console.print(f"  📊 After clear: price_buffer={buffer_after_clear['price_buffer_len']}, timestamp_buffer={buffer_after_clear['timestamp_buffer_len']}")
            
            buffer_clearing_logs.append({
                'window': i,
                'before': buffer_before_clear,
                'after': buffer_after_clear
            })
            
            # Feed exactly min_context_window bars (512 - THE BUG!)
            console.print(f"  🔄 Feeding {len(context_data)} bars to model...")
            bars_fed = 0
            
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
                bars_fed += 1
            
            console.print(f"  ✅ Fed {bars_fed} bars, buffer now: {len(tirex.input_processor.price_buffer)}")
            
            # Try to make prediction
            try:
                prediction = tirex.predict()
                results['predictions'] += 1
                results['windows_processed'] += 1
                
                if prediction is not None:
                    console.print(f"  🦖 Prediction: direction={prediction.direction}, confidence={prediction.confidence:.1%}")
                    if prediction.direction != 0:
                        results['signals'] += 1
                        predictions.append({
                            'window': i,
                            'direction': prediction.direction,
                            'confidence': prediction.confidence,
                            'volatility': prediction.volatility_forecast,
                            'bars_fed': bars_fed,
                            'buffer_size': len(tirex.input_processor.price_buffer)
                        })
                else:
                    console.print(f"  ❌ Prediction returned None")
                    
            except Exception as pred_error:
                console.print(f"  🚨 Prediction failed: {pred_error}")
                results['errors'].append(f"Window {i} prediction failed: {pred_error}")
        
        results['buffer_behavior'] = {
            'clearing_logs': buffer_clearing_logs,
            'final_buffer_size': len(tirex.input_processor.price_buffer),
            'context_window_size': min_context_window,
            'model_expected_size': results['sequence_length_used']
        }
        results['predictions_data'] = predictions
        results['success'] = True
        
    except Exception as e:
        console.print(f"❌ Extended approach failed: {e}")
        results['errors'].append(str(e))
        traceback.print_exc()
    
    results['execution_time'] = time.time() - start_time
    return results

def compare_results(original_results, extended_results):
    """Compare and analyze the results."""
    console.print(Panel("📊 COMPARATIVE ANALYSIS", style="magenta bold"))
    
    # Results table
    table = Table(title="🔍 Approach Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Original Approach", style="blue")
    table.add_column("Extended Approach", style="yellow")
    table.add_column("Analysis", style="green")
    
    table.add_row(
        "Success",
        "✅" if original_results['success'] else "❌",
        "✅" if extended_results['success'] else "❌",
        "Both approaches work" if original_results['success'] and extended_results['success'] else "Issues detected"
    )
    
    table.add_row(
        "Predictions Made",
        str(original_results['predictions']),
        str(extended_results['predictions']),
        f"Original: {original_results['predictions']}, Extended: {extended_results['predictions']}"
    )
    
    table.add_row(
        "Signals Generated",
        str(original_results['signals']),
        str(extended_results['signals']),
        f"Signal rates: Orig={original_results['signals']}/{original_results['predictions']}, Ext={extended_results['signals']}/{extended_results['predictions']}"
    )
    
    table.add_row(
        "Execution Time",
        f"{original_results['execution_time']:.2f}s",
        f"{extended_results['execution_time']:.2f}s",
        f"Extended is {extended_results['execution_time']/original_results['execution_time']:.1f}x slower" if original_results['execution_time'] > 0 else "N/A"
    )
    
    table.add_row(
        "Sequence Length",
        str(original_results.get('sequence_length_used', 'N/A')),
        str(extended_results.get('sequence_length_used', 'N/A')),
        "Models use same architecture" if original_results.get('sequence_length_used') == extended_results.get('sequence_length_used') else "Architecture mismatch"
    )
    
    table.add_row(
        "Buffer Handling",
        "Native sliding window",
        f"Manual windowing with {extended_results['buffer_behavior'].get('context_window_size', 'N/A')} bars",
        "🚨 BUG CONFIRMED" if extended_results['buffer_behavior'].get('context_window_size', 0) > extended_results.get('sequence_length_used', 0) else "Compatible"
    )
    
    table.add_row(
        "Errors",
        str(len(original_results['errors'])),
        str(len(extended_results['errors'])),
        "Extended has more issues" if len(extended_results['errors']) > len(original_results['errors']) else "Similar error rates"
    )
    
    console.print(table)
    
    # Error analysis
    if original_results['errors'] or extended_results['errors']:
        console.print("\n🚨 ERROR ANALYSIS:")
        if original_results['errors']:
            console.print("Original errors:")
            for error in original_results['errors']:
                console.print(f"  - {error}")
        if extended_results['errors']:
            console.print("Extended errors:")
            for error in extended_results['errors']:
                console.print(f"  - {error}")
    
    # Critical findings
    console.print(Panel("🎯 CRITICAL FINDINGS", style="red bold"))
    
    findings = []
    
    # Check sequence length bug
    if extended_results.get('buffer_behavior', {}).get('context_window_size', 0) > extended_results.get('sequence_length_used', 0):
        findings.append("🚨 CRITICAL BUG: Extended script feeds 512 bars to model expecting 128 bars")
    
    # Check buffer clearing effectiveness
    if extended_results.get('buffer_behavior', {}).get('clearing_logs'):
        findings.append("✅ Buffer clearing works but may be redundant (deque handles overflow)")
    
    # Check computational efficiency
    if extended_results['execution_time'] > original_results['execution_time'] * 2:
        findings.append(f"🚨 EFFICIENCY LOSS: Extended approach is {extended_results['execution_time']/original_results['execution_time']:.1f}x slower")
    
    # Check prediction quality
    if original_results['predictions'] > 0 and extended_results['predictions'] > 0:
        orig_signal_rate = original_results['signals'] / original_results['predictions']
        ext_signal_rate = extended_results['signals'] / extended_results['predictions']
        if abs(orig_signal_rate - ext_signal_rate) > 0.2:
            findings.append(f"🤔 SIGNAL RATE DIFFERENCE: Original {orig_signal_rate:.1%} vs Extended {ext_signal_rate:.1%}")
    
    for finding in findings:
        console.print(finding)
    
    if not findings:
        console.print("✅ No critical issues detected in comparison")

def main():
    """Run comprehensive debug comparison test."""
    console.print(Panel("🧪 DEBUG COMPARISON TEST: Original vs Extended TiRex Scripts", style="bold green"))
    console.print("🎯 Objective: Validate adversarial audit findings with actual execution")
    console.print("🔍 Focus: Sequence length bug, buffer clearing, computational efficiency")
    console.print()
    
    # Test original approach
    original_results = test_original_approach()
    console.print()
    
    # Test extended approach
    extended_results = test_extended_approach()
    console.print()
    
    # Compare results
    compare_results(original_results, extended_results)

if __name__ == "__main__":
    main()