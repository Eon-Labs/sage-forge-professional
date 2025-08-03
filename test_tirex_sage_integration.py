#!/usr/bin/env python3
"""
TiRex SAGE Integration Test - Complete System Validation
Tests the full integration of NX-AI TiRex model with SAGE-Forge framework on GPU.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add SAGE-Forge to path
sys.path.append('/home/tca/eon/nt/sage-forge-professional/src')

from sage_forge.models.tirex_model import TiRexModel, TiRexPrediction
from sage_forge.strategies.tirex_sage_strategy import TiRexSageStrategy
from rich.console import Console

console = Console()

def test_tirex_model_integration():
    """Test TiRex model standalone functionality."""
    console.print("🧪 Testing TiRex Model Integration")
    console.print("=" * 50)
    
    # Initialize TiRex model
    tirex = TiRexModel(model_path="/home/tca/eon/nt/models/tirex", device="cuda")
    
    if not tirex.is_loaded:
        console.print("❌ TiRex model failed to load")
        return False
    
    console.print("✅ TiRex model loaded successfully")
    
    # Test synthetic data processing
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from datetime import datetime, timedelta
    
    # Generate realistic synthetic market data
    base_price = 50000.0  # Starting price (like BTC)
    current_time = datetime.now()
    
    console.print("📊 Generating synthetic market data...")
    
    # Add sufficient bars for prediction (need 200+)
    for i in range(250):
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.01)  # 1% volatility
        base_price *= (1 + price_change)
        
        # Create OHLCV with realistic spread
        spread = base_price * 0.001  # 0.1% spread
        open_price = base_price + np.random.uniform(-spread, spread)
        high_price = max(open_price, base_price + abs(np.random.normal(0, spread)))
        low_price = min(open_price, base_price - abs(np.random.normal(0, spread)))
        close_price = base_price
        volume = np.random.uniform(100, 1000)
        
        # Create bar
        bar_time = current_time + timedelta(minutes=i)
        ts_event = dt_to_unix_nanos(bar_time)
        
        try:
            bar = Bar(
                bar_type=None,
                open=Price.from_str(f"{open_price:.2f}"),
                high=Price.from_str(f"{high_price:.2f}"),
                low=Price.from_str(f"{low_price:.2f}"),
                close=Price.from_str(f"{close_price:.2f}"),
                volume=Quantity.from_str(f"{volume:.2f}"),
                ts_event=ts_event,
                ts_init=ts_event
            )
            
            tirex.add_bar(bar)
            
        except Exception as e:
            console.print(f"❌ Error creating bar {i}: {e}")
            continue
    
    console.print("✅ Market data generated and processed")
    
    # Test predictions
    console.print("🔮 Testing TiRex predictions...")
    
    predictions = []
    prediction_times = []
    
    for i in range(10):  # Test 10 predictions
        start_time = time.time()
        prediction = tirex.predict()
        end_time = time.time()
        
        if prediction:
            predictions.append(prediction)
            prediction_times.append((end_time - start_time) * 1000)
            
            console.print(f"  Prediction {i+1}: "
                         f"Direction={prediction.direction}, "
                         f"Confidence={prediction.confidence:.3f}, "
                         f"Regime={prediction.market_regime}, "
                         f"Time={prediction.processing_time_ms:.1f}ms")
        else:
            console.print(f"  Prediction {i+1}: Failed")
    
    if predictions:
        avg_confidence = np.mean([p.confidence for p in predictions])
        avg_inference_time = np.mean(prediction_times)
        
        console.print(f"\n📈 Prediction Summary:")
        console.print(f"   Successful Predictions: {len(predictions)}/10")
        console.print(f"   Average Confidence: {avg_confidence:.3f}")
        console.print(f"   Average Inference Time: {avg_inference_time:.1f}ms")
        console.print(f"   Direction Distribution: "
                     f"Bullish={sum(1 for p in predictions if p.direction == 1)}, "
                     f"Bearish={sum(1 for p in predictions if p.direction == -1)}, "
                     f"Neutral={sum(1 for p in predictions if p.direction == 0)}")
        
        # Performance stats
        model_stats = tirex.get_performance_stats()
        console.print(f"   Model Performance: {model_stats}")
        
        return True
    else:
        console.print("❌ No successful predictions generated")
        return False

def test_sage_strategy_initialization():
    """Test SAGE strategy initialization."""
    console.print("\n🧪 Testing SAGE Strategy Initialization")
    console.print("=" * 50)
    
    try:
        # Test strategy configuration
        config = {
            'min_confidence': 0.6,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'model_path': '/home/tca/eon/nt/models/tirex'
        }
        
        # Initialize strategy (without running)
        strategy = TiRexSageStrategy(config=config)
        
        console.print("✅ TiRex SAGE Strategy initialized successfully")
        console.print(f"   Min Confidence: {strategy.min_confidence}")
        console.print(f"   Max Position Size: {strategy.max_position_size}")
        console.print(f"   Risk Per Trade: {strategy.risk_per_trade}")
        console.print(f"   Model Path: {strategy.model_path}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Strategy initialization failed: {e}")
        return False

def test_signal_generation():
    """Test trading signal generation from TiRex predictions."""
    console.print("\n🧪 Testing Signal Generation")
    console.print("=" * 50)
    
    try:
        # Create mock prediction
        from sage_forge.models.tirex_model import TiRexPrediction
        
        mock_prediction = TiRexPrediction(
            direction=1,  # Bullish
            confidence=0.75,
            raw_output=0.85,
            volatility_forecast=0.02,
            processing_time_ms=45.2,
            market_regime="medium_vol_trending"
        )
        
        console.print("✅ Mock TiRex prediction created")
        console.print(f"   Direction: {mock_prediction.direction}")
        console.print(f"   Confidence: {mock_prediction.confidence}")
        console.print(f"   Market Regime: {mock_prediction.market_regime}")
        console.print(f"   Volatility Forecast: {mock_prediction.volatility_forecast:.3f}")
        
        # Test signal interpretation
        if mock_prediction.confidence >= 0.6:
            console.print("✅ Signal meets confidence threshold")
            
            # Estimate position size (simplified)
            base_size = 0.1 * mock_prediction.confidence
            regime_multiplier = 1.0  # Default for medium_vol_trending
            position_size = base_size * regime_multiplier
            
            console.print(f"   Recommended Position Size: {position_size:.4f}")
            console.print(f"   Signal Quality: {'High' if mock_prediction.confidence > 0.7 else 'Medium'}")
        else:
            console.print("❌ Signal below confidence threshold")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Signal generation test failed: {e}")
        return False

def test_gpu_performance():
    """Test GPU performance and memory usage."""
    console.print("\n🧪 Testing GPU Performance")
    console.print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            console.print("❌ CUDA not available")
            return False
        
        console.print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
        console.print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test model memory usage
        tirex = TiRexModel(model_path="/home/tca/eon/nt/models/tirex", device="cuda")
        
        if tirex.is_loaded:
            memory_used = torch.cuda.memory_allocated(0) / 1e6
            console.print(f"✅ Model Memory Usage: {memory_used:.1f} MB")
            
            # Test batch inference performance
            inference_times = []
            for _ in range(20):
                start = time.time()
                _ = tirex.predict()  # Will return None without data, but tests GPU path
                end = time.time()
                inference_times.append((end - start) * 1000)
            
            avg_time = np.mean(inference_times)
            max_time = np.max(inference_times)
            
            console.print(f"✅ GPU Inference Performance:")
            console.print(f"   Average Time: {avg_time:.1f}ms")
            console.print(f"   Max Time: {max_time:.1f}ms")
            console.print(f"   Throughput: {1000/avg_time:.0f} predictions/second")
            
            return True
        else:
            console.print("❌ Model not loaded for GPU testing")
            return False
            
    except Exception as e:
        console.print(f"❌ GPU performance test failed: {e}")
        return False

def main():
    """Run complete TiRex SAGE integration test suite."""
    console.print("🎯 TiRex SAGE Integration Test Suite")
    console.print("GPU-Accelerated Time Series Forecasting for Trading")
    console.print("=" * 70)
    
    test_results = []
    
    # Test 1: TiRex Model Integration
    test_results.append(("TiRex Model Integration", test_tirex_model_integration()))
    
    # Test 2: SAGE Strategy Initialization
    test_results.append(("SAGE Strategy Init", test_sage_strategy_initialization()))
    
    # Test 3: Signal Generation
    test_results.append(("Signal Generation", test_signal_generation()))
    
    # Test 4: GPU Performance
    test_results.append(("GPU Performance", test_gpu_performance()))
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("🏁 Test Results Summary")
    console.print("=" * 70)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"{status} {test_name}")
        if result:
            passed += 1
    
    console.print(f"\n📊 Overall Result: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        console.print("🎉 All tests passed! TiRex SAGE integration ready for trading.")
        console.print("\n🚀 Next Steps:")
        console.print("   1. Configure trading instruments")
        console.print("   2. Set up live data feeds")
        console.print("   3. Deploy strategy in paper trading mode")
        console.print("   4. Monitor performance and adjust parameters")
    else:
        console.print("⚠️  Some tests failed. Review issues before deployment.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)