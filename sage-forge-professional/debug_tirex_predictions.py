#!/usr/bin/env python3
"""
Debug TiRex predictions to see exactly what the model is generating.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.models.tirex_model import TiRexModel, TIREX_AVAILABLE
from sage_forge.data.manager import ArrowDataManager
from rich.console import Console
from datetime import datetime

console = Console()

def debug_tirex_predictions():
    """Debug what TiRex is actually predicting."""
    console.print("🔍 Debugging TiRex Predictions - Raw Model Output")
    console.print("=" * 60)
    
    if not TIREX_AVAILABLE:
        console.print("❌ TiRex not available")
        return
    
    try:
        # Initialize TiRex model
        console.print("🔄 Loading TiRex model...")
        tirex = TiRexModel(model_name="NX-AI/TiRex", prediction_length=1)
        
        if not tirex.is_loaded:
            console.print("❌ TiRex model failed to load")
            return
            
        console.print("✅ TiRex model loaded successfully")
        
        # Get real market data with known significant movement
        console.print("\n📊 Fetching market data with +2.31% movement (Oct 15-17, 2024)...")
        data_manager = ArrowDataManager()
        
        df = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            timeframe="15m", 
            start_time=datetime(2024, 10, 15),
            end_time=datetime(2024, 10, 17)
        )
        
        # Convert to NT bars
        bars = data_manager.to_nautilus_bars(df, instrument_id="BTCUSDT-PERP.BINANCE")
        console.print(f"📈 Loaded {len(bars)} bars")
        
        # Show price movement
        first_bar = bars[0]
        last_bar = bars[-1]
        price_change = float(last_bar.close) - float(first_bar.open)
        change_pct = (price_change / float(first_bar.open)) * 100
        console.print(f"💹 Price movement: ${float(first_bar.open):.2f} → ${float(last_bar.close):.2f} ({change_pct:+.2f}%)")
        
        # Feed bars to TiRex and capture predictions
        console.print("\n🤖 Processing bars through TiRex...")
        predictions_made = 0
        signals_generated = 0
        
        for i, bar in enumerate(bars):
            # Add bar to model
            tirex.add_bar(bar)
            
            # Try to get prediction
            prediction = tirex.predict()
            
            if prediction is not None:
                predictions_made += 1
                
                # Debug prediction details
                current_price = float(bar.close)
                console.print(f"\n📊 Bar {i+1}: ${current_price:.2f}")
                console.print(f"   Direction: {prediction.direction} (-1=Bear, 0=Neutral, +1=Bull)")
                console.print(f"   Confidence: {prediction.confidence:.4f} (need ≥0.6)")
                console.print(f"   Raw forecast: {prediction.raw_forecast}")
                console.print(f"   Volatility: {prediction.volatility_forecast:.6f}")
                console.print(f"   Processing time: {prediction.processing_time_ms:.1f}ms")
                
                # Check if this would generate a signal
                if prediction.confidence >= 0.6:  # Default threshold
                    signals_generated += 1
                    console.print(f"   🚨 SIGNAL GENERATED! Direction={prediction.direction}, Confidence={prediction.confidence:.3f}")
                else:
                    console.print(f"   📊 No signal (confidence {prediction.confidence:.3f} < 0.6)")
                    
                # Test with lower thresholds
                if prediction.confidence >= 0.3:
                    console.print(f"   ⚡ Would signal at 30% confidence")
                if prediction.confidence >= 0.1:
                    console.print(f"   ⚡ Would signal at 10% confidence")
        
        # Summary
        console.print(f"\n📈 TIREX ANALYSIS SUMMARY:")
        console.print(f"   Total bars processed: {len(bars)}")
        console.print(f"   Predictions made: {predictions_made}")
        console.print(f"   Signals at 60% threshold: {signals_generated}")
        console.print(f"   Market movement: {change_pct:+.2f}%")
        
        if predictions_made == 0:
            console.print("❌ NO PREDICTIONS MADE - Model input issue")
        elif signals_generated == 0:
            console.print("⚠️ PREDICTIONS MADE BUT NO SIGNALS - Threshold too high")
        else:
            console.print("✅ SIGNALS GENERATED - Model working correctly")
            
        # Get model performance stats
        stats = tirex.get_performance_stats()
        if stats:
            console.print(f"\n⚡ Model Performance:")
            console.print(f"   Avg inference: {stats.get('avg_inference_time_ms', 0):.1f}ms")
            console.print(f"   Total predictions: {stats.get('total_predictions', 0)}")
            
    except Exception as e:
        console.print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tirex_predictions()