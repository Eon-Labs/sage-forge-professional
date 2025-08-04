#!/usr/bin/env python3
"""
Test TiRex directly with the exact format from their documentation to isolate the issue.
"""

import torch
import numpy as np
from rich.console import Console

console = Console()

def test_tirex_direct():
    """Test TiRex with exact format from documentation."""
    console.print("🧪 Testing TiRex Direct with Documentation Examples")
    console.print("=" * 60)
    
    try:
        from tirex import load_model, ForecastModel
        
        # Load model
        console.print("🔄 Loading TiRex model...")
        model: ForecastModel = load_model("NX-AI/TiRex")
        console.print("✅ Model loaded successfully")
        
        # Test 1: Exact format from README
        console.print("\n🧪 Test 1: Random data (exact format from README)")
        data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
        console.print(f"Input shape: {data.shape}, dtype: {data.dtype}")
        
        try:
            forecast = model.forecast(context=data, prediction_length=64)
            console.print(f"✅ SUCCESS: Forecast shape: {forecast[0].shape}")
        except Exception as e:
            console.print(f"❌ FAILED: {e}")
        
        # Test 2: Single time series (1D)
        console.print("\n🧪 Test 2: Single time series (1D)")
        data_1d = torch.rand(128)
        console.print(f"Input shape: {data_1d.shape}, dtype: {data_1d.dtype}")
        
        try:
            forecast = model.forecast(context=data_1d, prediction_length=1)
            console.print(f"✅ SUCCESS: Forecast shape: {forecast[0].shape}")
        except Exception as e:
            console.print(f"❌ FAILED: {e}")
        
        # Test 3: Real price data format (what we're using)
        console.print("\n🧪 Test 3: Real price data (our format)")
        # Generate realistic price data like ours
        base_price = 66000.0
        prices = []
        for i in range(128):
            noise = np.random.normal(0, 100)  # $100 volatility
            price = base_price + noise + i * 10  # Slight trend
            prices.append(price)
        
        price_tensor = torch.tensor(prices, dtype=torch.float32)
        console.print(f"Input shape: {price_tensor.shape}, dtype: {price_tensor.dtype}")
        console.print(f"Price range: ${prices[0]:.2f} - ${prices[-1]:.2f}")
        
        try:
            forecast = model.forecast(context=price_tensor, prediction_length=1)
            console.print(f"✅ SUCCESS: Forecast shape: {forecast[0].shape}")
            console.print(f"🎯 Forecast values: {forecast[0].squeeze()}")
            return True
        except Exception as e:
            console.print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Normalized price data
        console.print("\n🧪 Test 4: Normalized price data")
        normalized_prices = (np.array(prices) - np.mean(prices)) / np.std(prices)
        norm_tensor = torch.tensor(normalized_prices, dtype=torch.float32)
        console.print(f"Input shape: {norm_tensor.shape}, dtype: {norm_tensor.dtype}")
        console.print(f"Normalized range: {normalized_prices.min():.3f} - {normalized_prices.max():.3f}")
        
        try:
            forecast = model.forecast(context=norm_tensor, prediction_length=1)
            console.print(f"✅ SUCCESS: Forecast shape: {forecast[0].shape}")
            return True
        except Exception as e:
            console.print(f"❌ FAILED: {e}")
        
        return False
        
    except Exception as e:
        console.print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tirex_direct()
    if success:
        console.print("\n🎉 TiRex is working! Issue is in our implementation.")
    else:
        console.print("\n💥 TiRex has fundamental issues - need to investigate further.")