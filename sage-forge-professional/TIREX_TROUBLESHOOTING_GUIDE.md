# TiRex Integration Troubleshooting Guide

## 🔍 **Quick Diagnosis Flowchart**

```
TiRex Not Working?
├── No predictions made?
│   ├── Check sequence length ≥ 128 bars
│   ├── Verify model.is_loaded = True
│   └── Check input_processor.get_model_input() returns tensor
├── Tensor shape errors?
│   ├── "inhomogeneous shape" → Use 1D tensor [128] not 2D [1,128]
│   ├── "invalid index" → Unpack (quantiles, means) tuple
│   └── "0-dimensional" → Handle scalar arrays properly
├── No data found in NT?
│   ├── Check timestamps are historical not current
│   ├── Verify catalog directory structure
│   └── Match BacktestDataConfig time range
└── Predictions but no signals?
    ├── Lower confidence threshold from 60%
    ├── Check direction calculation logic
    └── Verify signal generation parameters
```

## ⚡ **Common Errors & Instant Fixes**

### Error: `setting an array element with a sequence`
**Cause**: Wrong data format to TiRex  
**Fix**: Use 1D tensor
```python
# WRONG
input_tensor = torch.tensor(prices).unsqueeze(0)  # [1, 128]
# RIGHT  
input_tensor = torch.tensor(prices)  # [128]
```

### Error: `invalid index to scalar variable`
**Cause**: Incorrect forecast result parsing  
**Fix**: Unpack tuple properly
```python
# WRONG
forecast = model.forecast(context=data)
mean = forecast.mean
# RIGHT
quantiles, means = model.forecast(context=data)
mean = means.squeeze().cpu().numpy()
```

### Error: No predictions generated
**Cause**: Insufficient sequence length  
**Fix**: Wait for 128+ bars
```python
if len(self.price_buffer) < self.sequence_length:  # 128
    return None  # Need more data
```

### Error: NT "No data found"
**Cause**: Timestamp mismatch  
**Fix**: Use historical timestamps
```python
# Check your data timestamps match backtest date range
print(f"Data timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Backtest range: {start_date} to {end_date}")
```

## 🧪 **Debugging Commands**

### Test TiRex Directly
```python
from tirex import load_model
model = load_model("NX-AI/TiRex")
data = torch.rand(128)  # Should work
forecast = model.forecast(context=data, prediction_length=1)
print(f"Success: {forecast[0].shape}")  # Should show torch.Size([1, 1, 9])
```

### Check Data Pipeline
```python
# 1. Verify DSM data
df = data_manager.fetch_real_market_data(symbol="BTCUSDT", timeframe="15m")
print(f"Timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")

# 2. Verify NT bars
bars = data_manager.to_nautilus_bars(df)
print(f"Bars: {len(bars)}, First: {bars[0].ts_event}")

# 3. Test TiRex input
tirex = TiRexModel()
for bar in bars:
    tirex.add_bar(bar)
    if len(tirex.input_processor.price_buffer) >= 128:
        input_tensor = tirex.input_processor.get_model_input()
        print(f"Input shape: {input_tensor.shape}")  # Should be [128]
        break
```

### Monitor Predictions
```python
# Add to TiRexModel.predict()
logger.info(f"Prediction: direction={direction}, confidence={confidence:.3f}")
logger.info(f"Forecast: {forecast_value:.2f}, Current: {current_price:.2f}")
```

## 📊 **Performance Benchmarks**

### Expected Performance (15m data)
- **Inference Time**: 8-12ms per prediction
- **Memory Usage**: <100MB for 128-bar context
- **Prediction Rate**: 1 prediction per bar after warmup
- **Typical Confidence**: 5-25% for real market data
- **Signal Rate**: 0-3 signals per 100 predictions at 60% threshold

### Red Flags
- **Inference >100ms**: Model loading issue
- **No predictions after 200 bars**: Data format problem
- **All confidence >90%**: Calculation error
- **All confidence <1%**: Threshold or uncertainty issue

## 🔧 **Configuration Tuning**

### Confidence Thresholds
```python
# Conservative (fewer, higher quality signals)
min_confidence = 0.6  # 60%

# Balanced (moderate signal frequency) 
min_confidence = 0.2  # 20%

# Aggressive (more signals, lower quality)
min_confidence = 0.1  # 10%
```

### Sequence Length
```python
# Default (good balance)
sequence_length = 128

# Longer context (potentially better predictions)
sequence_length = 256  # Slower inference

# Shorter context (faster warmup)
sequence_length = 64   # May hurt accuracy
```

## 🚨 **Emergency Recovery**

### TiRex Won't Load
1. Check CUDA availability: `torch.cuda.is_available()`
2. Set fallback: `os.environ['TIREX_NO_CUDA'] = '1'`
3. Verify model path: `"NX-AI/TiRex"` exactly

### Complete Prediction Failure
1. Test with synthetic data: `torch.rand(128)`
2. Check model.is_loaded status
3. Verify sequence length requirements
4. Reset input processor buffer

### Performance Degradation
1. Check GPU memory usage
2. Monitor batch size (should be 1)
3. Verify data types (float32)
4. Clear prediction history periodically

## ✅ **Success Indicators**

You know TiRex is working correctly when:
- [x] Model loads without CUDA errors
- [x] Predictions start after 128+ bars
- [x] Confidence values in 0.01-0.30 range
- [x] Inference time <20ms consistently
- [x] Forecast values close to actual prices
- [x] Direction classifications make sense
- [x] Some predictions meet signal thresholds

**Keep this guide handy for future TiRex debugging!**