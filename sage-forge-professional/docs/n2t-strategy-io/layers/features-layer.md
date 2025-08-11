### FEATURES Layer — Technical Indicators & Derived Intelligence

**Processing Role**: Post-TiRex feature engineering combining [PREDICTIONS](./predictions-layer.md) with [CONTEXT](./context-layer.md) data  
**Purpose**: Bridge between TiRex probabilistic forecasts and trading decision logic  
**Integration**: Primary input for [SIGNALS layer](./signals-layer.md) trading rules

---

#### Executive Summary

**Function**: Transform raw TiRex predictions and market data into actionable trading intelligence  
**Stability**: ✅ **Stable** - Standard technical analysis with TiRex enhancement  
**Columns**: 5 essential features combining traditional indicators with TiRex predictions  
**Innovation**: TiRex-derived edge calculation providing unique alpha generation capability

---

#### Complete FEATURES Column Specifications

| Column       | Type  | Definition          | Formula / Pseudocode                  | Tools     | Lineage                  | LeakageGuard |
| ------------ | ----- | ------------------- | ------------------------------------- | --------- | ------------------------ | ------------ |
| **edge_1**   | float | 1-step TiRex edge   | `tirex_mean_p50[t+1] - close[t]`      | —         | from PREDICTIONS/CONTEXT | roll ≤ t     |
| **atr_14**   | float | Average True Range  | `ATR(14)`                             | ta-lib    | CONTEXT (OHLC)           | roll ≤ t     |
| **ma_20**    | float | Moving average      | `SMA(close, 20)`                      | pandas_ta | CONTEXT (close)          | roll ≤ t     |
| **rsi_14**   | float | Momentum oscillator | `RSI(close, 14)`                      | pandas_ta | CONTEXT (close)          | roll ≤ t     |
| **pos_size** | float | Position sizing     | `risk_budget / (atr_14 * tick_value)` | —         | from atr_14              | roll ≤ t     |

---

#### TiRex-Enhanced Feature Engineering

##### Core Innovation: TiRex Edge Calculation

```python
# Revolutionary alpha generation from TiRex predictions
def calculate_tirex_edge(tirex_mean_p50, current_close):
    """
    Core alpha signal: TiRex 1-step ahead prediction vs current price

    Positive edge_1 = TiRex predicts price increase
    Negative edge_1 = TiRex predicts price decrease
    Magnitude = Predicted move size in price units
    """
    edge_1 = tirex_mean_p50[..., 0] - current_close  # Next period prediction - current

    # Enhanced edge features (optional)
    edge_normalized = edge_1 / current_close          # Percentage terms
    edge_volatility_adjusted = edge_1 / atr_14        # Risk-adjusted edge
    edge_confidence = calculate_prediction_confidence(tirex_quantiles)

    return edge_1, edge_normalized, edge_volatility_adjusted, edge_confidence

# Usage in strategy logic
if edge_1 > lambda_threshold * atr_14:
    signal = "BUY"  # TiRex predicts profitable upward move
elif edge_1 < -lambda_threshold * atr_14:
    signal = "SELL"  # TiRex predicts profitable downward move
```

##### Traditional Technical Indicators (TiRex-Enhanced)

```python
import talib
import pandas_ta as ta

# ATR for volatility-adjusted position sizing
def calculate_enhanced_atr(context_data, period=14):
    """ATR with TiRex prediction integration for dynamic risk adjustment"""
    traditional_atr = talib.ATR(context_data['high'], context_data['low'],
                               context_data['close'], timeperiod=period)

    # TiRex-enhanced: Forward-looking volatility estimate
    predicted_volatility = (tirex_q_p90 - tirex_q_p10) / 1.645  # Implied volatility
    hybrid_atr = 0.7 * traditional_atr + 0.3 * predicted_volatility

    return hybrid_atr

# Moving Average with TiRex trend confirmation
def calculate_trend_ma(close_prices, predictions, period=20):
    """MA enhanced with TiRex directional bias"""
    traditional_ma = close_prices.rolling(window=period).mean()

    # TiRex trend confirmation
    prediction_trend = (tirex_mean_p50[..., -1] - tirex_mean_p50[..., 0]) / len(tirex_mean_p50)
    trend_adjustment = 0.05 * prediction_trend  # Small adjustment based on predicted direction

    enhanced_ma = traditional_ma + trend_adjustment
    return enhanced_ma

# RSI with TiRex momentum validation
def calculate_enhanced_rsi(close_prices, predictions, period=14):
    """RSI with TiRex momentum confirmation"""
    traditional_rsi = talib.RSI(close_prices, timeperiod=period)

    # TiRex momentum validation
    predicted_momentum = tirex_mean_p50[..., 2] - tirex_mean_p50[..., 0]  # 2-step momentum
    momentum_signal = np.sign(predicted_momentum)

    # RSI adjustment for TiRex momentum agreement/disagreement
    rsi_adjustment = momentum_signal * 2.0  # Small RSI bias
    enhanced_rsi = np.clip(traditional_rsi + rsi_adjustment, 0, 100)

    return enhanced_rsi
```

---

#### Advanced Feature Engineering

##### TiRex Uncertainty Features

```python
# Extract uncertainty intelligence from TiRex predictions
def calculate_uncertainty_features(tirex_quantiles):
    """Advanced features from TiRex uncertainty quantification"""

    # Confidence intervals
    ci_80 = tirex_quantiles[..., 7] - tirex_quantiles[..., 1]  # 80% CI (0.2 to 0.8)
    ci_60 = tirex_quantiles[..., 6] - tirex_quantiles[..., 2]  # 60% CI (0.3 to 0.7)

    # Prediction skewness
    median = tirex_quantiles[..., 4]
    upper_tail = tirex_quantiles[..., 8] - median
    lower_tail = median - tirex_quantiles[..., 0]
    skewness = (upper_tail - lower_tail) / (upper_tail + lower_tail)

    # Regime detection
    uncertainty_regime = ci_80 / median  # Relative uncertainty
    high_uncertainty = uncertainty_regime > uncertainty_regime.rolling(50).quantile(0.8)

    return {
        'confidence_80': ci_80,
        'confidence_60': ci_60,
        'prediction_skewness': skewness,
        'uncertainty_regime': uncertainty_regime,
        'high_uncertainty_flag': high_uncertainty
    }
```

##### Multi-Timeframe Features

```python
# Combine multiple prediction horizons
def calculate_multi_horizon_features(context_data, guardian):
    """Features combining short and long-term TiRex predictions"""

    # Generate predictions for multiple horizons
    short_term, _ = guardian.safe_forecast(context_data, prediction_length=1)   # 5min
    medium_term, _ = guardian.safe_forecast(context_data, prediction_length=6)  # 30min
    long_term, _ = guardian.safe_forecast(context_data, prediction_length=12)   # 1hour

    # Cross-timeframe consistency
    directional_agreement = (
        np.sign(short_term[..., 4] - context_data[-1]) ==
        np.sign(long_term[..., 4] - context_data[-1])
    ).float()

    # Trend acceleration
    short_edge = short_term[..., 4] - context_data[-1]
    long_edge = long_term[..., 4] - context_data[-1]
    trend_acceleration = (long_edge / 12) - (short_edge / 1)  # Acceleration measure

    return {
        'timeframe_agreement': directional_agreement,
        'trend_acceleration': trend_acceleration,
        'short_term_edge': short_edge,
        'long_term_edge': long_edge
    }
```

---

#### Position Sizing Intelligence

##### Dynamic Position Sizing with TiRex Integration

```python
def calculate_optimal_position_size(edge_1, atr_14, tirex_quantiles,
                                  risk_budget=0.02, base_capital=100000):
    """
    Advanced position sizing combining Kelly criterion with TiRex uncertainty
    """

    # Traditional ATR-based sizing
    traditional_size = risk_budget * base_capital / (atr_14 * 100)  # Risk per unit

    # TiRex-enhanced sizing factors
    edge_magnitude = abs(edge_1) / atr_14                    # Risk-adjusted edge
    prediction_confidence = calculate_prediction_confidence(tirex_quantiles)
    uncertainty_discount = 1.0 - (prediction_confidence - 0.5) * 0.3  # Size adjustment

    # Kelly fraction approximation
    win_prob = prediction_confidence
    avg_win = (tirex_quantiles[..., 7] - tirex_quantiles[..., 4]).mean()  # Average win
    avg_loss = (tirex_quantiles[..., 4] - tirex_quantiles[..., 1]).mean() # Average loss

    kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
    kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Conservative cap at 25%

    # Combined position sizing
    tirex_enhanced_size = traditional_size * (1 + kelly_fraction) * uncertainty_discount

    return {
        'traditional_size': traditional_size,
        'tirex_enhanced_size': tirex_enhanced_size,
        'kelly_fraction': kelly_fraction,
        'confidence_adjustment': uncertainty_discount,
        'edge_magnitude': edge_magnitude
    }
```

---

#### Integration Patterns

##### FEATURES → SIGNALS Pipeline

```python
# Feature preparation for trading signals
def prepare_signal_features(context_data, predictions_data):
    """Complete feature engineering pipeline for signal generation"""

    # Core features
    features = {
        'edge_1': predictions_data['tirex_mean_p50'][0] - context_data['close'][-1],
        'atr_14': talib.ATR(context_data['high'], context_data['low'],
                           context_data['close'], timeperiod=14)[-1],
        'ma_20': context_data['close'].rolling(20).mean()[-1],
        'rsi_14': talib.RSI(context_data['close'], timeperiod=14)[-1]
    }

    # TiRex-enhanced features
    uncertainty_features = calculate_uncertainty_features(predictions_data['quantiles'])
    multi_timeframe = calculate_multi_horizon_features(context_data, guardian)
    position_sizing = calculate_optimal_position_size(
        features['edge_1'], features['atr_14'], predictions_data['quantiles']
    )

    # Combine all features
    features.update(uncertainty_features)
    features.update(multi_timeframe)
    features['pos_size'] = position_sizing['tirex_enhanced_size']

    return features
```

##### Real-time Feature Updates

```python
# Efficient incremental feature calculation
class TiRexFeatureEngine:
    def __init__(self, lookback_periods=500):
        self.lookback = lookback_periods
        self.feature_cache = {}

    def update_features(self, new_context, new_predictions):
        """Incremental feature updates for real-time trading"""

        # Update only features that need recalculation
        self.feature_cache['edge_1'] = self._update_edge(new_predictions)
        self.feature_cache['atr_14'] = self._update_atr(new_context)
        self.feature_cache['ma_20'] = self._update_ma(new_context)
        self.feature_cache['rsi_14'] = self._update_rsi(new_context)
        self.feature_cache['pos_size'] = self._update_position_size()

        return self.feature_cache

    def _update_edge(self, predictions):
        # Most critical feature - always recalculate
        return predictions['tirex_mean_p50'][0] - self.last_close

    # Additional incremental update methods...
```

---

#### Quality Monitoring

##### Feature Validation

```python
def validate_features(features_dict):
    """Comprehensive feature quality validation"""

    validation_results = {}

    # Range checks
    validation_results['edge_1_reasonable'] = abs(features_dict['edge_1']) < 10.0  # Price units
    validation_results['atr_positive'] = features_dict['atr_14'] > 0
    validation_results['rsi_valid'] = 0 <= features_dict['rsi_14'] <= 100
    validation_results['pos_size_reasonable'] = 0 < features_dict['pos_size'] < 1e6

    # Consistency checks
    validation_results['features_complete'] = all(
        not np.isnan(v) for v in features_dict.values() if isinstance(v, (int, float))
    )

    return validation_results
```

---

#### Advanced Applications

##### Feature Importance Analysis

```python
# Analyze which features contribute most to trading performance
def analyze_feature_importance(features_history, returns_history):
    """Determine which features provide most alpha"""

    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    # Prepare feature matrix
    feature_df = pd.DataFrame(features_history)
    target_returns = np.array(returns_history)

    # Train importance model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_df, target_returns)

    # Feature importance ranking
    importance_scores = pd.Series(rf.feature_importances_, index=feature_df.columns)

    return importance_scores.sort_values(ascending=False)
```

---

#### Conclusion

The FEATURES layer provides **intelligent bridge** between TiRex predictions and trading decisions:

**Core Innovation**: `edge_1` calculation providing direct TiRex alpha signal  
**Traditional Enhancement**: ATR, MA, RSI enhanced with TiRex intelligence  
**Advanced Capabilities**: Uncertainty quantification, multi-timeframe analysis, dynamic position sizing  
**Integration Ready**: Optimized for [SIGNALS layer](./signals-layer.md) consumption

**Performance Impact**: Proper FEATURES engineering can amplify the **2-4x improvement** from [TOKENIZED layer optimization](./tokenized-layer.md) by extracting maximum intelligence from enhanced TiRex predictions.

**Status**: ✅ **Production Ready** - Comprehensive feature engineering combining traditional technical analysis with TiRex probabilistic forecasting intelligence.

---

[← Back to Index](../strategy-io-contract.md#layer-navigation-tirex-native) | [Next: SIGNALS Layer →](./signals-layer.md)
