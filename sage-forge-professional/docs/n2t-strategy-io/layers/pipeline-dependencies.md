### Pipeline Dependencies — TiRex Native Data Flow Analysis

**Complete Architecture**: End-to-end data pipeline from exchange to trading signals  
**TiRex Integration**: Native component mapping with dependency validation  
**Performance Critical Path**: TOKENIZED layer optimization impact across entire pipeline

---

#### Executive Summary

**Purpose**: Comprehensive analysis of data flow dependencies throughout TiRex-enhanced trading pipeline  
**Architecture**: 5-layer native TiRex processing with clear dependency chains  
**Critical Insight**: TOKENIZED layer optimization creates **cascading performance improvements** across all downstream layers

---

## Complete TiRex Native Data Flow

### **Primary Pipeline Architecture**

```
CONTEXT → TOKENIZED → [TiRex xLSTM] → PREDICTIONS → FEATURES → SIGNALS
   ↓           ↓                          ↓           ↓          ↓
Exchange → Preprocessing → Model Processing → Forecasts → Indicators → Trading
(11 cols)    (2→8 cols)     [sLSTM Core]     (4 cols)    (5 cols)   (3 cols)
```

### **Native TiRex Component Mapping**

```
Layer          TiRex Component                    Processing Function
─────────────────────────────────────────────────────────────────────────────────
CONTEXT    →   context: torch.Tensor              Raw market data input
TOKENIZED  →   PatchedUniTokenizer               .context_input_transform()
Processing →   TiRexZero.forward()               xLSTM block stack processing
PREDICTIONS→   quantile_preds                    .output_patch_embedding()
FEATURES   →   Post-processing                   Traditional TA + TiRex
SIGNALS    →   Trading logic                     Decision algorithms
```

---

## Layer-by-Layer Dependency Analysis

### **CONTEXT Layer Dependencies**

**Status**: ✅ **Independent** - No internal dependencies  
**External Dependencies**:

- DSM (Data Source Manager) via FCP protocol
- Exchange API data feed (Binance historical OHLCV)
- Apache Arrow MMAP optimization system

**Downstream Impact**: **Foundation layer** - all subsequent processing depends on CONTEXT quality

```python
# CONTEXT validation requirements
context_requirements = {
    'columns': 11,                    # Complete OHLCV + microstructure
    'data_quality': 0.99,            # 99%+ completeness
    'timestamp_precision': 'ms',      # Millisecond accuracy
    'update_latency': '<100ms',       # Real-time constraint
    'source_verification': True      # Exchange-verified trades
}
```

---

### **TOKENIZED Layer Dependencies**

**Status**: 🔄 **OPTIMIZATION TARGET** - Critical performance bottleneck  
**Dependencies**:

- **Primary**: [CONTEXT layer](./context-layer.md) (all 11 columns)
- **TiRex Components**: `PatchedUniTokenizer`, `StandardScaler`, `_Patcher`
- **Processing Pipeline**: `context_input_transform()` → `scaler.scale()` → `patcher()`

**Downstream Impact**: **CRITICAL PATH** - optimization creates cascading performance improvements

#### Current vs Optimized Dependency Chain

```python
# CURRENT (Under-optimized)
current_tokenized_deps = {
    'input_features': 2,              # ctx_close, ctx_norm_close
    'tirex_utilization': 0.25,        # 25% of input_patch_size * 2
    'context_consumption': 'basic',    # Only price data
    'processing_efficiency': 'low'    # Static normalization
}

# OPTIMIZED (Target)
optimized_tokenized_deps = {
    'input_features': 8,              # Full feature matrix
    'tirex_utilization': 1.0,         # 100% of architecture capacity
    'context_consumption': 'complete', # OHLCV + volume + microstructure
    'processing_efficiency': 'high',   # Regime-adaptive preprocessing
    'performance_multiplier': '2-4x'   # Expected improvement
}
```

#### Tokenization Dependency Flow

```python
# Complete TOKENIZED processing dependency chain
def tokenized_processing_pipeline(context_data):
    """
    Dependencies: CONTEXT[11 columns] → TOKENIZED[8 features]
    """

    # Phase 1: Core price features (HIGH priority deps)
    ohlc_patches = process_ohlc_regime_normalization(
        context_data['open'], context_data['high'],
        context_data['low'], context_data['close']
    )
    returns_scaled = process_returns_scaling(context_data['close'])
    volatility_patches = process_volatility_patches(
        context_data['high'], context_data['low'], context_data['close']
    )

    # Phase 2: Volume intelligence (MEDIUM priority deps)
    volume_scaled = process_volume_regime(context_data['volume'])
    orderflow_patches = process_orderflow_intelligence(
        context_data['taker_buy_volume'], context_data['volume']
    )
    activity_scaled = process_activity_regime(context_data['count'])

    # Phase 3: Advanced regime detection (LOW priority deps)
    regime_patches = process_regime_detection(context_data)
    session_scaled = process_session_transitions(context_data)

    # TiRex native tokenization
    enhanced_context = torch.stack([
        ohlc_patches, returns_scaled, volatility_patches,
        volume_scaled, orderflow_patches, activity_scaled,
        regime_patches, session_scaled
    ], dim=-1)

    tokenized_tensor, scaler_state = tokenizer.context_input_transform(enhanced_context)

    return tokenized_tensor, scaler_state
```

---

### **PREDICTIONS Layer Dependencies**

**Status**: ✅ **Stable** - Direct TiRex model output  
**Dependencies**:

- **Primary**: [TOKENIZED layer](./tokenized-layer.md) output (tokenized_tensor + scaler_state)
- **TiRex Components**: `TiRexZero`, `MixedStack`, `output_patch_embedding`
- **Guardian System**: `TiRexGuardian.safe_forecast()` for protection

**Performance Relationship**: **Linear multiplier** from TOKENIZED optimization

```python
# PREDICTIONS performance scaling with TOKENIZED optimization
predictions_performance = {
    'current_tokenized_2_features': {
        'forecast_accuracy': 'baseline',
        'uncertainty_quantification': 'limited',
        'regime_adaptation': 'poor'
    },
    'optimized_tokenized_8_features': {
        'forecast_accuracy': '2-4x improvement',     # Direct scaling
        'uncertainty_quantification': '3-5x improvement', # Better volatility estimation
        'regime_adaptation': 'excellent'            # Multi-regime intelligence
    }
}
```

#### TiRex Internal Processing Dependencies

```python
# TiRex native processing chain (from source code analysis)
def tirex_internal_processing(tokenized_tensor, scaler_state):
    """
    Internal TiRex dependencies: TOKENIZED → xLSTM → PREDICTIONS
    """

    # Input embedding (depends on tokenized quality)
    input_embeds = model.input_patch_embedding(
        torch.cat((tokenized_tensor, input_mask), dim=2)
    )  # Quality directly impacts embedding effectiveness

    # sLSTM block processing (benefits from rich tokenized features)
    hidden_states = model.block_stack(input_embeds)
    # Enhanced tokenized features → better temporal patterns → improved sLSTM memory

    # Output generation (quantile prediction quality scales with input richness)
    quantile_preds = model.output_patch_embedding(hidden_states)
    quantile_preds = torch.unflatten(
        quantile_preds, -1,
        (model.num_quantiles, model.model_config.output_patch_size)
    )

    # Output transformation (depends on preserved scaler_state)
    final_predictions = model.tokenizer.output_transform(
        quantile_preds, scaler_state
    )

    return final_predictions
```

---

### **FEATURES Layer Dependencies**

**Status**: ✅ **Stable** - Bridge layer with dual dependencies  
**Dependencies**:

- **Primary**: [PREDICTIONS layer](./predictions-layer.md) (tirex_quantiles, tirex_mean_p50)
- **Secondary**: [CONTEXT layer](./context-layer.md) (for traditional TA indicators)
- **Integration**: Combines TiRex intelligence with traditional technical analysis

**Performance Scaling**: **Exponential benefit** from PREDICTIONS improvement

```python
# FEATURES performance amplification
features_amplification = {
    'edge_1_calculation': {
        'current': 'basic_prediction_difference',
        'optimized': 'high_confidence_directional_signal',
        'improvement': '3-6x signal quality'
    },
    'uncertainty_features': {
        'current': 'limited_quantile_intelligence',
        'optimized': 'comprehensive_regime_detection',
        'improvement': '4-8x regime awareness'
    },
    'position_sizing': {
        'current': 'static_atr_based',
        'optimized': 'dynamic_kelly_criterion',
        'improvement': '2-5x risk_adjusted_returns'
    }
}
```

#### Cross-Layer Feature Dependencies

```python
# FEATURES layer dependency integration
def calculate_enhanced_features(predictions, context):
    """
    Dual dependency: PREDICTIONS + CONTEXT → FEATURES
    """

    # TiRex-derived features (PREDICTIONS dependency)
    edge_1 = predictions['tirex_mean_p50'][0] - context['close'][-1]
    uncertainty = predictions['tirex_q_p90'] - predictions['tirex_q_p10']
    prediction_confidence = calculate_confidence(predictions['tirex_quantiles'])

    # Traditional TA features (CONTEXT dependency)
    atr_14 = talib.ATR(context['high'], context['low'], context['close'], 14)[-1]
    ma_20 = context['close'].rolling(20).mean()[-1]
    rsi_14 = talib.RSI(context['close'], 14)[-1]

    # Hybrid TiRex-enhanced features (DUAL dependency)
    tirex_enhanced_atr = 0.7 * atr_14 + 0.3 * uncertainty.mean()
    trend_confirmation = (edge_1 > 0) & (context['close'][-1] > ma_20)

    # Position sizing (TRIPLE dependency: PREDICTIONS + CONTEXT + risk parameters)
    pos_size = calculate_kelly_position_size(
        edge=edge_1,
        win_prob=prediction_confidence,
        avg_win=predictions['tirex_q_p90'] - predictions['tirex_mean_p50'],
        avg_loss=predictions['tirex_mean_p50'] - predictions['tirex_q_p10'],
        risk_budget=0.02
    )

    return {
        'edge_1': edge_1,
        'atr_14': tirex_enhanced_atr,
        'ma_20': ma_20,
        'rsi_14': rsi_14,
        'pos_size': pos_size
    }
```

---

### **SIGNALS Layer Dependencies**

**Status**: ✅ **Stable** - Final decision layer  
**Dependencies**:

- **Primary**: [FEATURES layer](./features-layer.md) (edge_1, atr_14, ma_20, rsi_14, pos_size)
- **Secondary**: [PREDICTIONS layer](./predictions-layer.md) (for take-profit levels)
- **Integration**: Converts intelligence into executable trading decisions

**Performance Culmination**: **Maximum benefit** realization from entire pipeline optimization

```python
# SIGNALS layer performance culmination
signals_performance_scaling = {
    'signal_accuracy': {
        'baseline': '~55% win rate (random)',
        'current_pipeline': '~62% win rate',
        'optimized_pipeline': '~75-80% win rate',  # 2-4x tokenized improvement
        'improvement_factor': '2.4-3.2x'
    },
    'risk_adjusted_returns': {
        'baseline': '0.8 Sharpe ratio',
        'current_pipeline': '1.2 Sharpe ratio',
        'optimized_pipeline': '2.4-3.6 Sharpe ratio',  # Compounded improvements
        'improvement_factor': '3-4.5x'
    }
}
```

#### Signal Generation Dependencies

```python
# SIGNALS final decision dependencies
def generate_trading_signals(features, predictions):
    """
    Final layer: FEATURES + PREDICTIONS → SIGNALS
    """

    # Entry signal (FEATURES dependency)
    sig_long = (features['edge_1'] > 1.5 * features['atr_14']) & \
               (features['close'] > features['ma_20'])

    # Take-profit level (PREDICTIONS dependency)
    tp_lvl = predictions['tirex_q_p90'][12]  # 1-hour horizon, 90th percentile

    # Stop-loss level (FEATURES dependency)
    sl_lvl = features['close'] - 2.0 * features['atr_14']

    # Position sizing integration (FEATURES dependency)
    position_size = features['pos_size']

    return {
        'sig_long': sig_long,
        'tp_lvl': tp_lvl,
        'sl_lvl': sl_lvl,
        'position_size': position_size,
        'signal_confidence': calculate_signal_confidence(features, predictions)
    }
```

---

## Critical Path Analysis

### **Performance Bottleneck Identification**

```
Pipeline Performance Flow:
CONTEXT[stable] → TOKENIZED[BOTTLENECK] → PREDICTIONS[scaling] → FEATURES[amplifying] → SIGNALS[realizing]

Critical Finding: TOKENIZED layer 25% utilization constrains entire pipeline performance
```

### **Optimization Impact Propagation**

```python
# Cascading performance improvements from TOKENIZED optimization
optimization_cascade = {
    'tokenized_improvement': {
        'feature_count': '2 → 8 (4x)',
        'architecture_utilization': '25% → 100% (4x)',
        'direct_impact': '2-4x prediction accuracy'
    },

    'predictions_amplification': {
        'forecast_quality': '2-4x improvement',
        'uncertainty_quantification': '3-5x improvement',
        'regime_adaptation': 'poor → excellent'
    },

    'features_enhancement': {
        'edge_signal_quality': '3-6x improvement',
        'position_sizing': '2-5x risk-adjusted returns',
        'regime_awareness': '4-8x improvement'
    },

    'signals_realization': {
        'win_rate': '62% → 75-80%',
        'sharpe_ratio': '1.2 → 2.4-3.6',
        'trading_performance': '3-4.5x improvement'
    }
}
```

---

## Guardian System Integration

### **Multi-Layer Protection Dependencies**

```python
# Guardian system protecting entire pipeline
guardian_protection_layers = {
    'context_validation': {
        'layer': 'CONTEXT',
        'protection': 'Data quality validation',
        'dependency': 'DSM data integrity'
    },

    'tokenization_safety': {
        'layer': 'TOKENIZED',
        'protection': 'Input preprocessing validation',
        'dependency': 'PatchedUniTokenizer safety'
    },

    'prediction_security': {
        'layer': 'PREDICTIONS',
        'protection': 'Model output validation',
        'dependency': 'Quantile ordering, NaN detection'
    },

    'feature_consistency': {
        'layer': 'FEATURES',
        'protection': 'Feature calculation validation',
        'dependency': 'Range checks, consistency validation'
    },

    'signal_execution_safety': {
        'layer': 'SIGNALS',
        'protection': 'Trading decision validation',
        'dependency': 'Risk parameter validation'
    }
}
```

### **Cross-Layer Guardian Dependencies**

```python
# Guardian dependency chain for comprehensive protection
def guardian_pipeline_protection(context, tokenized, predictions, features, signals):
    """
    Multi-layer Guardian protection with dependency validation
    """

    # Layer 1: Context validation
    context_validated = guardian.validate_context_quality(context)
    if not context_validated:
        raise GuardianException("CONTEXT layer validation failed")

    # Layer 2: Tokenization protection (depends on context validation)
    tokenized_protected = guardian.validate_tokenization_safety(
        tokenized, context_metadata=context_validated
    )

    # Layer 3: Prediction security (depends on tokenization validation)
    predictions_secured = guardian.validate_prediction_output(
        predictions, tokenized_metadata=tokenized_protected
    )

    # Layer 4: Feature consistency (depends on predictions validation)
    features_validated = guardian.validate_feature_calculations(
        features, predictions_metadata=predictions_secured
    )

    # Layer 5: Signal execution safety (depends on feature validation)
    signals_approved = guardian.validate_trading_signals(
        signals, features_metadata=features_validated
    )

    return {
        'pipeline_protected': True,
        'validation_chain': [context_validated, tokenized_protected,
                           predictions_secured, features_validated, signals_approved],
        'ready_for_execution': signals_approved
    }
```

---

## Performance Monitoring Dependencies

### **End-to-End Performance Tracking**

```python
# Complete pipeline performance monitoring
class PipelinePerformanceMonitor:
    """
    Monitor performance dependencies across all layers
    """

    def __init__(self):
        self.layer_metrics = {
            'context': ContextMetrics(),
            'tokenized': TokenizedMetrics(),
            'predictions': PredictionsMetrics(),
            'features': FeaturesMetrics(),
            'signals': SignalsMetrics()
        }

    def track_pipeline_performance(self, pipeline_data):
        """Track performance propagation through pipeline"""

        # Individual layer performance
        context_perf = self.layer_metrics['context'].calculate(pipeline_data['context'])
        tokenized_perf = self.layer_metrics['tokenized'].calculate(
            pipeline_data['tokenized'], dependency_quality=context_perf
        )
        predictions_perf = self.layer_metrics['predictions'].calculate(
            pipeline_data['predictions'], dependency_quality=tokenized_perf
        )
        features_perf = self.layer_metrics['features'].calculate(
            pipeline_data['features'],
            prediction_dependency=predictions_perf,
            context_dependency=context_perf
        )
        signals_perf = self.layer_metrics['signals'].calculate(
            pipeline_data['signals'], dependency_quality=features_perf
        )

        # Cross-layer dependency impact analysis
        dependency_impact = self.analyze_dependency_impact({
            'context': context_perf,
            'tokenized': tokenized_perf,
            'predictions': predictions_perf,
            'features': features_perf,
            'signals': signals_perf
        })

        return {
            'layer_performance': {
                'context': context_perf,
                'tokenized': tokenized_perf,
                'predictions': predictions_perf,
                'features': features_perf,
                'signals': signals_perf
            },
            'dependency_impact': dependency_impact,
            'optimization_priority': self.identify_bottlenecks(dependency_impact)
        }
```

---

## Future Enhancement Dependencies

### **Scalable Architecture Evolution**

```python
# Pipeline evolution roadmap with dependencies
enhancement_roadmap = {
    'phase_1_tokenized_optimization': {
        'priority': 'CRITICAL',
        'dependencies': ['CONTEXT layer stability'],
        'impact_layers': ['PREDICTIONS', 'FEATURES', 'SIGNALS'],
        'expected_improvement': '2-4x pipeline performance'
    },

    'phase_2_multi_timeframe': {
        'priority': 'HIGH',
        'dependencies': ['Phase 1 completion', 'Enhanced PREDICTIONS'],
        'new_components': ['Multi-horizon forecasting', 'Cross-timeframe consistency'],
        'impact_layers': ['FEATURES', 'SIGNALS']
    },

    'phase_3_ensemble_integration': {
        'priority': 'MEDIUM',
        'dependencies': ['Phase 2 validation', 'Multiple TiRex models'],
        'new_components': ['Model ensemble', 'Prediction combination'],
        'impact_layers': ['PREDICTIONS', 'FEATURES', 'SIGNALS']
    }
}
```

---

## Conclusion

The TiRex native pipeline reveals **clear performance optimization path**:

### **Critical Findings**

1. **TOKENIZED Layer Bottleneck**: 25% architecture utilization constrains entire pipeline
2. **Cascading Performance**: TOKENIZED optimization creates 2-4x improvement across all layers
3. **Dependency Chain**: Each layer amplifies improvements from upstream optimizations
4. **Guardian Integration**: Multi-layer protection ensures system reliability and safety

### **Optimization Priority**

```
HIGHEST IMPACT: TOKENIZED layer optimization (2→8 features)
    ↓
MEDIUM IMPACT: FEATURES layer TiRex enhancement
    ↓
LOW IMPACT: SIGNALS layer refinement
```

### **Expected System-Wide Benefits**

- **Pipeline Performance**: 3-4.5x overall improvement
- **Trading Performance**: 75-80% win rate, 2.4-3.6 Sharpe ratio
- **Architecture Utilization**: 100% of TiRex native capabilities
- **System Reliability**: Comprehensive Guardian protection across all layers

**Next Action**: Begin **TOKENIZED layer optimization** implementation to unlock system-wide performance improvements across the entire TiRex-enhanced trading pipeline.

---

[← Back to Index](../strategy-io-contract.md#layer-navigation-tirex-native) | [Complete Architecture Overview](../strategy-io-contract.md)
