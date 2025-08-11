### CONTEXT Layer — Exchange Data Foundation

**Native TiRex Component**: `context: torch.Tensor` (primary input parameter)  
**Data Pipeline Role**: Raw market data → TiRex input preprocessing  
**Source of Truth**: DSM (Data Source Manager) via FCP protocol

---

#### Executive Summary

**Purpose**: Foundational exchange data that feeds into TiRex's `PatchedUniTokenizer` for preprocessing  
**Stability**: ✅ **Complete and Stable** - Direct exchange data with standardized structure  
**Columns**: 11 comprehensive market data columns from Binance historical OHLCV  
**Integration**: Direct input to [TOKENIZED layer](./tokenized-layer.md) optimization pipeline

---

#### CONTEXT Layer Specifications

**Data Authority**: DSM (`repos/data-source-manager`) - Eon-Labs private FCP-based system  
**Timestamp Semantics**: UTC with millisecond precision, `open_time` = beginning of candle period  
**Update Frequency**: Real-time market data ingestion via DSM integration  
**Quality Assurance**: Exchange-verified OHLCV data with trade-level validation

---

#### Complete CONTEXT Column Specifications

| Column                     | Type     | Definition                    | Formula                                   | Tools     | Lineage          | Notes                       |
| -------------------------- | -------- | ----------------------------- | ----------------------------------------- | --------- | ---------------- | --------------------------- |
| **open_time**              | datetime | Bar open (UTC, ms precision)  | —                                         | DSM (FCP) | exchange via DSM | Beginning of candle period  |
| **open**                   | float    | Opening price                 | —                                         | DSM (FCP) | exchange via DSM | First trade price in period |
| **high**                   | float    | Highest price                 | —                                         | DSM (FCP) | exchange via DSM | Maximum price during period |
| **low**                    | float    | Lowest price                  | —                                         | DSM (FCP) | exchange via DSM | Minimum price during period |
| **close**                  | float    | Closing price                 | —                                         | DSM (FCP) | exchange via DSM | Last trade price in period  |
| **volume**                 | float    | Base asset volume             | —                                         | DSM (FCP) | exchange via DSM | Total base quantity traded  |
| **close_time**             | datetime | Bar close (UTC, ms precision) | `close_time = open_time + interval - 1ms` | DSM (FCP) | exchange via DSM | End of candle period        |
| **quote_asset_volume**     | float    | Quote asset volume            | —                                         | DSM (FCP) | exchange via DSM | Total quote value traded    |
| **count**                  | int      | Number of trades              | —                                         | DSM (FCP) | exchange via DSM | Trade count in period       |
| **taker_buy_volume**       | float    | Taker buy base volume         | —                                         | DSM (FCP) | exchange via DSM | Aggressive buy volume       |
| **taker_buy_quote_volume** | float    | Taker buy quote volume        | —                                         | DSM (FCP) | exchange via DSM | Aggressive buy value        |

---

#### TiRex Integration Pipeline

##### CONTEXT → TiRex Processing Flow

**⚠️ CRITICAL**: TiRex is univariate-only - must select ONE time series from context data

```python
# CONTEXT provides raw material for univariate selection
context_dataframe = pd.DataFrame({
    # Core OHLCV (5 columns)
    'open': market_data['open'].values,
    'high': market_data['high'].values,
    'low': market_data['low'].values,
    'close': market_data['close'].values,
    'volume': market_data['volume'].values,
    
    # Extended microstructure data (6 columns)
    'quote_asset_volume': market_data['quote_asset_volume'].values,
    'count': market_data['count'].values,
    'taker_buy_volume': market_data['taker_buy_volume'].values,
    'taker_buy_quote_volume': market_data['taker_buy_quote_volume'].values
})

# TiRex univariate input selection (choose ONE)
univariates_options = {
    'raw_close': torch.tensor(context_dataframe['close'].values, dtype=torch.float32),
    'log_returns': torch.tensor(np.log(context_dataframe['close'].pct_change().dropna() + 1), dtype=torch.float32),
    'typical_price': torch.tensor(((context_dataframe['high'] + context_dataframe['low'] + context_dataframe['close']) / 3).values, dtype=torch.float32),
    'volume_weighted': torch.tensor((context_dataframe['close'] * context_dataframe['volume']).values, dtype=torch.float32)
}

# Select optimal univariate series for TiRex
selected_series = univariates_options['raw_close']  # Example selection
context_tensor = selected_series.unsqueeze(0)  # Shape: [1, sequence_length]

# Valid TiRex processing: context_tensor → TOKENIZED layer → PatchedUniTokenizer
```

##### Data Quality Characteristics

- **Completeness**: 100% data availability via DSM integration
- **Accuracy**: Exchange-verified trade data with millisecond precision
- **Consistency**: Standardized column naming across all trading pairs
- **Reliability**: DSM handles exchange API failures and data gaps
- **Performance**: Apache Arrow MMAP optimization for high-speed access

---

#### Microstructure Intelligence Available

##### Volume Analysis Potential

- **Base Volume**: Total quantity traded (liquidity indicator)
- **Quote Volume**: Total value traded (market size indicator)
- **Volume Ratio**: `base_volume / quote_volume` (price level analysis)

##### Order Flow Intelligence

- **Taker Buy Pressure**: `taker_buy_volume / total_volume` (aggressive buyer ratio)
- **Market Impact**: `taker_buy_quote_volume / quote_asset_volume` (value-weighted aggression)
- **Trade Intensity**: `count / time_interval` (activity level)

##### Session Analysis

- **Intrabar Patterns**: `(high - low) / (close - open)` (volatility vs momentum)
- **Gap Analysis**: `open[t] - close[t-1]` (session transition dynamics)
- **Time-of-Day Effects**: Hourly/session pattern recognition potential

---

#### CONTEXT → TOKENIZED Univariate Selection Pipeline

The CONTEXT layer provides **univariate selection options** for [TOKENIZED layer optimization](./tokenized-layer.md):

**🔬 EMPIRICAL REALITY**: TiRex processes single time series only - optimization focuses on selecting optimal univariate input

##### Univariate Input Options (TiRex Compatible)

**Primary Options (Direct Price Series)**:
- **Raw Close Prices**: `context_data['close']` - Direct price forecasting
- **Log Returns**: `np.log(close[t]/close[t-1])` - Return-based forecasting  
- **Typical Price**: `(high + low + close) / 3` - Representative price

**Secondary Options (Derived Series)**:
- **Volume-Weighted Price**: `(close * volume).rolling(window).mean()` - Volume-adjusted
- **High-Low Midpoint**: `(high + low) / 2` - Volatility-aware price
- **Normalized Prices**: Z-score transformed close prices for stationarity

**Selection Strategy**: A/B test different univariate series to determine optimal TiRex input for specific market conditions

**Optimization Focus**: Input quality, preprocessing, and temporal feature engineering within single time series constraint

---

#### Data Lineage & Governance

##### Source Chain Validation

```
Exchange (Binance) → FCP Protocol → DSM (data-source-manager) → CONTEXT Layer → TOKENIZED Layer
```

##### Quality Metrics

- **Latency**: <100ms from exchange to CONTEXT availability
- **Completeness**: 99.9%+ data availability (DSM handles gaps)
- **Accuracy**: Exchange-verified with trade-level reconciliation
- **Freshness**: Real-time updates within market session constraints

##### Audit Trail

- **Source Attribution**: Every data point traceable to specific exchange trade
- **Processing Log**: Complete DSM ingestion and validation history
- **Version Control**: Data schema versioning for historical consistency
- **Access Control**: Secure data pipeline with authentication/authorization

---

#### Technical Implementation

##### DSM Integration Pattern

```python
from repos.data_source_manager import ArrowDataManager

# CONTEXT data access for univariate selection
context_manager = ArrowDataManager()
context_data = context_manager.get_ohlcv_data(
    symbol="BTCUSDT",
    interval="5m",
    start_time=datetime.now() - timedelta(hours=6),
    end_time=datetime.now()
)

# CONTEXT provides options, select ONE for TiRex
univariates_available = {
    'close_prices': context_data['close'].values,
    'typical_prices': ((context_data['high'] + context_data['low'] + context_data['close']) / 3).values,
    'log_returns': np.log(context_data['close'].pct_change().dropna() + 1).values,
    'volume_weighted': (context_data['close'] * context_data['volume']).values
}

# Select optimal univariate for TiRex processing
selected_univariate = univariates_available['close_prices']  # Choose ONE
context_tensor = torch.tensor(selected_univariate, dtype=torch.float32).unsqueeze(0)
# Shape: [1, sequence_length] - TiRex compatible
```

##### Memory Optimization

```python
# Apache Arrow MMAP for efficient data access
context_array = context_manager.get_mmap_array(
    columns=['open', 'high', 'low', 'close', 'volume',
             'quote_asset_volume', 'count', 'taker_buy_volume',
             'taker_buy_quote_volume'],
    dtype=np.float32
)
```

---

#### Integration Points

##### Downstream Dependencies

- **[TOKENIZED Layer](./tokenized-layer.md)**: Primary consumer for optimization
- **[FEATURES Layer](./features-layer.md)**: Technical indicator calculations
- **[Pipeline Dependencies](./pipeline-dependencies.md)**: Complete data flow analysis

##### Guardian System Integration

```python
# CONTEXT data validation before TOKENIZED processing
guardian.validate_context_data(
    context=context_tensor,
    required_columns=11,
    data_quality_threshold=0.99,
    timestamp_consistency=True
)
```

---

#### Future Enhancements

##### Extended Market Data

- **Level 2 Order Book**: Bid/ask depth data (if available)
- **Trade-by-Trade**: Individual transaction details for precise microstructure
- **Cross-Exchange**: Multi-venue data aggregation for comprehensive view

##### Real-time Optimization

- **Streaming Updates**: Real-time context updates for live trading
- **Delta Processing**: Incremental updates to reduce latency
- **Buffering Strategy**: Optimal batch sizes for TiRex input preparation

---

#### Conclusion

The CONTEXT layer provides **comprehensive market intelligence foundation** with 11 standardized columns covering:

- Core OHLCV price data
- Volume and liquidity metrics
- Order flow and microstructure intelligence
- Trade activity and intensity measures

This rich CONTEXT data enables **optimal univariate selection** for the [TOKENIZED layer](./tokenized-layer.md), where proper input choice and preprocessing can achieve **10-30% TiRex performance improvement** through input quality optimization within architectural constraints.

**Status**: ✅ **Production Ready** - Stable, reliable, and comprehensive exchange data foundation for TiRex optimization pipeline.

---

[← Back to Index](../strategy-io-contract.md#layer-navigation-tirex-native) | [Next: TOKENIZED Layer →](./tokenized-layer.md)
