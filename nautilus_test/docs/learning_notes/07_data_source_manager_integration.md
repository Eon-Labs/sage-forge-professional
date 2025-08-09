# NautilusTrader Learning Notes - Data Source Manager Integration

## 🌐 Advanced Data Pipeline Integration with DSM

This document covers the critical lessons learned from integrating the Data Source Manager (DSM) with NautilusTrader for production-grade data handling.

---

## 🎯 Overview: What is Data Source Manager?

### Purpose

The Data Source Manager (DSM) is a sophisticated data fetching and caching system that provides:

- **High-performance data retrieval** from multiple exchanges
- **Automatic failover and redundancy** (Failover Control Protocol - FCP)
- **Modern data processing** with Polars/PyArrow
- **Intelligent caching** and data quality validation

### Key Benefits

- **Production reliability** with error handling
- **Performance optimization** through caching
- **Data quality assurance** with validation
- **Multi-exchange support** with unified interface

---

## ⚠️ Critical Lesson: Market Type Configuration

### The Problem We Solved

```python
# WRONG - Using SPOT market for FUTURES data
manager = DataSourceManager.create(DataProvider.BINANCE, MarketType.SPOT)
```

**Result**: 62.8% data completeness, 66 skipped bars, unreliable backtests

### The Correct Solution

```python
# CORRECT - Using FUTURES_USDT for perpetual futures
manager = DataSourceManager.create(DataProvider.BINANCE, MarketType.FUTURES_USDT)
```

**Result**: 100% data completeness, 0 skipped bars, perfect reliability

### Market Type Reference

| Instrument            | Correct Market Type       | API Endpoint     | Symbol Format |
| --------------------- | ------------------------- | ---------------- | ------------- |
| **BTCUSDT spot**      | `MarketType.SPOT`         | api.binance.com  | "BTCUSDT"     |
| **BTCUSDT perpetual** | `MarketType.FUTURES_USDT` | fapi.binance.com | "BTCUSDT"     |
| **BTCUSD perpetual**  | `MarketType.FUTURES_COIN` | dapi.binance.com | "BTCUSD_PERP" |

### Key Takeaway

**Always match your DSM market type to your actual trading instrument. This is critical for data quality.**

---

## 📊 Data Quality Validation Pipeline

### Implementation Pattern

```python
class EnhancedDataPipeline:
    def fetch_and_validate(self, symbol, limit=200):
        # 1. Fetch data with correct market type
        df = self.data_manager.fetch_real_market_data(symbol, limit=limit)

        # 2. Validate data quality
        self._validate_data_quality(df)

        # 3. Adjust precision to match instrument
        df = self._adjust_precision(df, self.instrument)

        # 4. Create bars with validation
        bars = self._create_validated_bars(df, self.bar_type)

        return bars

    def _validate_data_quality(self, df):
        valid_rows = df.dropna().shape[0]
        total_rows = df.shape[0]
        completeness = valid_rows / total_rows

        if completeness < 0.8:
            raise ValueError(f"Data only {completeness:.1%} complete")

        console.print(f"✅ Data quality: {completeness:.1%} complete")
```

### Quality Metrics to Monitor

```python
quality_metrics = {
    'data_completeness': f"{valid_rows}/{total_rows} ({completeness:.1%})",
    'price_range': f"${min_price:.2f} - ${max_price:.2f}",
    'volatility': f"{price_range/min_price*100:.2f}% swing",
    'api_endpoint': "fapi.binance.com (correct futures endpoint)",
    'market_type': "FUTURES_USDT (perpetual futures)"
}
```

---

## 🔧 Precision Handling for Different Data Sources

### The Challenge

DSM returns data with different precision than NautilusTrader instruments expect:

- **DSM Data**: Often 5+ decimal places for prices
- **Binance Futures**: 2 decimal places for BTCUSDT
- **NautilusTrader**: Requires exact precision match

### The Solution

```python
def adjust_data_precision(self, df, instrument):
    """Adjust data precision to match instrument specifications."""
    try:
        # Handle both Polars and Pandas DataFrames
        if hasattr(df, 'with_columns'):  # Polars
            expressions = []
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    expressions.append(pl.col(col).round(instrument.price_precision))

            if 'volume' in df.columns:
                expressions.append(pl.col('volume').round(instrument.size_precision))

            if expressions:
                df = df.with_columns(expressions)
        else:  # Pandas
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].round(instrument.price_precision)

            if 'volume' in df.columns:
                df['volume'] = df['volume'].round(instrument.size_precision)

        return df
    except Exception as e:
        console.print(f"[yellow]⚠️ Precision adjustment failed: {e}[/yellow]")
        return df
```

### Bar Creation with Validation

```python
def create_validated_bars(self, df, instrument, bar_type):
    """Create NautilusTrader bars with exact precision validation."""
    bars = []

    for i, row in df.iterrows():
        try:
            # Validate data before bar creation
            if any(pd.isna(row[col]) for col in ['open', 'high', 'low', 'close', 'volume']):
                console.print(f"[yellow]⚠️ Skipping bar {i}: contains NaN values[/yellow]")
                continue

            # Create timestamp
            timestamp = self._create_timestamp(row, i, len(df))
            ts_ns = int(timestamp.timestamp() * 1_000_000_000)

            # Create bar with exact precision
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{float(row['open']):.{instrument.price_precision}f}"),
                high=Price.from_str(f"{float(row['high']):.{instrument.price_precision}f}"),
                low=Price.from_str(f"{float(row['low']):.{instrument.price_precision}f}"),
                close=Price.from_str(f"{float(row['close']):.{instrument.price_precision}f}"),
                volume=Quantity.from_str(f"{float(row['volume']):.{instrument.size_precision}f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)

        except Exception as e:
            console.print(f"[yellow]⚠️ Skipping bar {i}: {e}[/yellow]")
            continue

    console.print(f"✅ Created {len(bars)} validated bars")
    return bars
```

---

## 🚨 Error Handling and Failover

### DSM Failover Control Protocol (FCP)

The DSM includes sophisticated error handling:

```python
# Common FCP warnings and their meanings
warnings = {
    "Recent data coverage concern: 62.8% complete":
        "Data source has gaps in recent time periods",

    "Significant gap at end: 2025-07-14 02:06:00 to 2025-07-14 03:13:00":
        "Live data feed stopped, reindexing will create NaN values",

    "Reindexed DataFrame contains 66/180 rows (36.67%) with missing data":
        "Automatic gap-filling created NaN values requiring handling"
}
```

### Handling DSM Failures

```python
def fetch_with_fallback(self, symbol, limit):
    """Fetch data with automatic fallback strategies."""
    try:
        # Primary: Use DSM with correct market type
        if self.has_dsm:
            return self._fetch_with_dsm(symbol, limit)
    except Exception as e:
        console.print(f"[yellow]⚠️ DSM fetch failed: {e}[/yellow]")
        console.print("[blue]🔄 Falling back to synthetic data generation[/blue]")

        # Fallback: Generate synthetic data with real specifications
        return self._create_synthetic_bars_with_real_specs(symbol, limit)
```

### Synthetic Data Generation (Fallback)

```python
def create_synthetic_bars_with_real_specs(self, instrument, count=200):
    """Create realistic synthetic data when live data unavailable."""
    if not self.specs_manager.specs:
        raise ValueError("Real specifications required for synthetic data")

    current_price = self.specs_manager.specs['current_price']
    bars = []

    for i in range(count):
        # Simple random walk with realistic volatility
        price_change = random.uniform(-0.002, 0.002)  # ±0.2% per minute
        current_price *= (1 + price_change)

        # Create OHLC with proper precision
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{open_price:.{instrument.price_precision}f}"),
            high=Price.from_str(f"{high_price:.{instrument.price_precision}f}"),
            low=Price.from_str(f"{low_price:.{instrument.price_precision}f}"),
            close=Price.from_str(f"{close_price:.{instrument.price_precision}f}"),
            volume=Quantity.from_str(f"{volume:.{instrument.size_precision}f}"),
            ts_event=timestamp,
            ts_init=timestamp,
        )
        bars.append(bar)

    return bars
```

---

## 📈 Performance Optimization

### Caching Strategy

```python
def cache_to_parquet(self, df, filename):
    """Cache data using high-performance Parquet format."""
    cache_path = self.cache_dir / f"{filename}.parquet"
    df.write_parquet(cache_path, use_pyarrow=True)
    console.print(f"📊 Data cached to: {cache_path.name}")
    return cache_path

def load_from_cache(self, filename):
    """Load cached data if available and recent."""
    cache_path = self.cache_dir / f"{filename}.parquet"
    if cache_path.exists():
        # Check if cache is recent (within last hour)
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age < 3600:  # 1 hour
            console.print(f"📂 Loading from cache: {cache_path.name}")
            return pl.read_parquet(cache_path, use_pyarrow=True)
    return None
```

### Memory Management

```python
def get_data_stats(self, df):
    """Monitor memory usage and data characteristics."""
    return {
        'rows': df.height,
        'columns': df.width,
        'memory_usage_mb': df.estimated_size("mb"),
        'price_stats': {
            'mean': df["close"].mean(),
            'std': df["close"].std(),
            'range': df["close"].max() - df["close"].min()
        },
        'volume_stats': {
            'total': df["volume"].sum(),
            'mean': df["volume"].mean()
        }
    }
```

---

## 🎯 Integration Best Practices

### 1. Configuration Management

```python
class DSMConfig:
    """Centralized DSM configuration management."""

    MARKET_TYPE_MAP = {
        'BTCUSDT-PERP': MarketType.FUTURES_USDT,
        'BTCUSDT': MarketType.SPOT,
        'BTCUSD_PERP': MarketType.FUTURES_COIN
    }

    @classmethod
    def get_market_type(cls, symbol):
        """Determine correct market type from symbol."""
        if '-PERP' in symbol and 'USDT' in symbol:
            return MarketType.FUTURES_USDT
        elif '_PERP' in symbol:
            return MarketType.FUTURES_COIN
        else:
            return MarketType.SPOT
```

### 2. Data Validation Pipeline

```python
def validate_complete_pipeline(self, symbol, expected_bars=180):
    """Validate entire data pipeline end-to-end."""

    # Step 1: Configuration validation
    market_type = DSMConfig.get_market_type(symbol)
    assert market_type == MarketType.FUTURES_USDT, "Wrong market type"

    # Step 2: Data fetch validation
    df = self.fetch_data(symbol)
    assert len(df) >= expected_bars * 0.8, "Insufficient data"

    # Step 3: Quality validation
    completeness = df.dropna().shape[0] / df.shape[0]
    assert completeness >= 0.9, f"Poor data quality: {completeness:.1%}"

    # Step 4: Precision validation
    bars = self.create_bars(df)
    assert len(bars) >= expected_bars * 0.8, "Too many bars failed creation"

    # Step 5: Final validation
    assert all(not pd.isna(bar.close) for bar in bars), "NaN values in bars"

    return bars
```

### 3. Monitoring and Alerting

```python
def monitor_data_quality(self, bars, expected_count=180):
    """Monitor data pipeline health and alert on issues."""

    quality_metrics = {
        'bars_created': len(bars),
        'expected_bars': expected_count,
        'success_rate': len(bars) / expected_count,
        'data_completeness': self._calculate_completeness(bars),
        'precision_accuracy': self._validate_precision(bars)
    }

    # Alert on poor quality
    if quality_metrics['success_rate'] < 0.9:
        console.print(f"[red]🚨 ALERT: Low bar success rate {quality_metrics['success_rate']:.1%}[/red]")

    if quality_metrics['data_completeness'] < 0.95:
        console.print(f"[yellow]⚠️ WARNING: Data completeness {quality_metrics['data_completeness']:.1%}[/yellow]")

    return quality_metrics
```

---

## 🚀 Production Deployment Checklist

### Pre-deployment Validation

- [ ] **Market type configuration verified** for target instrument
- [ ] **Data quality consistently >95%** across multiple test runs
- [ ] **Precision alignment confirmed** between DSM and NautilusTrader
- [ ] **Error handling tested** with network failures and API errors
- [ ] **Caching performance validated** for expected data volumes
- [ ] **Memory usage profiled** and within acceptable limits

### Runtime Monitoring

- [ ] **Data completeness alerts** for <90% quality
- [ ] **Bar creation success rate** monitoring (target: >95%)
- [ ] **API endpoint validation** (correct fapi.binance.com usage)
- [ ] **Cache hit rate monitoring** for performance optimization
- [ ] **Memory usage tracking** to prevent memory leaks
- [ ] **Error rate monitoring** with automatic failover testing

### Performance Targets

| Metric                | Target | Alert Threshold |
| --------------------- | ------ | --------------- |
| **Data Completeness** | >99%   | <95%            |
| **Bar Success Rate**  | >98%   | <90%            |
| **Cache Hit Rate**    | >80%   | <60%            |
| **Memory Usage**      | <100MB | >200MB          |
| **API Response Time** | <2s    | >5s             |

---

## 🎯 Summary: DSM Integration Success

### What We Achieved

1. ✅ **Perfect data quality** (100% completeness vs 62.8% before)
2. ✅ **Zero skipped bars** (vs 66 skipped before)
3. ✅ **Correct market configuration** (FUTURES_USDT vs SPOT)
4. ✅ **Robust error handling** with fallback strategies
5. ✅ **Production-ready monitoring** and validation

### Key Success Factors

- **Correct market type mapping** for each instrument
- **Comprehensive data validation** at every pipeline stage
- **Precision alignment** between data sources and instruments
- **Fallback strategies** for data source failures
- **Performance optimization** through caching and memory management

### The Ultimate Lesson

**DSM integration requires careful configuration and validation, but when done correctly, provides enterprise-grade data reliability for trading systems.**

---

_This document captures the complete DSM integration journey from failure (62.8% data quality) to success (100% data quality). Study the patterns and apply them to your own integrations._
