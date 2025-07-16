"""Modern data integration utilities for NautilusTrader.

This module provides high-performance data processing capabilities using the latest
Arrow ecosystem technologies (Polars, PyArrow) integrated with NautilusTrader's
native data infrastructure.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import polars as pl
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from rich.console import Console

console = Console()


class ArrowDataManager:
    """High-performance data manager using Arrow ecosystem."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize data manager with optional cache directory."""
        self.cache_dir = cache_dir or Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_real_market_data(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1m",
        limit: int = 1000,
        start_time: Optional[datetime] = None,  # New param
        end_time: Optional[datetime] = None,    # New param
    ) -> pl.DataFrame:
        """Fetch real market data using Data Source Manager."""
        import sys
        from datetime import datetime
        from pathlib import Path

        # Add data-source-manager to path
        dsm_path = Path("/Users/terryli/eon/data-source-manager")
        if str(dsm_path) not in sys.path:
            sys.path.insert(0, str(dsm_path))

        try:
            console.print("[cyan]🌐 Fetching real market data using Data Source Manager...[/cyan]")

            # Import DSM components
            from core.sync.data_source_manager import DataSourceManager
            from utils.market_constraints import DataProvider, Interval, MarketType

            # Create manager for Binance USDT-margined futures (perpetual futures)
            # BTCUSDT perpetual futures use FUTURES_USDT market type
            manager = DataSourceManager.create(DataProvider.BINANCE, MarketType.FUTURES_USDT)

            # Calculate time range for recent data
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(hours=limit // 60)
            assert isinstance(start_time, datetime), "start_time must be datetime"
            assert isinstance(end_time, datetime), "end_time must be datetime"

            # Map timeframe to DSM interval
            interval_map = {
                "1m": Interval.MINUTE_1,
                "5m": Interval.MINUTE_5,
                "15m": Interval.MINUTE_15,
                "1h": Interval.HOUR_1,
                "1d": Interval.DAY_1,
            }
            interval = interval_map.get(timeframe, Interval.MINUTE_1)

            # TIME SPAN VERIFICATION LOGGING
            console.print(f"[bold green]🔍 TIME SPAN VERIFICATION: Using start_time={start_time}[/bold green]")
            console.print(f"[bold green]🔍 TIME SPAN VERIFICATION: Using end_time={end_time}[/bold green]")
            console.print(f"[bold green]🔍 TIME SPAN VERIFICATION: Duration={(end_time-start_time).total_seconds()/3600:.1f} hours[/bold green]")
            console.print(f"[blue]📊 Fetching {symbol} PERPETUAL FUTURES from {start_time} to {end_time}...[/blue]")
            console.print("[cyan]🎯 Market Type: FUTURES_USDT (USDT-margined perpetual futures)[/cyan]")
            console.print("[cyan]🔗 API Endpoint: fapi.binance.com (Binance USDT-margined futures)[/cyan]")

            # Fetch real perpetual futures data using DSM
            df_pandas = manager.get_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=interval,
            )

            if df_pandas is None or df_pandas.empty:
                raise Exception("No data returned from DSM")

            console.print(f"[green]✅ DSM returned {len(df_pandas)} PERPETUAL FUTURES data points[/green]")
            
            # TIME SPAN VERIFICATION: Check actual data timestamps
            if not df_pandas.empty and 'timestamp' in df_pandas.columns:
                first_timestamp = df_pandas['timestamp'].iloc[0]
                last_timestamp = df_pandas['timestamp'].iloc[-1]
                console.print(f"[bold yellow]🔍 DATA VERIFICATION: First timestamp={first_timestamp}[/bold yellow]")
                console.print(f"[bold yellow]🔍 DATA VERIFICATION: Last timestamp={last_timestamp}[/bold yellow]")
                
                # Check if data matches expected time span
                if hasattr(first_timestamp, 'strftime'):
                    actual_start = first_timestamp.strftime('%Y-%m-%d %H:%M')
                    actual_end = last_timestamp.strftime('%Y-%m-%d %H:%M')
                    expected_start = start_time.strftime('%Y-%m-%d %H:%M')
                    expected_end = end_time.strftime('%Y-%m-%d %H:%M')
                    
                    console.print(f"[bold cyan]🎯 EXPECTED: {expected_start} to {expected_end}[/bold cyan]")
                    console.print(f"[bold cyan]🎯 ACTUAL: {actual_start} to {actual_end}[/bold cyan]")
                    
                    if actual_start == expected_start and actual_end == expected_end:
                        console.print(f"[bold green]✅ TIME SPAN MATCH: Data matches expected time period![/bold green]")
                    else:
                        console.print(f"[bold red]❌ TIME SPAN MISMATCH: Data doesn't match expected time period![/bold red]")
                else:
                    console.print(f"[yellow]⚠️ Unable to verify timestamps - format issue[/yellow]")

            # Validate data quality for futures
            valid_rows = df_pandas.dropna().shape[0]
            total_rows = df_pandas.shape[0]
            completeness = (valid_rows / total_rows) if total_rows > 0 else 0

            console.print(f"[blue]📊 Data Quality: {valid_rows}/{total_rows} valid rows ({completeness:.1%} complete)[/blue]")

            if completeness < 0.8:
                console.print(f"[yellow]⚠️ Low data completeness ({completeness:.1%}) - may indicate API issues[/yellow]")
            else:
                console.print(f"[green]✅ High data completeness ({completeness:.1%}) - futures data quality good[/green]")

            # Convert pandas to polars and rename columns to our format
            df_polars = pl.from_pandas(df_pandas)

            # Debug: Check what columns we actually have
            console.print(f"[blue]Available columns: {list(df_polars.columns)}[/blue]")
            
            # Additional timestamp verification with polars
            if 'timestamp' in df_polars.columns:
                console.print(f"[bold green]🕰️ POLARS DATA: {len(df_polars)} rows with timestamp column[/bold green]")
                first_ts = df_polars['timestamp'][0]
                last_ts = df_polars['timestamp'][-1]
                console.print(f"[bold green]🕰️ POLARS RANGE: {first_ts} to {last_ts}[/bold green]")
            else:
                console.print(f"[yellow]⚠️ No timestamp column found in polars data[/yellow]")

            # Map DSM columns to our expected format
            column_mapping = {}
            available_cols = df_polars.columns

            # Find timestamp column (could be 'open_time', 'timestamp', or index)
            if "open_time" in available_cols:
                column_mapping["open_time"] = "timestamp"
            elif "timestamp" not in available_cols and df_polars.height > 0:
                # Use index as timestamp if no timestamp column
                df_polars = df_polars.with_row_index("timestamp")
                # Convert row index to actual timestamps
                base_time = datetime.now() - timedelta(minutes=df_polars.height)
                df_polars = df_polars.with_columns([
                    (pl.lit(base_time) + pl.duration(minutes=pl.col("timestamp"))).alias("timestamp"),
                ])

            # Rename columns if needed
            for old_col, new_col in column_mapping.items():
                if old_col in df_polars.columns:
                    df_polars = df_polars.rename({old_col: new_col})

            # Add symbol column if missing
            if "symbol" not in df_polars.columns:
                df_polars = df_polars.with_columns(pl.lit(symbol).alias("symbol"))

            # Ensure timestamp exists and is datetime
            if "timestamp" not in df_polars.columns:
                # Create timestamp from row index as fallback
                base_time = datetime.now() - timedelta(minutes=df_polars.height)
                df_polars = df_polars.with_row_index().with_columns([
                    (pl.lit(base_time) + pl.duration(minutes=pl.col("index"))).alias("timestamp"),
                ]).drop("index")

            # Ensure timestamp is datetime type
            df_polars = df_polars.with_columns(
                pl.col("timestamp").cast(pl.Datetime),
            )

            # Display perpetual futures price information
            min_price = df_polars["close"].min()
            max_price = df_polars["close"].max()
            price_range = max_price - min_price
            console.print(f"[magenta]📈 PERPETUAL FUTURES price range: ${min_price:.2f} - ${max_price:.2f}[/magenta]")
            console.print(f"[blue]💰 Futures volatility: ${price_range:.2f} range ({price_range/min_price*100:.2f}% swing)[/blue]")

            return df_polars

        except Exception as e:
            console.print(f"[red]❌ Could not fetch real data using DSM for {symbol}: {e}[/red]")
            raise Exception(f"Data Source Manager failed to fetch {symbol} data: {e}")


    def process_ohlcv_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enhanced data processing using Polars' latest features."""
        return (
            df
            .with_columns([
                # Add technical indicators using Polars expressions
                pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
                pl.col("close").rolling_std(window_size=20).alias("volatility_20"),
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("close") - pl.col("open")).alias("change"),
                ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("change_pct"),
            ])
            .with_columns([
                # Bollinger Bands
                (pl.col("sma_20") + 2 * pl.col("volatility_20")).alias("bb_upper"),
                (pl.col("sma_20") - 2 * pl.col("volatility_20")).alias("bb_lower"),
            ])
            .sort("timestamp")
        )

    def to_nautilus_bars(
        self,
        df: pl.DataFrame,
        instrument_id: str = "BTCUSDT.BINANCE",
    ) -> list[Bar]:
        """Convert Polars DataFrame to NautilusTrader Bar objects."""
        bars = []

        # Convert to Arrow for zero-copy interop
        arrow_table = df.to_arrow()

        for batch in arrow_table.to_batches():
            batch_df = pl.from_arrow(batch)

            # Convert polars to pandas, then to dict records
            pandas_df = batch_df.to_pandas()
            for row in pandas_df.to_dict("records"):
                # Skip rows with missing data
                if (row["open"] is None or row["high"] is None or
                    row["low"] is None or row["close"] is None or row["volume"] is None):
                    continue

                # Create BarType
                bar_spec = BarSpecification(
                    step=1,
                    aggregation=BarAggregation.MINUTE,
                    price_type=PriceType.LAST,
                )
                bar_type = BarType(
                    instrument_id=InstrumentId.from_str(instrument_id),
                    bar_spec=bar_spec,
                )

                # Convert timestamp to nanoseconds
                ts_ns = dt_to_unix_nanos(row["timestamp"])

                bar = Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{float(row['open']):.5f}"),
                    high=Price.from_str(f"{float(row['high']):.5f}"),
                    low=Price.from_str(f"{float(row['low']):.5f}"),
                    close=Price.from_str(f"{float(row['close']):.5f}"),
                    volume=Quantity.from_str(str(int(float(row["volume"])))),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
                bars.append(bar)

        return bars

    def cache_to_parquet(self, df: pl.DataFrame, filename: str) -> Path:
        """Cache DataFrame to Parquet using Arrow format."""
        file_path = self.cache_dir / f"{filename}.parquet"
        df.write_parquet(file_path, use_pyarrow=True)
        return file_path

    def load_from_parquet(self, filename: str) -> Optional[pl.DataFrame]:
        """Load DataFrame from Parquet cache."""
        file_path = self.cache_dir / f"{filename}.parquet"
        if file_path.exists():
            return pl.read_parquet(file_path, use_pyarrow=True)
        return None

    def get_data_stats(self, df: pl.DataFrame) -> dict[str, Any]:
        """Get comprehensive data statistics using Polars."""
        return {
            "rows": df.height,
            "columns": df.width,
            "memory_usage_mb": df.estimated_size("mb"),
            "schema": dict(df.schema),
            "price_stats": {
                "mean": df["close"].mean(),
                "std": df["close"].std(),
                "min": df["close"].min() or 0,
                "max": df["close"].max() or 0,
                "range": (df["close"].max() or 0) - (df["close"].min() or 0),
            },
            "volume_stats": {
                "total": df["volume"].sum(),
                "mean": df["volume"].mean(),
                "max": df["volume"].max(),
            },
        }


class DataPipeline:
    """Modern data pipeline orchestrator."""

    def __init__(self, data_manager: ArrowDataManager) -> None:
        """Initialize pipeline with data manager."""
        self.data_manager = data_manager

    def run_real_data_pipeline(self, symbol: str = "EURUSD", limit: int = 500) -> dict[str, Any]:
        """Run complete data pipeline with real market data."""
        # 1. Fetch real market data
        raw_data = self.data_manager.fetch_real_market_data(symbol, limit=limit)

        # 2. Process with Polars
        processed_data = self.data_manager.process_ohlcv_data(raw_data)

        # 3. Cache to Parquet
        cache_path = self.data_manager.cache_to_parquet(processed_data, f"{symbol}_real_data")

        # 4. Convert to NautilusTrader format
        nautilus_bars = self.data_manager.to_nautilus_bars(processed_data)

        # 5. Generate statistics
        stats = self.data_manager.get_data_stats(processed_data)

        return {
            "raw_data_shape": (raw_data.height, raw_data.width),
            "processed_data_shape": (processed_data.height, processed_data.width),
            "cache_path": str(cache_path),
            "nautilus_bars_count": len(nautilus_bars),
            "stats": stats,
            "sample_bars": nautilus_bars[:5] if nautilus_bars else [],
            "data_source": "real_market_data",
        }
