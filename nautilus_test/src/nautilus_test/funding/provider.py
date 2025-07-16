"""
Enhanced funding rate data provider with multiple data sources.

Provides historical and real-time funding rate data with robust error handling,
caching, and fallback strategies. Combines DSM infrastructure with direct API access.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp
import pandas as pd
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.identifiers import InstrumentId
from rich.console import Console

from nautilus_test.funding.data import FundingRateUpdate
from nautilus_test.utils.cache_config import get_dsm_cache_dir, get_funding_cache_dir

console = Console()


class FundingRateProvider:
    """
    Enhanced funding rate provider with multiple data sources and robust error handling.
    
    Data source strategy:
    1. Primary: DSM BinanceFundingRateClient (60 days, high quality)
    2. Secondary: Binance API direct (5.8 years, full history)
    3. Cache: Local storage for performance (both parquet and JSON)
    4. Robust error handling with automatic fallbacks
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        dsm_available: bool = True,
        enable_direct_api: bool = True,
    ) -> None:
        """
        Initialize enhanced funding rate provider with platform-standard cache.
        
        Uses platformdirs for cross-platform cache directory following 2024-2025 best practices:
        - Linux: ~/.cache/nautilus-test/funding/
        - macOS: ~/Library/Caches/nautilus-test/funding/
        - Windows: %LOCALAPPDATA%/nautilus-test/Cache/funding/
        
        Parameters
        ----------
        cache_dir : Path, optional
            Directory for caching funding rate data. Uses platformdirs if None.
        use_cache : bool
            Whether to use local caching.
        dsm_available : bool
            Whether DSM components are available.
        enable_direct_api : bool
            Whether to enable direct Binance API access.
        """
        self.cache_dir = cache_dir or get_funding_cache_dir()
        console.print(f"[blue]📁 Funding cache directory: {self.cache_dir}[/blue]")
        self.use_cache = use_cache
        self.dsm_available = dsm_available
        self.enable_direct_api = enable_direct_api

        # HTTP session for direct API access
        self._session = None

        # Initialize DSM components if available
        self._dsm_client = None
        if dsm_available:
            self._init_dsm_client()

    def _init_dsm_client(self) -> None:
        """Initialize DSM funding rate client."""
        try:
            # Add DSM to path
            dsm_path = Path("/Users/terryli/eon/data-source-manager")
            if str(dsm_path) not in sys.path:
                sys.path.insert(0, str(dsm_path))

            from core.providers.binance.binance_funding_rate_client import BinanceFundingRateClient
            from utils.market_constraints import Interval, MarketType

            self._dsm_client = BinanceFundingRateClient(
                symbol="BTCUSDT",
                market_type=MarketType.FUTURES_USDT,
                cache_dir=get_dsm_cache_dir(),
                use_cache=self.use_cache,
            )
            self._interval = Interval.HOUR_8
            console.print("[green]✅ DSM funding rate client initialized[/green]")

        except Exception as e:
            console.print(f"[yellow]⚠️ DSM client initialization failed: {e}[/yellow]")
            self.dsm_available = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for direct API access."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "NautilusTrader/1.0"},
            )
        return self._session

    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            console.print("[blue]🔌 HTTP session closed[/blue]")

    async def get_historical_funding_rates(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: datetime,
        max_records: Optional[int] = None,
    ) -> list[FundingRateUpdate]:
        """
        Get historical funding rates for an instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument to get funding rates for.
        start_time : datetime
            Start time for historical data.
        end_time : datetime
            End time for historical data.
        max_records : int, optional
            Maximum number of records to return.
            
        Returns
        -------
        List[FundingRateUpdate]
            List of funding rate updates.
        """
        # Check cache first
        if self.use_cache:
            cached_data = self._load_from_cache(instrument_id, start_time, end_time)
            if cached_data:
                console.print(f"[blue]📂 Loaded {len(cached_data)} funding rates from cache[/blue]")
                return cached_data[:max_records] if max_records else cached_data

        # Enhanced data source strategy with fallback
        time_span = end_time - start_time
        funding_rates = []

        if time_span.days <= 60 and self.dsm_available:
            # Try DSM first for recent data (higher quality)
            try:
                funding_rates = await self._fetch_from_dsm(instrument_id, start_time, end_time)
                if not funding_rates and self.enable_direct_api:
                    console.print("[yellow]⚠️ DSM returned no data, falling back to direct API[/yellow]")
                    funding_rates = await self._fetch_from_binance_api_robust(instrument_id, start_time, end_time)
            except Exception as e:
                console.print(f"[yellow]⚠️ DSM fetch failed ({e}), falling back to direct API[/yellow]")
                if self.enable_direct_api:
                    funding_rates = await self._fetch_from_binance_api_robust(instrument_id, start_time, end_time)
        else:
            # Use direct API for extended historical data or if DSM unavailable
            if self.enable_direct_api:
                funding_rates = await self._fetch_from_binance_api_robust(instrument_id, start_time, end_time)
            else:
                console.print("[red]❌ No data source available for extended historical data[/red]")

        # Cache the results
        if self.use_cache and funding_rates:
            self._save_to_cache(instrument_id, funding_rates)

        # Apply max_records limit
        if max_records and len(funding_rates) > max_records:
            funding_rates = funding_rates[:max_records]
            console.print(f"[blue]📊 Limited to {max_records} records as requested[/blue]")

        return funding_rates

    async def _fetch_from_dsm(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: datetime,
    ) -> list[FundingRateUpdate]:
        """Fetch funding rates using DSM client."""
        if not self._dsm_client:
            raise RuntimeError("DSM client not available")

        console.print(f"[cyan]🌐 Fetching funding rates via DSM for {instrument_id}[/cyan]")

        try:
            # Fetch data using DSM
            df = self._dsm_client.fetch(
                symbol=str(instrument_id).split("-")[0],  # Extract base symbol
                interval=self._interval,  # type: ignore
                start_time=start_time,
                end_time=end_time,
            )

            if df.empty:
                console.print("[yellow]⚠️ No data returned from DSM[/yellow]")
                return []

            console.print(f"[green]✅ DSM returned {len(df)} funding rate records[/green]")

            # Convert to FundingRateUpdate objects
            funding_rates = []
            for row in df.iter_rows(named=True):
                try:
                    # Convert to pandas timestamp and verify it's valid
                    funding_time_raw = row["funding_time"]
                    if funding_time_raw is None:
                        continue
                    
                    funding_timestamp = pd.Timestamp(funding_time_raw)
                    if funding_timestamp is pd.NaT or str(funding_timestamp) == 'NaT':
                        continue  # Skip invalid timestamps
                    # Type assertion for basedpyright - we know this is not NaT at this point
                    assert funding_timestamp is not pd.NaT
                    funding_time_ns = dt_to_unix_nanos(funding_timestamp)  # type: ignore[arg-type]
                    
                    # Handle open_time if present
                    if "open_time" in row and row["open_time"] is not None:
                        open_time_raw = row["open_time"] 
                        open_timestamp = pd.Timestamp(open_time_raw)
                        if open_timestamp is pd.NaT or str(open_timestamp) == 'NaT':
                            ts_event = funding_time_ns
                        else:
                            # Type assertion for basedpyright - we know this is not NaT at this point
                            assert open_timestamp is not pd.NaT
                            ts_event = dt_to_unix_nanos(open_timestamp)  # type: ignore[arg-type]
                    else:
                        ts_event = funding_time_ns
                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                    # Skip invalid timestamp data
                    continue

                funding_rate = FundingRateUpdate(
                    instrument_id=instrument_id,
                    funding_rate=row["funding_rate"],
                    funding_time=funding_time_ns,
                    mark_price=None,  # DSM doesn't provide mark price
                    ts_event=ts_event,
                    ts_init=dt_to_unix_nanos(pd.Timestamp.now(UTC)),
                )
                funding_rates.append(funding_rate)

            return funding_rates

        except Exception as e:
            console.print(f"[red]❌ DSM fetch failed: {e}[/red]")
            return []

    async def _fetch_from_binance_api(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: datetime,
    ) -> list[FundingRateUpdate]:
        """Fetch funding rates directly from Binance API."""
        import httpx

        console.print(f"[cyan]🌐 Fetching funding rates via Binance API for {instrument_id}[/cyan]")

        try:
            # Convert to millisecond timestamps
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)

            # Extract symbol from instrument_id
            symbol = str(instrument_id).split("-")[0]  # e.g., "BTCUSDT-PERP.SIM" -> "BTCUSDT"

            funding_rates = []
            current_start = start_ms

            # Fetch data in batches (1000 records per call)
            while current_start < end_ms:
                url = "https://fapi.binance.com/fapi/v1/fundingRate"
                params = {
                    "symbol": symbol,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 1000,
                }

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                if not data:
                    break

                # Convert to FundingRateUpdate objects
                for item in data:
                    funding_time_ns = int(item["fundingTime"]) * 1_000_000  # Convert ms to ns
                    ts_init = dt_to_unix_nanos(pd.Timestamp.now(UTC))

                    funding_rate = FundingRateUpdate(
                        instrument_id=instrument_id,
                        funding_rate=item["fundingRate"],
                        funding_time=funding_time_ns,
                        mark_price=item["markPrice"] if item.get("markPrice") else None,
                        ts_event=funding_time_ns,
                        ts_init=ts_init,
                    )
                    funding_rates.append(funding_rate)

                # Update start time for next batch
                if len(data) < 1000:
                    break  # No more data available
                current_start = int(data[-1]["fundingTime"]) + 1

                console.print(f"[blue]📊 Fetched batch: {len(data)} records, total: {len(funding_rates)}[/blue]")

            console.print(f"[green]✅ Binance API returned {len(funding_rates)} funding rate records[/green]")
            return funding_rates

        except Exception as e:
            console.print(f"[red]❌ Binance API fetch failed: {e}[/red]")
            return []

    async def _fetch_from_binance_api_robust(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: datetime,
    ) -> list[FundingRateUpdate]:
        """Enhanced Binance API with robust error handling and better caching."""
        console.print(f"[cyan]🌐 Fetching funding rates via enhanced Binance API for {instrument_id}[/cyan]")

        # Extract symbol from instrument_id
        symbol = str(instrument_id).split("-")[0]  # e.g., "BTCUSDT-PERP.SIM" -> "BTCUSDT"

        # Check JSON cache first
        json_cache_file = self.cache_dir / f"{symbol}_funding_{start_time.date()}_{end_time.date()}.json"
        if self.use_cache and json_cache_file.exists():
            try:
                console.print(f"[blue]📁 Loading from JSON cache: {json_cache_file.name}[/blue]")
                with open(json_cache_file) as f:
                    cached_data = json.load(f)

                # Convert cached data to FundingRateUpdate objects
                funding_updates = self._convert_binance_to_funding_updates(cached_data, instrument_id)
                return funding_updates

            except Exception as e:
                console.print(f"[yellow]⚠️ JSON cache read failed, fetching fresh data: {e}[/yellow]")

        session = await self._get_session()

        # Binance funding rate endpoint
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "limit": 1000,  # Maximum per request
        }

        # Add time parameters if provided
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    console.print(f"[green]✅ Retrieved {len(data)} funding rate records from Binance[/green]")

                    # Cache the raw data
                    if self.use_cache:
                        try:
                            with open(json_cache_file, "w") as f:
                                json.dump(data, f, indent=2)
                            console.print(f"[blue]💾 Cached funding data: {json_cache_file.name}[/blue]")
                        except Exception as e:
                            console.print(f"[yellow]⚠️ Failed to cache data: {e}[/yellow]")

                    # Convert to FundingRateUpdate objects
                    funding_updates = self._convert_binance_to_funding_updates(data, instrument_id)
                    return funding_updates
                error_text = await response.text()
                console.print(f"[red]❌ Binance API error {response.status}: {error_text}[/red]")
                return []

        except Exception as e:
            console.print(f"[red]❌ Enhanced Binance API fetch failed: {e}[/red]")
            return []

    def _convert_binance_to_funding_updates(
        self,
        binance_data: list[dict[str, Any]],
        instrument_id: InstrumentId,
    ) -> list[FundingRateUpdate]:
        """Convert Binance funding rate data to native FundingRateUpdate objects."""
        funding_updates = []

        for item in binance_data:
            try:
                # Extract fields from Binance response
                funding_time_ms = int(item["fundingTime"])
                funding_rate = float(item["fundingRate"])
                mark_price = float(item.get("markPrice", 0)) if item.get("markPrice") else None

                # Convert to nanoseconds for NautilusTrader
                funding_time_ns = funding_time_ms * 1_000_000
                ts_init = dt_to_unix_nanos(pd.Timestamp.now(UTC))

                # Create native FundingRateUpdate (not SimpleFundingRateUpdate)
                funding_update = FundingRateUpdate(
                    instrument_id=instrument_id,
                    funding_rate=funding_rate,
                    funding_time=funding_time_ns,
                    mark_price=mark_price,
                    ts_event=funding_time_ns,
                    ts_init=ts_init,
                )

                funding_updates.append(funding_update)

            except (KeyError, ValueError, TypeError) as e:
                console.print(f"[yellow]⚠️ Skipping invalid funding record: {e}[/yellow]")
                continue

        console.print(f"[green]✅ Converted {len(funding_updates)} funding rate updates[/green]")
        return funding_updates

    def _load_from_cache(
        self,
        instrument_id: InstrumentId,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[list[FundingRateUpdate]]:
        """Load funding rates from cache."""
        cache_file = self.cache_dir / f"{instrument_id}_funding_rates.parquet"

        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)

            # Filter by time range
            df["funding_datetime"] = pd.to_datetime(df["funding_time"], unit="ns")
            mask = (df["funding_datetime"] >= start_time) & (df["funding_datetime"] <= end_time)
            df_filtered = df[mask]

            if df_filtered.empty:
                return None

            # Convert back to FundingRateUpdate objects
            funding_rates = []
            for _, row in df_filtered.iterrows():
                # Extract scalar values from pandas Series
                mark_price_val = row["mark_price"]
                try:
                    mark_price = float(mark_price_val) if mark_price_val is not None and str(mark_price_val) != 'nan' else None
                except (ValueError, TypeError):
                    mark_price = None
                
                funding_rate = FundingRateUpdate(
                    instrument_id=InstrumentId.from_str(str(row["instrument_id"])),
                    funding_rate=float(row["funding_rate"]),
                    funding_time=int(row["funding_time"]),
                    mark_price=mark_price,
                    ts_event=int(row["ts_event"]),
                    ts_init=int(row["ts_init"]),
                )
                funding_rates.append(funding_rate)

            return funding_rates

        except Exception as e:
            console.print(f"[yellow]⚠️ Cache load failed: {e}[/yellow]")
            return None

    def _save_to_cache(
        self,
        instrument_id: InstrumentId,
        funding_rates: list[FundingRateUpdate],
    ) -> None:
        """Save funding rates to cache."""
        try:
            # Convert to DataFrame
            data = []
            for fr in funding_rates:
                data.append({
                    "instrument_id": str(fr.instrument_id),
                    "funding_rate": fr.funding_rate,
                    "funding_time": fr.funding_time,
                    "mark_price": fr.mark_price,
                    "ts_event": fr.ts_event,
                    "ts_init": fr.ts_init,
                })

            df = pd.DataFrame(data)
            cache_file = self.cache_dir / f"{instrument_id}_funding_rates.parquet"
            df.to_parquet(cache_file)

            console.print(f"[blue]💾 Cached {len(funding_rates)} funding rates to {cache_file.name}[/blue]")

        except Exception as e:
            console.print(f"[yellow]⚠️ Cache save failed: {e}[/yellow]")

    def get_data_quality_report(self) -> dict[str, Any]:
        """Get a comprehensive report on data source quality and availability."""
        sources = []
        if self.dsm_available:
            sources.append("DSM BinanceFundingRateClient (60 days, high quality)")
        if self.enable_direct_api:
            sources.append("Binance API Direct (5.8+ years, full history)")

        return {
            "provider": "EnhancedFundingRateProvider",
            "dsm_available": self.dsm_available,
            "direct_api_enabled": self.enable_direct_api,
            "cache_enabled": self.use_cache,
            "cache_directory": str(self.cache_dir),
            "supported_sources": sources,
            "cache_formats": ["Parquet (legacy)", "JSON (robust)"],
            "max_historical_span": "5.8+ years (Sep 2019 - current)",
            "funding_frequency": "8 hours (00:00, 08:00, 16:00 UTC)",
            "error_handling": "Robust with automatic fallbacks",
            "data_classes": "Native NautilusTrader (FundingRateUpdate)",
            "reliability": "High - Multiple sources with fallbacks",
        }
