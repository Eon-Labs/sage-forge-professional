"""
Binance specification manager following NautilusTrader patterns.

Fetches real Binance specifications for accurate backtesting.
"""

from datetime import datetime
from typing import Any

from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from rich.console import Console
from rich.table import Table

console = Console()


class BinanceSpecificationManager:
    """
    Manages real Binance specifications using python-binance.
    
    Follows NautilusTrader patterns for data providers and instrument creation.
    """

    def __init__(self):
        self.specs: dict[str, Any] | None = None
        self.last_updated: datetime | None = None

    def fetch_btcusdt_perpetual_specs(self) -> bool:
        """Fetch current BTCUSDT perpetual futures specifications."""
        try:
            from binance import Client

            console.print(
                "[bold blue]🔍 Fetching Real Binance BTCUSDT-PERP "
                "Specifications...[/bold blue]"
            )

            client = Client()
            exchange_info = client.futures_exchange_info()
            btc_symbol = next(s for s in exchange_info["symbols"] if s["symbol"] == "BTCUSDT")
            filters = {f["filterType"]: f for f in btc_symbol["filters"]}

            # Get current market data
            ticker = client.futures_symbol_ticker(symbol="BTCUSDT")
            funding = client.futures_funding_rate(symbol="BTCUSDT", limit=1)

            self.specs = {
                "symbol": btc_symbol["symbol"],
                "status": btc_symbol["status"],
                "price_precision": btc_symbol["pricePrecision"],
                "quantity_precision": btc_symbol["quantityPrecision"],
                "base_asset_precision": btc_symbol["baseAssetPrecision"],
                "quote_precision": btc_symbol["quotePrecision"],
                "tick_size": filters["PRICE_FILTER"]["tickSize"],
                "step_size": filters["LOT_SIZE"]["stepSize"],
                "min_qty": filters["LOT_SIZE"]["minQty"],
                "max_qty": filters["LOT_SIZE"]["maxQty"],
                "min_notional": filters["MIN_NOTIONAL"]["notional"],
                "current_price": float(ticker["price"]),
                "funding_rate": float(funding[0]["fundingRate"]) if funding else 0.0,
                "funding_time": funding[0]["fundingTime"] if funding else None,
            }

            self.last_updated = datetime.now()
            console.print("✅ Successfully fetched real Binance specifications")
            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to fetch Binance specs: {e}[/red]")
            return False

    def create_nautilus_instrument(self) -> CryptoPerpetual:
        """Create NautilusTrader instrument with REAL Binance specifications."""
        if not self.specs:
            raise ValueError("Must fetch specifications first")

        console.print(
            "[bold green]🔧 Creating NautilusTrader Instrument "
            "with REAL Specs...[/bold green]"
        )

        # Display specification comparison
        self._display_specification_comparison()

        # Create instrument with real specifications
        from decimal import Decimal
        
        instrument = CryptoPerpetual(
            instrument_id=InstrumentId(
                symbol=Symbol("BTCUSDT-PERP"),
                venue=Venue("SIM"),
            ),
            raw_symbol=Symbol("BTCUSDT"),
            base_currency=BTC,
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=self.specs["price_precision"],
            size_precision=self.specs["quantity_precision"],
            price_increment=Price.from_str(self.specs["tick_size"]),
            size_increment=Quantity.from_str(self.specs["step_size"]),
            max_quantity=Quantity.from_str(self.specs["max_qty"]),
            min_quantity=Quantity.from_str(self.specs["min_qty"]),
            max_notional=None,
            min_notional=Money.from_str(f"{self.specs['min_notional']} USDT"),
            max_price=Price.from_str("1000000.00"),
            min_price=Price.from_str(self.specs["tick_size"]),
            margin_init=Decimal("0.0500"),
            margin_maint=Decimal("0.0250"),
            maker_fee=Decimal("0.000120"),
            taker_fee=Decimal("0.000320"),
            ts_event=0,
            ts_init=0,
        )

        console.print(
            "✅ NautilusTrader instrument created with REAL specifications"
        )
        return instrument

    def _display_specification_comparison(self) -> None:
        """Display comparison between demo and real specifications."""
        table = Table(title="⚔️ Specification Correction")
        table.add_column("Specification", style="bold")
        table.add_column("DSM Demo (WRONG)", style="")
        table.add_column("Real Binance (CORRECT)", style="")
        table.add_column("Impact", style="")

        # Compare specifications
        comparisons = [
            ("Price Precision", "5", str(self.specs["price_precision"]), "API accuracy"),
            ("Size Precision", "0", str(self.specs["quantity_precision"]), "Order precision"),
            ("Tick Size", "0.00001", self.specs["tick_size"], "Price increments"),
            ("Step Size", "1", self.specs["step_size"], "Position sizing"),
            ("Min Quantity", "1", self.specs["min_qty"], "Minimum orders"),
            ("Min Notional", "$5", f"${self.specs['min_notional']}", "Order value"),
        ]

        for spec, demo_val, real_val, impact in comparisons:
            table.add_row(spec, demo_val, real_val, impact)

        console.print(table)

    def get_current_price(self) -> float:
        """Get current BTC price from specifications."""
        if not self.specs:
            raise ValueError("Must fetch specifications first")
        return self.specs["current_price"]

    def get_funding_rate(self) -> float:
        """Get current funding rate."""
        if not self.specs:
            raise ValueError("Must fetch specifications first")
        return self.specs["funding_rate"]
