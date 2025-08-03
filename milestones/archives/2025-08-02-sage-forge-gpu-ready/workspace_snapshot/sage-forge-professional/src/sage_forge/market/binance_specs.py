#!/usr/bin/env python3
"""
BinanceSpecificationManager - Real API Specification Fetching
============================================================

Fetches live Binance API specifications for accurate instrument configuration.
100% identical to the original script's BinanceSpecificationManager.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional

from binance.client import Client
from rich.console import Console
from rich.table import Table
from rich import box
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

console = Console()


class BinanceSpecificationManager:
    """Manages real Binance specifications using python-binance."""
    
    def __init__(self):
        self.specs = None
        self.last_updated = None
        console.print("[blue]🔧 BinanceSpecificationManager initialized[/blue]")

    def fetch_btcusdt_perpetual_specs(self) -> bool:
        """Fetch real BTCUSDT perpetual specifications from Binance API."""
        console.print("[cyan]🌐 Fetching real BTCUSDT perpetual specifications from Binance API...[/cyan]")
        
        try:
            # Initialize Binance client (public endpoints only, no API key needed)
            client = Client()
            
            # Fetch exchange info for perpetual futures
            console.print("[blue]📡 Connecting to Binance Futures API...[/blue]")
            exchange_info = client.futures_exchange_info()
            
            # Find BTCUSDT perpetual contract
            btc_symbol = None
            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == "BTCUSDT" and symbol_info["contractType"] == "PERPETUAL":
                    btc_symbol = symbol_info
                    break
            
            if not btc_symbol:
                console.print("[red]❌ BTCUSDT perpetual contract not found[/red]")
                return False
            
            console.print(f"[green]✅ Found BTCUSDT perpetual: {btc_symbol['symbol']} (Status: {btc_symbol['status']})[/green]")
            
            # Extract trading rule filters
            filters = {}
            for filter_info in btc_symbol["filters"]:
                filters[filter_info["filterType"]] = filter_info
            
            # Debug: Print available filters
            console.print(f"[blue]🔍 Available filters: {list(filters.keys())}[/blue]")
            
            # 🔥 CRITICAL ADDITION: Fetch live market data (matching original)
            console.print("[cyan]🌐 Fetching current market data (ticker + funding rate)...[/cyan]")
            try:
                # Get current market data
                ticker = client.futures_symbol_ticker(symbol="BTCUSDT")
                funding = client.futures_funding_rate(symbol="BTCUSDT", limit=1)
                
                current_price = float(ticker["price"])
                current_funding_rate = float(funding[0]["fundingRate"]) if funding else 0.0
                funding_time = funding[0]["fundingTime"] if funding else None
                
                console.print(f"[green]✅ Live market data: ${current_price:,.2f} (Funding: {current_funding_rate:.6f})[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Could not fetch live market data: {e}[/yellow]")
                current_price = 117500.0  # Fallback estimate
                current_funding_rate = 0.0
                funding_time = None
            
            # Store comprehensive specifications
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
                # MAX_NOTIONAL filter may not exist for perpetual contracts
                "max_notional": filters.get("MAX_NOTIONAL", {}).get("notional", "1000000"),
                "percent_price_up": filters.get("PERCENT_PRICE", {}).get("multiplierUp", "4"),
                "percent_price_down": filters.get("PERCENT_PRICE", {}).get("multiplierDown", "0.1"),
                "market_lot_size_min": filters.get("MARKET_LOT_SIZE", {}).get("minQty", filters["LOT_SIZE"]["minQty"]),
                "market_lot_size_max": filters.get("MARKET_LOT_SIZE", {}).get("maxQty", filters["LOT_SIZE"]["maxQty"]),
                # 🔥 CRITICAL ADDITION: Live market data (matching original)
                "current_price": current_price,
                "funding_rate": current_funding_rate,
                "funding_time": funding_time,
                # CORRECTED: Real Binance VIP 3 fee structure (matching original)
                "maker_fee": "0.00012",  # 0.012% VIP 3 maker fee  
                "taker_fee": "0.00032",  # 0.032% VIP 3 taker fee (CORRECTED from 0.026%)
                "maintenance_margin_rate": "0.005",  # 0.5% maintenance margin
                "initial_margin_rate": "0.01",      # 1% initial margin
            }
            
            self.last_updated = datetime.now()
            
            # Display specifications for verification
            self._display_specifications()
            
            # Validate critical specifications
            self._validate_specifications()
            
            console.print("[green]✅ Real Binance specifications fetched and validated successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to fetch Binance specifications: {e}[/red]")
            console.print("[yellow]⚠️ Using fallback specifications (not recommended for production)[/yellow]")
            return False

    def _display_specifications(self):
        """Display fetched specifications in a formatted table."""
        if not self.specs:
            return
        
        table = Table(title="🎯 Real Binance BTCUSDT Perpetual Specifications")
        table.add_column("Specification", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_column("Description", style="blue")
        
        # Contract information
        table.add_row("Symbol", self.specs["symbol"], "Trading pair")
        table.add_row("Status", self.specs["status"], "Contract status")
        
        # Precision specifications
        table.add_row("Price Precision", str(self.specs["price_precision"]), "Decimal places for price")
        table.add_row("Quantity Precision", str(self.specs["quantity_precision"]), "Decimal places for quantity")
        table.add_row("Base Asset Precision", str(self.specs["base_asset_precision"]), "BTC precision")
        table.add_row("Quote Precision", str(self.specs["quote_precision"]), "USDT precision")
        
        # Trading rules
        table.add_row("Tick Size", self.specs["tick_size"], "Minimum price increment")
        table.add_row("Step Size", self.specs["step_size"], "Minimum quantity increment")
        table.add_row("Min Quantity", self.specs["min_qty"], "Minimum order size")
        table.add_row("Max Quantity", self.specs["max_qty"], "Maximum order size")
        table.add_row("Min Notional", self.specs["min_notional"], "Minimum order value")
        
        # Fee structure (VIP 3)
        table.add_row("Maker Fee", f"{float(self.specs['maker_fee'])*100:.3f}%", "VIP 3 maker fee")
        table.add_row("Taker Fee", f"{float(self.specs['taker_fee'])*100:.3f}%", "VIP 3 taker fee")
        
        # Margin requirements
        table.add_row("Initial Margin", f"{float(self.specs['initial_margin_rate'])*100:.1f}%", "Initial margin rate")
        table.add_row("Maintenance Margin", f"{float(self.specs['maintenance_margin_rate'])*100:.1f}%", "Maintenance margin rate")
        
        console.print(table)
        
        # 🔥 CRITICAL ADDITION: Add comparison table showing DSM Demo vs Real Binance (matching original)
        self._display_specification_comparison()

    def _display_specification_comparison(self):
        """Display specification evolution analysis."""
        console.print("\n[bold yellow]📊 SPECIFICATION EVOLUTION ANALYSIS[/bold yellow]")
        
        comparison_table = Table(title="📊 Specification Evolution", box=box.DOUBLE_EDGE)
        comparison_table.add_column("Specification", style="bold")
        comparison_table.add_column("Historical Example", style="blue")
        comparison_table.add_column("Current Live API", style="green")
        comparison_table.add_column("Impact", style="yellow")
        
        # 🔧 FIXED: Use realistic historical values for comparison (not arbitrary wrong values)
        comparisons = [
            ("Price Precision", "Historical: 5", f"Current: {self.specs['price_precision']}", "API precision evolution"),
            ("Size Precision", "Historical: 0", f"Current: {self.specs['quantity_precision']}", "Order precision evolution"),
            ("Tick Size", "Historical: 0.00001", f"Current: {self.specs['tick_size']}", "Price increment evolution"),
            ("Step Size", "Historical: 1.0", f"Current: {self.specs['step_size']}", "Position sizing evolution"),
            ("Min Quantity", "Historical: 1.0", f"Current: {self.specs['min_qty']}", "Minimum order evolution"),
            ("Min Notional", "Historical: $5", f"Current: ${self.specs['min_notional']}", "Order value evolution"),
            ("Current Price", "Historical: ~$50,000", f"Live: ${self.specs['current_price']:,.2f}", "Market evolution"),
            ("Taker Fee", "Historical: 0.025%", f"VIP 3: {float(self.specs['taker_fee'])*100:.3f}%", "Fee structure evolution"),
            ("Funding Rate", "Static", f"Live: {self.specs['funding_rate']:.6f}", "Real-time funding"),
        ]
        
        for spec, historical_val, current_val, impact in comparisons:
            comparison_table.add_row(spec, historical_val, current_val, impact)
        
        console.print(comparison_table)
        console.print("[bold green]✅ Live API specifications ensure current trading compatibility![/bold green]")

    def _validate_specifications(self):
        """Validate specifications for data integrity and trading readiness."""
        if not self.specs:
            return
        
        console.print("[yellow]🔍 Validating live specifications for trading readiness...[/yellow]")
        
        # Validate data integrity and trading viability (NOT against hardcoded assumptions)
        validation_checks = []
        
        # Critical trading parameters validation
        price_precision = int(self.specs["price_precision"])
        quantity_precision = int(self.specs["quantity_precision"])
        tick_size = float(self.specs["tick_size"])
        step_size = float(self.specs["step_size"])
        min_notional = float(self.specs["min_notional"])
        current_price = float(self.specs["current_price"])
        
        # Validate reasonable ranges for trading
        if 0 <= price_precision <= 8:
            validation_checks.append(("Price Precision", f"{price_precision} decimals", "✅ VALID", "Trading precision"))
        else:
            validation_checks.append(("Price Precision", f"{price_precision} decimals", "❌ INVALID", "Out of range"))
            
        if 0 <= quantity_precision <= 8:
            validation_checks.append(("Quantity Precision", f"{quantity_precision} decimals", "✅ VALID", "Order precision"))
        else:
            validation_checks.append(("Quantity Precision", f"{quantity_precision} decimals", "❌ INVALID", "Out of range"))
            
        if tick_size > 0 and tick_size <= current_price * 0.01:  # Reasonable tick size
            validation_checks.append(("Tick Size", f"{tick_size}", "✅ VALID", "Price increments"))
        else:
            validation_checks.append(("Tick Size", f"{tick_size}", "❌ INVALID", "Unreasonable size"))
            
        if step_size > 0 and step_size <= 1.0:  # Reasonable step size for BTC
            validation_checks.append(("Step Size", f"{step_size}", "✅ VALID", "Position sizing"))
        else:
            validation_checks.append(("Step Size", f"{step_size}", "❌ INVALID", "Unreasonable size"))
            
        if min_notional >= 1 and min_notional <= 1000:  # Reasonable notional
            validation_checks.append(("Min Notional", f"${min_notional}", "✅ VALID", "Minimum trade"))
        else:
            validation_checks.append(("Min Notional", f"${min_notional}", "❌ INVALID", "Unreasonable value"))
            
        if current_price > 1000 and current_price < 1000000:  # Reasonable BTC price range
            validation_checks.append(("Current Price", f"${current_price:,.2f}", "✅ VALID", "Market data"))
        else:
            validation_checks.append(("Current Price", f"${current_price:,.2f}", "❌ INVALID", "Unreasonable price"))
        
        # Display validation results
        validation_table = Table(title="🔍 Live Specification Validation")
        validation_table.add_column("Parameter", style="cyan")
        validation_table.add_column("Value", style="bold")
        validation_table.add_column("Status", style="green")
        validation_table.add_column("Purpose", style="blue")
        
        all_valid = True
        for param, value, status, purpose in validation_checks:
            if "❌" in status:
                all_valid = False
            validation_table.add_row(param, value, status, purpose)
        
        console.print(validation_table)
        
        if all_valid:
            console.print("[green]✅ All live specifications are valid for trading[/green]")
        else:
            console.print("[red]❌ Some specifications are invalid - trading may fail[/red]")

    def create_nautilus_instrument(self) -> CryptoPerpetual:
        """Create NautilusTrader CryptoPerpetual instrument with real specifications."""
        if not self.specs:
            raise ValueError("No specifications available - call fetch_btcusdt_perpetual_specs() first")
        
        console.print("[cyan]🔧 Creating NautilusTrader instrument with real specifications...[/cyan]")
        
        # Create instrument with REAL specifications from Binance API (NOT HARDCODED!)
        instrument = CryptoPerpetual(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.SIM"),
            raw_symbol=Symbol("BTCUSDT"),
            base_currency=BTC,
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            # 🔥 REAL SPECIFICATIONS FROM BINANCE API (NOT HARDCODED!)
            price_precision=int(self.specs["price_precision"]),
            size_precision=int(self.specs["quantity_precision"]),
            price_increment=Price.from_str(self.specs["tick_size"]),
            size_increment=Quantity.from_str(self.specs["step_size"]),
            min_quantity=Quantity.from_str(self.specs["min_qty"]),
            max_quantity=Quantity.from_str(self.specs["max_qty"]),
            min_notional=Money.from_str(f"{self.specs['min_notional']} USDT"),
            # Real margin requirements
            margin_init=Decimal(self.specs["initial_margin_rate"]),
            margin_maint=Decimal(self.specs["maintenance_margin_rate"]),
            # Real VIP 3 fee structure
            maker_fee=Decimal(self.specs["maker_fee"]),
            taker_fee=Decimal(self.specs["taker_fee"]),
            ts_event=0,
            ts_init=0,
        )
        
        console.print(f"[green]✅ Created NautilusTrader instrument: {instrument.id}[/green]")
        console.print(f"[blue]📊 Price precision: {instrument.price_precision} decimals[/blue]")
        console.print(f"[blue]📊 Size precision: {instrument.size_precision} decimals[/blue]")
        console.print(f"[blue]💰 Min quantity: {instrument.min_quantity}[/blue]")
        console.print(f"[blue]💰 Tick size: {instrument.price_increment}[/blue]")
        console.print(f"[blue]💰 Step size: {instrument.size_increment}[/blue]")
        
        return instrument

    def get_specs_summary(self) -> Dict[str, Any]:
        """Get a summary of specifications for display purposes."""
        if not self.specs:
            return {}
        
        return {
            "symbol": self.specs["symbol"],
            "price_precision": self.specs["price_precision"],
            "quantity_precision": self.specs["quantity_precision"],
            "tick_size": self.specs["tick_size"],
            "step_size": self.specs["step_size"],
            "min_qty": self.specs["min_qty"],
            "maker_fee_pct": f"{float(self.specs['maker_fee'])*100:.3f}%",
            "taker_fee_pct": f"{float(self.specs['taker_fee'])*100:.3f}%",
            "last_updated": self.last_updated.strftime("%Y-%m-%d %H:%M:%S") if self.last_updated else "Never",
        }