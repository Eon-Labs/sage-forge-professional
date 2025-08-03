#!/usr/bin/env python3
"""
Backtesting Specification Manager - Streamlined
===============================================

Minimal specification manager focused only on P&L-affecting parameters
for backtesting scenarios. Removes execution constraints that are 
handled naturally by historical OHLCV data.
"""

from datetime import datetime
from typing import Dict, Any
from binance.client import Client
from rich.console import Console

console = Console()


class BacktestingSpecificationManager:
    """Streamlined specification manager for backtesting - only P&L essentials."""
    
    def __init__(self):
        """Initialize with minimal Binance client setup."""
        try:
            # Minimal client setup for public endpoints only
            self.client = Client()
            console.print("[blue]🔧 BacktestingSpecificationManager initialized (P&L essentials only)[/blue]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Binance client setup warning: {e}[/yellow]")
            self.client = None
        
        self.specs = None
        self.last_updated = None

    def fetch_essential_specs(self) -> bool:
        """Fetch only specifications that directly affect P&L in backtesting."""
        if not self.client:
            console.print("[yellow]⚠️ Using fallback specifications (no client available)[/yellow]")
            return self._use_fallback_specs()
        
        try:
            console.print("[cyan]📊 Fetching essential P&L specifications for backtesting...[/cyan]")
            
            # Get current funding rate (affects position P&L)
            funding_rate = self._get_funding_rate()
            
            # Get current price (optional reference)
            current_price = self._get_current_price()
            
            # Essential specifications for backtesting
            self.specs = {
                # DIRECT P&L IMPACT - ESSENTIAL
                "maker_fee": "0.00012",    # 0.012% VIP 3 maker fee
                "taker_fee": "0.00032",    # 0.032% VIP 3 taker fee  
                "funding_rate": funding_rate,
                
                # OPTIONAL CONTEXT
                "symbol": "BTCUSDT",
                "current_price": current_price,
                "last_updated": datetime.now().isoformat(),
                
                # REASONABLE DEFAULTS (historical data provides actual constraints)
                "price_precision": 2,      # Reasonable default
                "quantity_precision": 3,   # Reasonable default
                "tick_size": "0.01",      # Reasonable default
                "step_size": "0.001",     # Reasonable default
                "min_qty": "0.001",       # Reasonable default
                "min_notional": "100"     # Reasonable default
            }
            
            self.last_updated = datetime.now()
            
            console.print(f"[green]✅ Essential specifications fetched successfully[/green]")
            console.print(f"[blue]💰 Maker Fee: {float(self.specs['maker_fee'])*100:.3f}% | Taker Fee: {float(self.specs['taker_fee'])*100:.3f}%[/blue]")
            console.print(f"[blue]📊 Funding Rate: {self.specs['funding_rate']:.6f} | Price: ${self.specs['current_price']:,.2f}[/blue]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to fetch essential specifications: {e}[/red]")
            return self._use_fallback_specs()

    def _get_funding_rate(self) -> float:
        """Get current funding rate for position cost calculation."""
        try:
            funding = self.client.futures_funding_rate(symbol="BTCUSDT", limit=1)
            return float(funding[0]["fundingRate"]) if funding else 0.0
        except Exception:
            return -0.000014  # Reasonable fallback

    def _get_current_price(self) -> float:
        """Get current price for optional reference."""
        try:
            ticker = self.client.futures_symbol_ticker(symbol="BTCUSDT")
            return float(ticker["price"])
        except Exception:
            return 115000.0  # Reasonable fallback

    def _use_fallback_specs(self) -> bool:
        """Use fallback specifications when API is unavailable."""
        console.print("[yellow]🔄 Using fallback specifications for backtesting[/yellow]")
        
        self.specs = {
            # ESSENTIAL P&L SPECIFICATIONS
            "maker_fee": "0.00012",     # VIP 3 maker fee
            "taker_fee": "0.00032",     # VIP 3 taker fee
            "funding_rate": -0.000014,  # Typical recent rate
            
            # CONTEXT
            "symbol": "BTCUSDT",
            "current_price": 115000.0,   # Reasonable estimate
            "last_updated": datetime.now().isoformat(),
            
            # DEFAULTS (historical data provides actual constraints)
            "price_precision": 2,
            "quantity_precision": 3,
            "tick_size": "0.01",
            "step_size": "0.001",
            "min_qty": "0.001",
            "min_notional": "100"
        }
        
        self.last_updated = datetime.now()
        console.print("[green]✅ Fallback specifications ready for backtesting[/green]")
        return True

    def create_backtesting_instrument(self):
        """Create NautilusTrader instrument optimized for backtesting."""
        if not self.specs:
            raise ValueError("No specifications available - call fetch_essential_specs() first")
        
        from nautilus_trader.model.currencies import BTC, USDT
        from nautilus_trader.model.identifiers import InstrumentId, Symbol
        from nautilus_trader.model.instruments import CryptoPerpetual
        from nautilus_trader.model.objects import Price, Quantity, Money
        from decimal import Decimal
        
        console.print("[cyan]🔧 Creating backtesting-optimized NautilusTrader instrument...[/cyan]")
        
        # Create instrument with essential specifications only
        instrument = CryptoPerpetual(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.SIM"),
            raw_symbol=Symbol("BTCUSDT"),
            base_currency=BTC,
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            
            # REASONABLE DEFAULTS - historical data provides actual constraints
            price_precision=int(self.specs["price_precision"]),
            size_precision=int(self.specs["quantity_precision"]),
            price_increment=Price.from_str(self.specs["tick_size"]),
            size_increment=Quantity.from_str(self.specs["step_size"]),
            min_quantity=Quantity.from_str(self.specs["min_qty"]),
            max_quantity=Quantity.from_str("1000"),  # Reasonable maximum
            min_notional=Money.from_str(f"{self.specs['min_notional']} USDT"),
            
            # ESSENTIAL - Direct P&L impact
            maker_fee=Decimal(self.specs["maker_fee"]),
            taker_fee=Decimal(self.specs["taker_fee"]),
            
            # MARGIN DEFAULTS
            margin_init=Decimal("0.01"),    # 1% initial margin
            margin_maint=Decimal("0.005"),  # 0.5% maintenance margin
            
            ts_event=0,
            ts_init=0,
        )
        
        console.print(f"[green]✅ Created backtesting instrument: {instrument.id}[/green]")
        console.print(f"[blue]💰 Fee Structure: Maker {float(self.specs['maker_fee'])*100:.3f}% | Taker {float(self.specs['taker_fee'])*100:.3f}%[/blue]")
        
        return instrument

    def get_specs_summary(self) -> Dict[str, Any]:
        """Get streamlined specification summary."""
        if not self.specs:
            return {}
        
        return {
            "symbol": self.specs["symbol"],
            "maker_fee_pct": f"{float(self.specs['maker_fee'])*100:.3f}%",
            "taker_fee_pct": f"{float(self.specs['taker_fee'])*100:.3f}%",
            "funding_rate": self.specs["funding_rate"],
            "current_price": self.specs["current_price"],
            "last_updated": self.last_updated.strftime("%Y-%m-%d %H:%M:%S") if self.last_updated else "Never",
        }

    def display_essential_specs(self):
        """Display only the essential specifications for backtesting."""
        if not self.specs:
            console.print("[red]❌ No specifications available[/red]")
            return
        
        from rich.table import Table
        
        table = Table(title="💰 Essential Backtesting Specifications (P&L Impact Only)")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_column("P&L Impact", style="blue")
        
        # Only show P&L-affecting specifications
        table.add_row("Maker Fee", f"{float(self.specs['maker_fee'])*100:.3f}%", "Direct trading cost")
        table.add_row("Taker Fee", f"{float(self.specs['taker_fee'])*100:.3f}%", "Direct trading cost")
        table.add_row("Funding Rate", f"{self.specs['funding_rate']:.6f}", "Position holding cost")
        table.add_row("Current Price", f"${self.specs['current_price']:,.2f}", "Reference only")
        
        console.print(table)
        
        console.print("\n[yellow]📝 Note: Execution constraints (precision, tick size, etc.) are handled "
                     "naturally by historical OHLCV data and are not fetched for backtesting.[/yellow]")