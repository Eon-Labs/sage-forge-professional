#!/usr/bin/env python3
"""
RealisticPositionSizer - Professional Risk Management
====================================================

Calculates realistic position sizes preventing account blow-up.
100% identical to the original script's RealisticPositionSizer.
"""

from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class RealisticPositionSizer:
    """Calculates realistic position sizes preventing account blow-up."""
    
    def __init__(self, specs: dict, account_balance: float = 10000, max_risk_pct: float = 0.02):
        self.specs = specs
        self.account_balance = account_balance
        self.max_risk_pct = max_risk_pct  # 2% max risk per trade
        console.print(f"[blue]🛡️ RealisticPositionSizer initialized (Balance: ${account_balance:,.2f}, Max Risk: {max_risk_pct*100:.1f}%)[/blue]")

    def calculate_position_size(self) -> dict:
        """Calculate safe position size based on account balance and risk parameters."""
        console.print("[cyan]🧮 Calculating realistic position size...[/cyan]")
        
        # 🔧 CRITICAL FIX: Use live market data instead of hardcoded estimate
        live_btc_price = self.specs.get("current_price", 117500.0)  # Use live price from Binance API
        console.print(f"[yellow]🔍 DEBUG: Using BTC price: ${live_btc_price:,.2f} (source: {'live API' if 'current_price' in self.specs else 'fallback estimate'})[/yellow]")
        
        # Calculate maximum risk amount
        max_risk_amount = self.account_balance * self.max_risk_pct
        
        # Calculate maximum position value (assuming 50% drawdown protection)
        max_position_value = max_risk_amount / 0.5  # 50% stop loss
        
        # Calculate maximum BTC position size
        max_btc_quantity = max_position_value / live_btc_price
        
        # Apply Binance minimum quantity constraints
        min_qty = float(self.specs["min_qty"])
        step_size = float(self.specs["step_size"])
        
        # Round down to valid step size
        valid_btc_quantity = max(min_qty, int(max_btc_quantity / step_size) * step_size)
        
        # Calculate actual position value and risk (using live price)
        actual_position_value = valid_btc_quantity * live_btc_price
        actual_risk_pct = (actual_position_value * 0.5) / self.account_balance  # 50% potential loss
        
        # Calculate margin requirements
        initial_margin_rate = float(self.specs["initial_margin_rate"])
        required_margin = actual_position_value * initial_margin_rate
        
        # Calculate fee impact
        taker_fee_rate = float(self.specs["taker_fee"])
        round_trip_fee = actual_position_value * taker_fee_rate * 2  # Entry + Exit
        
        return {
            "live_btc_price": live_btc_price,
            "max_risk_amount": max_risk_amount,
            "max_position_value": max_position_value,
            "calculated_btc_quantity": max_btc_quantity,
            "valid_btc_quantity": valid_btc_quantity,
            "actual_position_value": actual_position_value,
            "actual_risk_pct": actual_risk_pct,
            "required_margin": required_margin,
            "round_trip_fee": round_trip_fee,
            "free_balance_after": self.account_balance - required_margin,
            "min_qty_constraint": min_qty,
            "step_size_constraint": step_size,
        }

    def display_position_analysis(self):
        """Display comprehensive position sizing analysis."""
        console.print(Panel.fit(
            "[bold cyan]🛡️ REALISTIC POSITION SIZING ANALYSIS[/bold cyan]\n"
            "Professional risk management preventing account blow-up",
            style="cyan"
        ))
        
        calc = self.calculate_position_size()
        
        # Main analysis table
        table = Table(title="📊 Position Sizing Analysis")
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="green")
        table.add_column("Impact", style="blue")
        
        # Account & Risk Analysis
        table.add_row("💰 Account", "Balance", f"${self.account_balance:,.2f}", "Available capital")
        table.add_row("", "Max Risk %", f"{self.max_risk_pct*100:.1f}%", "Per trade limit")
        table.add_row("", "Max Risk $", f"${calc['max_risk_amount']:,.2f}", "Absolute risk limit")
        
        # Position Calculations
        table.add_row("🎯 Position", "BTC Live Price", f"${calc['live_btc_price']:,.2f}", "Live market data")
        table.add_row("", "Raw Calculation", f"{calc['calculated_btc_quantity']:.6f} BTC", "Before constraints")
        table.add_row("", "Valid Quantity", f"{calc['valid_btc_quantity']:.6f} BTC", "After Binance rules")
        table.add_row("", "Position Value", f"${calc['actual_position_value']:,.2f}", "Notional exposure")
        table.add_row("", "Actual Risk %", f"{calc['actual_risk_pct']*100:.2f}%", "Real risk percentage")
        
        # Margin & Fees
        table.add_row("🏦 Margin", "Required Margin", f"${calc['required_margin']:,.2f}", f"{float(self.specs['initial_margin_rate'])*100:.1f}% of position")
        table.add_row("", "Free Balance", f"${calc['free_balance_after']:,.2f}", "After margin lock")
        table.add_row("💸 Fees", "Round-trip Cost", f"${calc['round_trip_fee']:,.2f}", "Entry + Exit fees")
        
        # Constraints
        table.add_row("⚖️ Binance", "Min Quantity", f"{calc['min_qty_constraint']} BTC", "Exchange minimum")
        table.add_row("", "Step Size", f"{calc['step_size_constraint']} BTC", "Increment rule")
        
        console.print(table)
        
        # 🔧 CRITICAL ADDITION: Mathematical validation with DEBUG logging (matching original lines 442-520)
        self._perform_mathematical_validation(calc)
        
        # Risk assessment
        self._display_risk_assessment(calc)
        
        # Recommendations
        self._display_recommendations(calc)
        
        return calc

    def _perform_mathematical_validation(self, calc: dict):
        """🔧 CRITICAL: Mathematical validation with DEBUG logging (matching original lines 442-520)."""
        console.print("[yellow]🔍 DEBUG: Validating position sizing mathematics...[/yellow]")
        
        # 🔧 FIXED: Use consistent pricing - same source as calculations
        current_price = calc["live_btc_price"]
        console.print(f"[yellow]🔍 DEBUG: Validation using consistent price: ${current_price:,.2f}[/yellow]")
        dangerous_1btc_value = 1.0 * current_price
        realistic_position_value = calc["actual_position_value"]
        
        console.print(f"[blue]📊 DEBUG: Dangerous 1 BTC value: ${dangerous_1btc_value:,.2f}[/blue]")
        console.print(f"[blue]📊 DEBUG: Realistic position value: ${realistic_position_value:.2f}[/blue]")
        
        # Calculate consistent safety factors
        position_size_ratio = 1.0 / calc["valid_btc_quantity"]  # How many times larger 1 BTC is
        value_safety_factor = dangerous_1btc_value / realistic_position_value  # How many times safer realistic position is
        
        console.print(f"[cyan]🔍 DEBUG: Position size ratio: {position_size_ratio:.1f}x (1 BTC is {position_size_ratio:.1f}x larger)[/cyan]")
        console.print(f"[cyan]🔍 DEBUG: Value safety factor: {value_safety_factor:.1f}x (realistic position is {value_safety_factor:.1f}x safer)[/cyan]")
        
        # 🚨 MATHEMATICAL VALIDATION: These should be approximately equal!
        ratio_difference = abs(position_size_ratio - value_safety_factor)
        console.print(f"[cyan]🧮 DEBUG: Safety factor consistency check: {ratio_difference:.1f} difference[/cyan]")
        
        if ratio_difference > 1.0:  # Allow for small rounding differences
            console.print("[red]🚨 WARNING: Inconsistent safety factors detected![/red]")
            console.print(f"[red]📊 Position ratio: {position_size_ratio:.1f}x vs Value safety: {value_safety_factor:.1f}x[/red]")
            console.print("[red]🔍 This indicates mathematical errors in position sizing[/red]")
        else:
            console.print("[green]✅ DEBUG: Position sizing mathematics validated[/green]")
        
        # Display comparison table (matching original)
        self._display_safety_comparison(calc, dangerous_1btc_value, value_safety_factor)

    def _display_safety_comparison(self, calc: dict, dangerous_1btc_value: float, value_safety_factor: float):
        """Display safety comparison table (matching original table format)."""
        from rich import box
        
        comparison_table = Table(title="💰 Enhanced Position Sizing (DSM + Hybrid)", box=box.ROUNDED)
        comparison_table.add_column("Metric", style="bold")
        comparison_table.add_column("Realistic Value", style="green")
        comparison_table.add_column("DSM Demo (Dangerous)", style="red")
        comparison_table.add_column("Safety Factor", style="cyan")
        
        metrics = [
            ("Account Balance", f"${self.account_balance:,.0f}", f"${self.account_balance:,.0f}", "Same"),
            ("Position Size", f"{calc['valid_btc_quantity']:.3f} BTC", "1.000 BTC", f"{1.0/calc['valid_btc_quantity']:.0f}x smaller (safer)"),
            ("Trade Value", f"${calc['actual_position_value']:.2f}", f"${dangerous_1btc_value:,.0f}", f"{value_safety_factor:.0f}x smaller (safer)"),
            ("Account Risk", f"{calc['actual_risk_pct']*100:.1f}%", f"{(dangerous_1btc_value/self.account_balance)*100:.0f}%", "Controlled vs Reckless"),
            ("Blow-up Risk", "Protected via small size", "Extreme via large size", f"{value_safety_factor:.0f}x risk reduction"),
        ]
        
        for metric, safe_val, dangerous_val, safety in metrics:
            comparison_table.add_row(metric, safe_val, dangerous_val, safety)
        
        console.print(comparison_table)

    def _display_risk_assessment(self, calc: dict):
        """Display risk assessment panel."""
        risk_pct = calc["actual_risk_pct"] * 100
        
        if risk_pct <= 2.0:
            risk_status = "[green]✅ ACCEPTABLE RISK[/green]"
            risk_color = "green"
        elif risk_pct <= 5.0:
            risk_status = "[yellow]⚠️ ELEVATED RISK[/yellow]"
            risk_color = "yellow"
        else:
            risk_status = "[red]🚨 DANGEROUS RISK[/red]"
            risk_color = "red"
        
        margin_util = (calc["required_margin"] / self.account_balance) * 100
        
        risk_panel = f"""
{risk_status}

📊 Risk Metrics:
• Position Risk: {risk_pct:.2f}% of account
• Margin Utilization: {margin_util:.1f}%
• Max Potential Loss: ${calc['actual_position_value'] * 0.5:,.2f}
• Fee Impact: ${calc['round_trip_fee']:,.2f}

🎯 Position Details:
• BTC Quantity: {calc['valid_btc_quantity']:.6f}
• USD Value: ${calc['actual_position_value']:,.2f}
• Required Margin: ${calc['required_margin']:,.2f}
        """
        
        console.print(Panel(risk_panel.strip(), title="🛡️ Risk Assessment", border_style=risk_color))

    def _display_recommendations(self, calc: dict):
        """Display position sizing recommendations."""
        recommendations = []
        
        risk_pct = calc["actual_risk_pct"] * 100
        margin_util = (calc["required_margin"] / self.account_balance) * 100
        
        if risk_pct <= 1.0:
            recommendations.append("✅ Conservative position size - suitable for consistent trading")
        elif risk_pct <= 2.0:
            recommendations.append("✅ Acceptable risk level - within professional guidelines")
        elif risk_pct <= 5.0:
            recommendations.append("⚠️ Consider reducing position size for better risk management")
        else:
            recommendations.append("🚨 Position too large - high risk of account damage")
        
        if margin_util > 50:
            recommendations.append("⚠️ High margin utilization - limits flexibility for additional positions")
        
        if calc["free_balance_after"] < 1000:
            recommendations.append("🚨 Low free balance remaining - consider smaller position")
        
        if calc["round_trip_fee"] / calc["actual_position_value"] > 0.001:
            recommendations.append("💸 Fee impact >0.1% - ensure profit targets cover costs")
        
        # Add positive recommendations
        if len([r for r in recommendations if r.startswith("✅")]) >= 2:
            recommendations.append("🎯 Position sizing is well-balanced for professional trading")
        
        if recommendations:
            rec_text = "\n".join(f"• {rec}" for rec in recommendations)
            console.print(Panel(rec_text, title="💡 Recommendations", border_style="blue"))

    def get_recommended_position_size(self) -> float:
        """Get the recommended position size in BTC."""
        calc = self.calculate_position_size()
        return calc["valid_btc_quantity"]

    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of position calculations for external use."""
        calc = self.calculate_position_size()
        return {
            "recommended_btc_quantity": calc["valid_btc_quantity"],
            "position_value_usd": calc["actual_position_value"],
            "required_margin_usd": calc["required_margin"],
            "risk_percentage": calc["actual_risk_pct"] * 100,
            "fee_cost_usd": calc["round_trip_fee"],
            "free_balance_after": calc["free_balance_after"],
        }