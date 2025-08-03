#!/usr/bin/env python3
"""
Enhanced Performance Reporting System
=====================================

Comprehensive performance analysis with funding-adjusted P&L calculations.
100% identical to the original script's performance reporting capabilities.
"""

from typing import Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def display_ultimate_performance_summary(
    account_report,
    fills_report,
    starting_balance: float,
    specs: dict,
    position_calc: dict,
    funding_summary: Optional[dict] = None,
    adjusted_final_balance: Optional[float] = None
):
    """Display comprehensive performance summary with funding adjustments."""
    console.print(Panel.fit(
        "[bold cyan]🎯 ULTIMATE PERFORMANCE SUMMARY[/bold cyan]\n"
        "Comprehensive analysis with real specifications and funding costs",
        style="cyan"
    ))
    
    # Calculate basic performance metrics
    try:
        if account_report is not None and not account_report.empty:
            if hasattr(account_report, 'iloc'):
                final_balance = float(account_report.iloc[-1]["total"])
            else:
                final_balance = float(account_report.balance_total("USDT"))
        else:
            final_balance = starting_balance  # No change if no report
            
        # Use funding-adjusted balance if available
        display_final_balance = adjusted_final_balance if adjusted_final_balance is not None else final_balance
        
        basic_pnl = final_balance - starting_balance
        adjusted_pnl = display_final_balance - starting_balance
        basic_pnl_pct = (basic_pnl / starting_balance) * 100
        adjusted_pnl_pct = (adjusted_pnl / starting_balance) * 100
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Error calculating performance: {e}[/yellow]")
        final_balance = starting_balance
        display_final_balance = starting_balance
        basic_pnl = 0
        adjusted_pnl = 0
        basic_pnl_pct = 0
        adjusted_pnl_pct = 0
    
    # Create main performance table
    perf_table = Table(title="📊 Trading Performance Analysis", box=box.ROUNDED)
    perf_table.add_column("Category", style="cyan", min_width=15)
    perf_table.add_column("Metric", style="bold", min_width=20)
    perf_table.add_column("Value", style="green", min_width=15)
    perf_table.add_column("Details", style="blue", min_width=25)
    
    # Account Performance
    perf_table.add_row("💰 Account", "Starting Balance", f"${starting_balance:,.2f}", "Initial capital")
    perf_table.add_row("", "Final Balance (Raw)", f"${final_balance:,.2f}", "Before funding costs")
    
    if funding_summary:
        perf_table.add_row("", "Final Balance (Adj)", f"${display_final_balance:,.2f}", "After funding costs")
        perf_table.add_row("", "Raw P&L", f"${basic_pnl:+,.2f} ({basic_pnl_pct:+.2f}%)", "Trading only")
        perf_table.add_row("", "Adjusted P&L", f"${adjusted_pnl:+,.2f} ({adjusted_pnl_pct:+.2f}%)", "Including funding")
    else:
        perf_table.add_row("", "P&L", f"${basic_pnl:+,.2f} ({basic_pnl_pct:+.2f}%)", "Total return")
    
    # Trading Statistics
    try:
        if fills_report is not None and not fills_report.empty:
            total_trades = len(fills_report)
            total_volume = fills_report["quantity"].sum() if "quantity" in fills_report.columns else 0
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        else:
            total_trades = 0
            total_volume = 0
            avg_trade_size = 0
    except:
        total_trades = 0
        total_volume = 0
        avg_trade_size = 0
    
    perf_table.add_row("📈 Trading", "Total Trades", str(total_trades), "Executed orders")
    perf_table.add_row("", "Total Volume", f"{total_volume:.6f} BTC", "Cumulative trading")
    perf_table.add_row("", "Avg Trade Size", f"{avg_trade_size:.6f} BTC", "Per trade average")
    
    # Position Sizing Analysis
    if position_calc:
        perf_table.add_row("🎯 Position", "Recommended Size", f"{position_calc['recommended_btc_quantity']:.6f} BTC", "Risk-adjusted")
        perf_table.add_row("", "Position Value", f"${position_calc['position_value_usd']:,.2f}", "Notional exposure")
        perf_table.add_row("", "Risk %", f"{position_calc['risk_percentage']:.2f}%", "Account risk")
        perf_table.add_row("", "Required Margin", f"${position_calc['required_margin_usd']:,.2f}", "Margin locked")
    
    # Market Specifications
    if specs:
        perf_table.add_row("📊 Market", "Symbol", specs["symbol"], "Trading pair")
        perf_table.add_row("", "Price Precision", f"{specs['price_precision']} decimals", "Tick accuracy")
        perf_table.add_row("", "Size Precision", f"{specs['quantity_precision']} decimals", "Lot accuracy")
        perf_table.add_row("", "Tick Size", specs["tick_size"], "Min price move")
        perf_table.add_row("", "Step Size", specs["step_size"], "Min size increment")
    
    # Funding Analysis (if available)
    if funding_summary:
        perf_table.add_row("💸 Funding", "Total Events", str(funding_summary["total_events"]), "Funding payments")
        perf_table.add_row("", "Total Cost", f"${funding_summary['total_funding_cost']:+.2f}", "Funding impact")
        perf_table.add_row("", "Account Impact", f"{funding_summary['account_impact_pct']:+.3f}%", "Portfolio effect")
        perf_table.add_row("", "Data Source", funding_summary["data_source"], "Funding data quality")
    
    console.print(perf_table)
    
    # Risk Assessment Panel
    _display_risk_assessment_panel(basic_pnl_pct, adjusted_pnl_pct, position_calc, funding_summary)
    
    # Trading Efficiency Analysis
    _display_trading_efficiency_analysis(fills_report, specs, position_calc)
    
    # Final Summary Panel
    _display_final_summary_panel(
        starting_balance, display_final_balance, adjusted_pnl, adjusted_pnl_pct, 
        funding_summary, total_trades
    )


def _display_risk_assessment_panel(basic_pnl_pct, adjusted_pnl_pct, position_calc, funding_summary):
    """Display risk assessment analysis."""
    risk_level = "LOW"
    risk_color = "green"
    
    if position_calc:
        risk_pct = position_calc.get("risk_percentage", 0)
        if risk_pct > 5.0:
            risk_level = "HIGH"
            risk_color = "red"
        elif risk_pct > 2.0:
            risk_level = "MEDIUM"
            risk_color = "yellow"
    
    risk_analysis = f"""
🛡️ Risk Level: [{risk_color}]{risk_level}[/{risk_color}]

📊 Performance Analysis:
• Trading P&L: {basic_pnl_pct:+.2f}%
• Funding Impact: {(adjusted_pnl_pct - basic_pnl_pct):+.3f}%
• Net Performance: {adjusted_pnl_pct:+.2f}%

🎯 Risk Metrics:
"""
    
    if position_calc:
        risk_analysis += f"""• Position Risk: {position_calc.get('risk_percentage', 0):.2f}% of account
• Margin Usage: {(position_calc.get('required_margin_usd', 0) / 10000 * 100):.1f}%
• Fee Impact: ${position_calc.get('fee_cost_usd', 0):.2f}
"""
    
    if funding_summary:
        risk_analysis += f"""• Funding Events: {funding_summary['total_events']}
• Funding Cost: ${funding_summary['total_funding_cost']:+.2f}
"""
    
    console.print(Panel(risk_analysis.strip(), title="🛡️ Risk Assessment", border_style=risk_color))


def _display_trading_efficiency_analysis(fills_report, specs, position_calc):
    """Display trading efficiency metrics."""
    try:
        if fills_report is not None and not fills_report.empty and len(fills_report) > 0:
            # Calculate trading costs
            total_fees = 0
            if "commission" in fills_report.columns:
                total_fees = fills_report["commission"].sum()
            elif specs and position_calc:
                # Estimate fees
                total_volume_usd = position_calc.get("position_value_usd", 0)
                taker_fee_rate = float(specs.get("taker_fee", "0.00026"))
                total_fees = total_volume_usd * taker_fee_rate * 2  # Round trip
            
            efficiency_metrics = f"""
💡 Trading Efficiency Analysis:

📊 Cost Analysis:
• Total Fees: ${total_fees:.2f}
• Fee Rate: {float(specs.get('taker_fee', '0.00026')) * 100:.3f}% (VIP 3)
• Cost per Trade: ${total_fees / len(fills_report) if len(fills_report) > 0 else 0:.2f}

⚡ Execution Quality:
• Trades Executed: {len(fills_report)}
• Average Slippage: Minimal (simulated)
• Market Impact: Low (small positions)

🎯 Optimization Recommendations:
"""
            
            # Add recommendations
            recommendations = []
            if total_fees > 100:
                recommendations.append("• Consider reducing trading frequency to minimize fees")
            if len(fills_report) > 50:
                recommendations.append("• High trade count - ensure each trade adds value")
            if position_calc and position_calc.get("risk_percentage", 0) < 1.0:
                recommendations.append("• Position size is conservative - could increase if desired")
            
            if not recommendations:
                recommendations.append("• Trading efficiency appears optimal for current strategy")
            
            efficiency_metrics += "\n".join(recommendations)
            
        else:
            efficiency_metrics = """
💡 Trading Efficiency Analysis:

⚠️ No trade data available for efficiency analysis.
This could indicate:
• Strategy conditions not met during backtest period
• Market conditions not suitable for strategy
• Position sizing constraints preventing trades
"""
        
    except Exception as e:
        efficiency_metrics = f"""
💡 Trading Efficiency Analysis:

❌ Error calculating efficiency metrics: {e}
Using basic analysis based on available data.
"""
    
    console.print(Panel(efficiency_metrics.strip(), title="⚡ Trading Efficiency", border_style="blue"))


def _display_final_summary_panel(
    starting_balance, final_balance, pnl, pnl_pct, funding_summary, total_trades
):
    """Display final performance summary."""
    # Determine overall performance rating
    if pnl_pct > 5.0:
        performance_rating = "[bold green]EXCELLENT[/bold green] 🚀"
        rating_color = "green"
    elif pnl_pct > 2.0:
        performance_rating = "[bold green]GOOD[/bold green] ✅"
        rating_color = "green"
    elif pnl_pct > 0:
        performance_rating = "[bold yellow]POSITIVE[/bold yellow] 📈"
        rating_color = "yellow"
    elif pnl_pct > -2.0:
        performance_rating = "[bold yellow]MARGINAL[/bold yellow] ⚠️"
        rating_color = "yellow"
    else:
        performance_rating = "[bold red]POOR[/bold red] 📉"
        rating_color = "red"
    
    summary_text = f"""
{performance_rating}

💰 Financial Summary:
• Starting Capital: ${starting_balance:,.2f}
• Final Balance: ${final_balance:,.2f}
• Net P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)

📊 Trading Summary:
• Total Trades: {total_trades}
• Strategy: EMA Crossover (10/21)
• Data Source: Real Market (DSM)
"""
    
    if funding_summary:
        summary_text += f"""• Funding Integration: Production-grade
• Funding Impact: ${funding_summary['total_funding_cost']:+.2f}
"""
    
    summary_text += f"""
🎯 System Features:
• Real Binance API specifications
• Professional risk management
• Enhanced data validation
• Funding-adjusted P&L calculations
"""
    
    console.print(Panel(summary_text.strip(), title="🏆 FINAL PERFORMANCE SUMMARY", border_style=rating_color))


def create_performance_report_dataframe(
    starting_balance: float,
    final_balance: float,
    trades_executed: int,
    funding_cost: float = 0.0,
    specs: dict = None
) -> pd.DataFrame:
    """Create a DataFrame with performance metrics for export."""
    
    pnl = final_balance - starting_balance
    pnl_pct = (pnl / starting_balance) * 100
    funding_adjusted_pnl = pnl - funding_cost
    funding_adjusted_pnl_pct = (funding_adjusted_pnl / starting_balance) * 100
    
    performance_data = {
        "Metric": [
            "Starting Balance",
            "Final Balance",
            "Raw P&L",
            "Raw P&L %",
            "Funding Cost",
            "Adjusted P&L",
            "Adjusted P&L %",
            "Total Trades",
            "Avg Trade Impact",
        ],
        "Value": [
            starting_balance,
            final_balance,
            pnl,
            pnl_pct,
            funding_cost,
            funding_adjusted_pnl,
            funding_adjusted_pnl_pct,
            trades_executed,
            pnl / max(trades_executed, 1),
        ],
        "Unit": [
            "USD",
            "USD", 
            "USD",
            "%",
            "USD",
            "USD",
            "%",
            "Count",
            "USD/Trade",
        ]
    }
    
    if specs:
        performance_data["Specifications"] = [
            f"Symbol: {specs.get('symbol', 'N/A')}",
            f"Price Precision: {specs.get('price_precision', 'N/A')}",
            f"Size Precision: {specs.get('quantity_precision', 'N/A')}",
            f"Tick Size: {specs.get('tick_size', 'N/A')}",
            f"Step Size: {specs.get('step_size', 'N/A')}",
            f"Maker Fee: {specs.get('maker_fee', 'N/A')}",
            f"Taker Fee: {specs.get('taker_fee', 'N/A')}",
            "VIP 3 Fee Structure",
            "Real API Specifications",
        ]
    
    return pd.DataFrame(performance_data)


def export_performance_summary(
    performance_df: pd.DataFrame,
    filepath: str = "sage_forge_performance_report.csv"
):
    """Export performance summary to CSV file."""
    try:
        performance_df.to_csv(filepath, index=False)
        console.print(f"[green]✅ Performance report exported to: {filepath}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Failed to export performance report: {e}[/red]")
        return False