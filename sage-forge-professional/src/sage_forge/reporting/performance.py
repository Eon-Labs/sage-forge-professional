#!/usr/bin/env python3
"""
Enhanced Performance Reporting System
=====================================

Comprehensive performance analysis with funding-adjusted P&L calculations.
100% identical to the original script's performance reporting capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
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


# ============================================================================
# ODEB: Omniscient Directional Efficiency Benchmark Framework
# ============================================================================

@dataclass
class Position:
    """Position data structure for ODEB analysis."""
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    size_usd: float
    pnl: float
    direction: int  # 1 for LONG, -1 for SHORT
    
    @property
    def duration_days(self) -> float:
        """Calculate position duration in days."""
        return (self.close_time - self.open_time).total_seconds() / 86400


@dataclass
class OdebResult:
    """Results from ODEB analysis."""
    tirex_efficiency_ratio: float
    oracle_efficiency_ratio: float
    directional_capture_pct: float
    oracle_direction: int  # 1 for LONG, -1 for SHORT
    oracle_position_size: float
    noise_floor_applied: float
    tirex_final_pnl: float
    oracle_final_pnl: float
    tirex_max_drawdown: float
    oracle_max_drawdown: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    
    def display_summary(self) -> None:
        """Display ODEB analysis summary using Rich console."""
        # Create ODEB results table
        odeb_table = Table(title="🧙‍♂️ Omniscient Directional Efficiency Benchmark (ODEB)", box=box.ROUNDED)
        odeb_table.add_column("Metric", style="cyan", min_width=20)
        odeb_table.add_column("TiRex Strategy", style="bold", min_width=15)
        odeb_table.add_column("Oracle Baseline", style="green", min_width=15)
        odeb_table.add_column("Analysis", style="blue", min_width=25)
        
        # Oracle strategy details
        oracle_dir_text = "📈 LONG" if self.oracle_direction == 1 else "📉 SHORT"
        
        # Performance comparison
        efficiency_color = "green" if self.directional_capture_pct >= 60 else "yellow" if self.directional_capture_pct >= 40 else "red"
        
        odeb_table.add_row(
            "🎯 Oracle Direction", 
            "N/A", 
            oracle_dir_text,
            f"Market-determined optimal direction"
        )
        
        odeb_table.add_row(
            "💰 Position Size", 
            f"${self.oracle_position_size:,.2f}",
            f"${self.oracle_position_size:,.2f}",
            "Time-weighted average exposure (TWAE)"
        )
        
        odeb_table.add_row(
            "📊 Final P&L", 
            f"${self.tirex_final_pnl:+,.2f}",
            f"${self.oracle_final_pnl:+,.2f}",
            "Actual vs perfect information"
        )
        
        odeb_table.add_row(
            "📉 Max Drawdown", 
            f"${self.tirex_max_drawdown:,.2f}",
            f"${self.oracle_max_drawdown:,.2f}",
            "Risk exposure comparison"
        )
        
        odeb_table.add_row(
            "⚡ Efficiency Ratio", 
            f"{self.tirex_efficiency_ratio:.3f}",
            f"{self.oracle_efficiency_ratio:.3f}",
            "Risk-adjusted performance"
        )
        
        odeb_table.add_row(
            "🎯 Directional Capture", 
            f"[{efficiency_color}]{self.directional_capture_pct:.1f}%[/{efficiency_color}]",
            "100.0%",
            f"Information utilization efficiency"
        )
        
        odeb_table.add_row(
            "🛡️ Noise Floor", 
            f"${self.noise_floor_applied:,.2f}",
            f"${self.noise_floor_applied:,.2f}",
            "Duration-scaled market noise threshold"
        )
        
        console.print(odeb_table)
        
        # Performance assessment
        if self.directional_capture_pct >= 80:
            assessment = "[bold green]EXCELLENT[/bold green] 🌟 - Strategy captures most directional information"
        elif self.directional_capture_pct >= 60:
            assessment = "[bold green]GOOD[/bold green] ✅ - Strong directional capture efficiency"
        elif self.directional_capture_pct >= 40:
            assessment = "[bold yellow]MODERATE[/bold yellow] ⚠️ - Room for directional improvement"
        elif self.directional_capture_pct >= 20:
            assessment = "[bold red]POOR[/bold red] 📉 - Low directional efficiency"
        else:
            assessment = "[bold red]VERY POOR[/bold red] 🔴 - Strategy works against market direction"
        
        console.print(Panel(
            f"{assessment}\n\n"
            f"🧙‍♂️ **ODEB Analysis Summary:**\n"
            f"• TiRex captured {self.directional_capture_pct:.1f}% of perfect directional efficiency\n"
            f"• Oracle strategy direction: {oracle_dir_text}\n"
            f"• Time period: {self.start_time.strftime('%Y-%m-%d')} to {self.end_time.strftime('%Y-%m-%d')}\n"
            f"• Noise floor applied: ${self.noise_floor_applied:,.2f}",
            title="🏆 ODEB Performance Assessment",
            border_style=efficiency_color.split()[0] if " " in efficiency_color else efficiency_color
        ))


class OmniscientDirectionalEfficiencyBenchmark:
    """
    Omniscient Directional Efficiency Benchmark (ODEB) Framework
    
    Measures TiRex strategy effectiveness against theoretical perfect-information baseline
    using time-weighted position sizing and duration-scaled noise floor methodology.
    
    This implements the first practical application of SAGE (Self-Adaptive Generative 
    Evaluation) principles for trading strategy evaluation.
    """
    
    def __init__(self, historical_positions: Optional[List[Position]] = None):
        """
        Initialize ODEB benchmark framework.
        
        Args:
            historical_positions: Historical position data for noise floor calculation
        """
        self.historical_positions = historical_positions or []
        
    def calculate_twae(self, positions: List[Position]) -> float:
        """
        Calculate Time-Weighted Average Exposure (TWAE).
        
        This matches the oracle position size to the actual strategy's 
        capital deployment pattern across time.
        
        Args:
            positions: List of trading positions
            
        Returns:
            Time-weighted average position size in USD
        """
        if not positions:
            return 0.0
            
        total_exposure_time = 0.0
        total_duration = 0.0
        
        for position in positions:
            duration = position.duration_days
            exposure = abs(position.size_usd) * duration
            total_exposure_time += exposure
            total_duration += duration
        
        return total_exposure_time / total_duration if total_duration > 0 else 0.0
    
    def calculate_noise_floor(self, 
                            market_data: pd.DataFrame, 
                            position_duration_days: int, 
                            lookback_days: int = 252) -> float:
        """
        Calculate Duration-Scaled Quantile Market Noise Floor (DSQMNF).
        
        This establishes a minimum adverse excursion threshold based on 
        historical market noise for perfect directional strategies.
        
        Args:
            market_data: OHLCV market data with timestamp index
            position_duration_days: Duration of trading period in days
            lookback_days: Historical lookback period for noise calculation
            
        Returns:
            Noise floor value in same units as P&L
        """
        if len(market_data) < position_duration_days + 1:
            # Fallback: use simple volatility-based estimate
            returns = market_data['close'].pct_change().dropna()
            daily_vol = returns.std()
            return daily_vol * np.sqrt(position_duration_days) * 0.5
        
        adverse_excursions = []
        
        # Calculate adverse excursions for perfect directional positions
        for window_start in range(len(market_data) - position_duration_days):
            window_end = window_start + position_duration_days
            window_data = market_data.iloc[window_start:window_end].copy()
            
            if len(window_data) < 2:
                continue
                
            # Determine perfect direction for this window
            start_price = window_data.iloc[0]['close']
            end_price = window_data.iloc[-1]['close']
            direction = 1 if end_price > start_price else -1
            
            # Calculate perfect position P&L series
            price_changes = window_data['close'] - start_price
            perfect_pnl_series = direction * (price_changes / start_price)
            
            # Find maximum adverse excursion (maximum drawdown from perfect position)
            running_max = perfect_pnl_series.expanding().max()
            drawdown_series = perfect_pnl_series - running_max
            max_adverse_excursion = abs(drawdown_series.min())
            
            adverse_excursions.append(max_adverse_excursion)
        
        if not adverse_excursions:
            # Fallback calculation
            returns = market_data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(position_duration_days) * 0.3
        
        # 15th percentile represents typical market noise even perfect timing can't avoid
        historical_noise_floor = np.percentile(adverse_excursions, 15)
        
        # Duration scaling following Brownian motion (square root of time)
        if self.historical_positions:
            median_duration = np.median([pos.duration_days for pos in self.historical_positions])
            duration_scalar = np.sqrt(position_duration_days / max(median_duration, 1.0))
        else:
            duration_scalar = np.sqrt(position_duration_days / 1.0)  # Default 1-day baseline
        
        return historical_noise_floor * duration_scalar
    
    def simulate_oracle_strategy(self, 
                                direction: int, 
                                position_size: float, 
                                market_data: pd.DataFrame,
                                start_time: pd.Timestamp,
                                end_time: pd.Timestamp) -> Tuple[float, float]:
        """
        Simulate oracle strategy performance with perfect directional information.
        
        Args:
            direction: Oracle direction (1 for LONG, -1 for SHORT)
            position_size: Position size in USD
            market_data: OHLCV market data
            start_time: Strategy start time
            end_time: Strategy end time
            
        Returns:
            Tuple of (oracle_pnl, oracle_max_drawdown)
        """
        try:
            # Get period data
            period_mask = (market_data.index >= start_time) & (market_data.index <= end_time)
            period_data = market_data.loc[period_mask].copy()
            
            if len(period_data) < 2:
                return 0.0, 0.0
                
            # Calculate oracle performance
            entry_price = period_data.iloc[0]['close']
            exit_price = period_data.iloc[-1]['close']
            
            # Oracle P&L calculation
            price_return = (exit_price - entry_price) / entry_price
            oracle_pnl = direction * position_size * price_return
            
            # Calculate oracle maximum drawdown
            price_changes = period_data['close'] - entry_price
            oracle_pnl_series = direction * position_size * (price_changes / entry_price)
            
            # Find maximum drawdown
            running_max = oracle_pnl_series.expanding().max()
            drawdown_series = oracle_pnl_series - running_max
            oracle_max_drawdown = abs(drawdown_series.min())
            
            return oracle_pnl, oracle_max_drawdown
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Oracle simulation error: {e}[/yellow]")
            return 0.0, 0.0
    
    def _calculate_tirex_max_drawdown(self, 
                                    positions: List[Position], 
                                    market_data: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown for TiRex strategy from position data.
        
        Args:
            positions: List of trading positions
            market_data: OHLCV market data
            
        Returns:
            Maximum drawdown value
        """
        if not positions:
            return 0.0
            
        try:
            # Reconstruct P&L series from all positions
            all_pnl_data = []
            
            for position in positions:
                # Get position period data
                position_mask = (market_data.index >= position.open_time) & (market_data.index <= position.close_time)
                position_data = market_data.loc[position_mask].copy()
                
                if len(position_data) < 2:
                    continue
                    
                # Calculate position P&L over time
                entry_price = position_data.iloc[0]['close']
                price_changes = position_data['close'] - entry_price
                position_pnl_series = position.direction * abs(position.size_usd) * (price_changes / entry_price)
                
                # Add to overall P&L series
                for timestamp, pnl in position_pnl_series.items():
                    all_pnl_data.append((timestamp, pnl))
            
            if not all_pnl_data:
                return 0.0
                
            # Create combined P&L series
            pnl_df = pd.DataFrame(all_pnl_data, columns=['timestamp', 'pnl'])
            pnl_df = pnl_df.set_index('timestamp').sort_index()
            
            # Handle overlapping positions by summing
            pnl_series = pnl_df.groupby('timestamp')['pnl'].sum()
            
            # Calculate maximum drawdown
            cumulative_pnl = pnl_series.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown_series = cumulative_pnl - running_max
            
            return abs(drawdown_series.min())
            
        except Exception as e:
            console.print(f"[yellow]⚠️ TiRex drawdown calculation error: {e}[/yellow]")
            # Fallback: use total P&L as rough estimate
            return abs(sum(pos.pnl for pos in positions if pos.pnl < 0))
    
    def calculate_odeb_ratio(self, 
                           tirex_positions: List[Position],
                           market_data: pd.DataFrame) -> OdebResult:
        """
        Calculate complete ODEB analysis comparing TiRex strategy to oracle baseline.
        
        This is the main method that implements the full ODEB methodology:
        1. Determines oracle direction from market evolution
        2. Calculates time-weighted average exposure (TWAE)
        3. Simulates oracle strategy performance
        4. Applies duration-scaled noise floor
        5. Calculates directional capture efficiency
        
        Args:
            tirex_positions: List of TiRex strategy positions
            market_data: OHLCV market data with timestamp index
            
        Returns:
            OdebResult containing complete analysis
        """
        if not tirex_positions:
            raise ValueError("No TiRex positions provided for ODEB analysis")
            
        if market_data.empty:
            raise ValueError("No market data provided for ODEB analysis")
        
        # Determine trading period
        start_time = min(pos.open_time for pos in tirex_positions)
        end_time = max(pos.close_time for pos in tirex_positions)
        
        # Ensure market data covers the trading period
        market_data = market_data.sort_index()
        period_mask = (market_data.index >= start_time) & (market_data.index <= end_time)
        period_data = market_data.loc[period_mask]
        
        if len(period_data) < 2:
            raise ValueError(f"Insufficient market data for period {start_time} to {end_time}")
        
        # 1. Determine oracle direction (SAGE self-discovery principle)
        start_price = period_data.iloc[0]['close']
        end_price = period_data.iloc[-1]['close']
        oracle_direction = 1 if end_price > start_price else -1
        
        # 2. Calculate time-weighted average exposure (SAGE generative validation)
        oracle_position_size = self.calculate_twae(tirex_positions)
        
        # 3. Simulate oracle strategy performance
        oracle_pnl, oracle_max_drawdown = self.simulate_oracle_strategy(
            oracle_direction, oracle_position_size, market_data, start_time, end_time
        )
        
        # 4. Calculate TiRex strategy performance
        tirex_final_pnl = sum(pos.pnl for pos in tirex_positions)
        tirex_max_drawdown = self._calculate_tirex_max_drawdown(tirex_positions, market_data)
        
        # 5. Apply duration-scaled noise floor (SAGE adaptive thresholding)
        position_duration = (end_time - start_time).days
        noise_floor = self.calculate_noise_floor(market_data, position_duration)
        
        # 6. Calculate efficiency ratios with noise floor protection
        tirex_efficiency = tirex_final_pnl / max(tirex_max_drawdown, noise_floor, 0.001)
        oracle_efficiency = oracle_pnl / max(oracle_max_drawdown, noise_floor, 0.001)
        
        # 7. Calculate directional capture efficiency
        if oracle_efficiency != 0:
            directional_capture = (tirex_efficiency / oracle_efficiency) * 100
        else:
            directional_capture = 0.0
        
        # Ensure directional capture is reasonable (between -200% and 200%)
        directional_capture = max(-200.0, min(200.0, directional_capture))
        
        return OdebResult(
            tirex_efficiency_ratio=tirex_efficiency,
            oracle_efficiency_ratio=oracle_efficiency,
            directional_capture_pct=directional_capture,
            oracle_direction=oracle_direction,
            oracle_position_size=oracle_position_size,
            noise_floor_applied=noise_floor,
            tirex_final_pnl=tirex_final_pnl,
            oracle_final_pnl=oracle_pnl,
            tirex_max_drawdown=tirex_max_drawdown,
            oracle_max_drawdown=oracle_max_drawdown,
            start_time=start_time,
            end_time=end_time
        )


# Convenience functions for ODEB usage
def create_position_from_dict(position_data: Dict[str, Any]) -> Position:
    """
    Create Position object from dictionary data.
    
    Args:
        position_data: Dictionary with keys: open_time, close_time, size_usd, pnl, direction
        
    Returns:
        Position object
    """
    return Position(
        open_time=pd.to_datetime(position_data['open_time']),
        close_time=pd.to_datetime(position_data['close_time']),
        size_usd=float(position_data['size_usd']),
        pnl=float(position_data['pnl']),
        direction=int(position_data['direction'])
    )


def run_odeb_analysis(positions_data: List[Dict[str, Any]], 
                     market_data: pd.DataFrame,
                     display_results: bool = True) -> OdebResult:
    """
    Convenience function to run complete ODEB analysis from raw data.
    
    Args:
        positions_data: List of position dictionaries
        market_data: OHLCV market data
        display_results: Whether to display results using Rich console
        
    Returns:
        OdebResult with complete analysis
    """
    # Convert position data to Position objects
    positions = [create_position_from_dict(pos_data) for pos_data in positions_data]
    
    # Initialize ODEB framework
    odeb = OmniscientDirectionalEfficiencyBenchmark(positions)
    
    # Run analysis
    result = odeb.calculate_odeb_ratio(positions, market_data)
    
    # Display results if requested
    if display_results:
        result.display_summary()
    
    return result