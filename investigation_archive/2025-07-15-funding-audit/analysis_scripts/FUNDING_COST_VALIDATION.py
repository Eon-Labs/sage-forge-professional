#!/usr/bin/env python3
"""
🔍 FUNDING COST MATHEMATICAL VALIDATION

This script validates whether the $0.09 funding cost for 0.002 BTC over 2 days
is mathematically correct or if there are calculation errors.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def validate_funding_mathematics():
    """Validate funding cost calculations against realistic parameters."""
    
    console.print(Panel.fit(
        "[bold cyan]🔍 FUNDING COST MATHEMATICAL VALIDATION[/bold cyan]\n"
        "Analyzing whether $0.09 funding cost is realistic",
        title="MATHEMATICAL VERIFICATION"
    ))
    
    # System reported values
    reported_funding_cost = 0.09
    position_size_btc = 0.002
    btc_price_approx = 117500  # Approximate from recent data
    time_period_days = 2
    
    console.print(f"\n[bold yellow]📊 SYSTEM REPORTED VALUES[/bold yellow]")
    console.print(f"Position Size: {position_size_btc} BTC")
    console.print(f"BTC Price: ~${btc_price_approx:,}")
    console.print(f"Time Period: {time_period_days} days")
    console.print(f"Reported Funding Cost: ${reported_funding_cost}")
    
    # Calculate realistic expectations
    console.print(f"\n[bold blue]🧮 REALISTIC FUNDING RATE SCENARIOS[/bold blue]")
    
    scenarios = [
        ("Very Low", 0.0001, "0.01%"),
        ("Typical", 0.0005, "0.05%"), 
        ("High", 0.001, "0.10%"),
        ("Extreme", 0.002, "0.20%"),
    ]
    
    results_table = Table(title="Funding Cost Validation Analysis")
    results_table.add_column("Scenario", style="bold")
    results_table.add_column("Rate", justify="center")
    results_table.add_column("Per Interval", justify="right")
    results_table.add_column("Per Day (3x)", justify="right") 
    results_table.add_column("2-Day Total", justify="right")
    results_table.add_column("vs Reported", style="cyan")
    
    notional_value = position_size_btc * btc_price_approx
    intervals_per_day = 3  # 8-hour intervals
    total_intervals = time_period_days * intervals_per_day
    
    console.print(f"Notional Value: {position_size_btc} × ${btc_price_approx:,} = ${notional_value:,.2f}")
    console.print(f"Total Intervals: {time_period_days} days × {intervals_per_day} intervals = {total_intervals} intervals")
    
    for scenario_name, rate, rate_pct in scenarios:
        cost_per_interval = notional_value * rate
        cost_per_day = cost_per_interval * intervals_per_day
        total_cost = cost_per_day * time_period_days
        
        vs_reported = total_cost / reported_funding_cost
        vs_text = f"{vs_reported:.1f}x {'higher' if vs_reported > 1 else 'lower'}"
        
        results_table.add_row(
            scenario_name,
            f"{rate:.4f} ({rate_pct})",
            f"${cost_per_interval:.4f}",
            f"${cost_per_day:.3f}",
            f"${total_cost:.3f}",
            vs_text
        )
    
    console.print(results_table)
    
    # Analysis
    console.print(f"\n[bold red]🚨 MATHEMATICAL ANALYSIS[/bold red]")
    
    # Calculate what rate would give $0.09
    implied_rate = reported_funding_cost / (notional_value * total_intervals)
    implied_rate_pct = implied_rate * 100
    
    console.print(f"For ${reported_funding_cost} to be correct, funding rate would need to be:")
    console.print(f"Rate: {implied_rate:.6f} ({implied_rate_pct:.4f}%)")
    
    if implied_rate < 0.00001:  # Less than 0.001%
        console.print(f"[red]❌ UNREALISTIC: {implied_rate_pct:.4f}% is extremely low for BTC perpetual futures[/red]")
        console.print(f"[red]📊 Typical BTC funding rates range from 0.01% to 0.10%[/red]")
        console.print(f"[red]💥 This suggests a calculation error in the funding system[/red]")
    elif implied_rate > 0.002:  # More than 0.2%
        console.print(f"[red]❌ UNREALISTIC: {implied_rate_pct:.4f}% is extremely high[/red]")
    else:
        console.print(f"[green]✅ REALISTIC: {implied_rate_pct:.4f}% is within reasonable range[/green]")
    
    # Final verdict
    console.print(f"\n[bold cyan]🎯 FINAL VERDICT[/bold cyan]")
    
    typical_expected = notional_value * 0.0005 * total_intervals  # 0.05% rate
    difference = abs(typical_expected - reported_funding_cost)
    difference_pct = (difference / typical_expected) * 100
    
    console.print(f"Expected (typical 0.05% rate): ${typical_expected:.3f}")
    console.print(f"Reported: ${reported_funding_cost:.3f}")
    console.print(f"Difference: ${difference:.3f} ({difference_pct:.1f}%)")
    
    if difference_pct > 50:
        console.print(f"[red]❌ FUNDING COST CALCULATION IS LIKELY INCORRECT[/red]")
        console.print(f"[red]📊 {difference_pct:.1f}% deviation from expected values[/red]")
        return False
    else:
        console.print(f"[green]✅ Funding cost calculation appears reasonable[/green]")
        return True

if __name__ == "__main__":
    is_valid = validate_funding_mathematics()
    
    if is_valid:
        console.print("\n[bold green]🎉 Funding cost validation PASSED[/bold green]")
    else:
        console.print("\n[bold red]🚨 Funding cost validation FAILED[/bold red]")
        console.print("[yellow]⚠️ This indicates a calculation error in the funding system[/yellow]")