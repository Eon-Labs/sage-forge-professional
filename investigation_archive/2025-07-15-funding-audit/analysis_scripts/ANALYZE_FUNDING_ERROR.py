#!/usr/bin/env python3
"""
Analyze the exact funding calculation error that's causing 87% deviation.
"""

import json
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table

console = Console()

def analyze_funding_error():
    """Analyze the funding calculation error in detail."""
    
    console.print("[bold cyan]🔍 ANALYZING FUNDING CALCULATION ERROR[/bold cyan]")
    
    # Load the actual funding data being used
    with open('/Users/terryli/eon/nt/nautilus_test/data_cache/funding_integration/BTCUSDT_funding_2025-07-14_2025-07-15.json', 'r') as f:
        data = json.load(f)
    
    console.print(f"[blue]📊 Loaded {len(data)} funding intervals[/blue]")
    
    # Calculate actual system values
    position_size = 0.002  # BTC
    actual_total = 0.0
    
    console.print(f"\n[yellow]💰 ACTUAL SYSTEM CALCULATIONS (Position: {position_size} BTC)[/yellow]")
    
    table = Table(title="Actual Funding Calculations")
    table.add_column("Interval", style="bold")
    table.add_column("Time", style="cyan")
    table.add_column("Rate", justify="right")
    table.add_column("Mark Price", justify="right")
    table.add_column("Cost", justify="right", style="green")
    
    for i, entry in enumerate(data):
        funding_time = datetime.fromtimestamp(entry['fundingTime'] / 1000, tz=timezone.utc)
        rate = float(entry['fundingRate'])
        mark_price = float(entry['markPrice'])
        
        cost = position_size * mark_price * rate
        actual_total += cost
        
        table.add_row(
            str(i+1),
            funding_time.strftime("%m-%d %H:%M UTC"),
            f"{rate:.6f} ({rate*100:.3f}%)",
            f"${mark_price:,.2f}",
            f"${cost:.6f}"
        )
    
    console.print(table)
    console.print(f"[green]💵 ACTUAL TOTAL: ${actual_total:.6f}[/green]")
    
    # Calculate with realistic funding rates
    console.print(f"\n[yellow]🎯 REALISTIC CALCULATIONS (0.05% typical rate)[/yellow]")
    
    realistic_rate = 0.0005  # 0.05% typical BTC funding
    realistic_total = 0.0
    
    realistic_table = Table(title="Realistic Funding Calculations")
    realistic_table.add_column("Interval", style="bold")
    realistic_table.add_column("Mark Price", justify="right")
    realistic_table.add_column("Rate", justify="right")
    realistic_table.add_column("Cost", justify="right", style="red")
    
    for i, entry in enumerate(data):
        mark_price = float(entry['markPrice'])
        cost = position_size * mark_price * realistic_rate
        realistic_total += cost
        
        realistic_table.add_row(
            str(i+1),
            f"${mark_price:,.2f}",
            f"{realistic_rate:.6f} ({realistic_rate*100:.3f}%)",
            f"${cost:.6f}"
        )
    
    console.print(realistic_table)
    console.print(f"[red]💵 REALISTIC TOTAL: ${realistic_total:.6f}[/red]")
    
    # Analysis
    difference = realistic_total - actual_total
    error_pct = (difference / realistic_total) * 100
    
    console.print(f"\n[bold red]🚨 ERROR ANALYSIS[/bold red]")
    console.print(f"Actual: ${actual_total:.6f}")
    console.print(f"Expected: ${realistic_total:.6f}")
    console.print(f"Difference: ${difference:.6f}")
    console.print(f"Error: {error_pct:.1f}% underestimation")
    
    # Root cause
    actual_rate = float(data[0]['fundingRate'])
    rate_error = (realistic_rate - actual_rate) / realistic_rate * 100
    
    console.print(f"\n[bold yellow]🔍 ROOT CAUSE IDENTIFIED[/bold yellow]")
    console.print(f"System uses: {actual_rate:.6f} ({actual_rate*100:.3f}%)")
    console.print(f"Should use: {realistic_rate:.6f} ({realistic_rate*100:.3f}%)")
    console.print(f"Rate error: {rate_error:.1f}% too low")
    
    # Check interval count
    expected_intervals_2_days = 6  # Every 8 hours for 2 days = 6 intervals
    actual_intervals = len(data)
    
    console.print(f"\n[bold blue]⏰ INTERVAL ANALYSIS[/bold blue]")
    console.print(f"Expected intervals (2 days): {expected_intervals_2_days}")
    console.print(f"Actual intervals: {actual_intervals}")
    
    if actual_intervals < expected_intervals_2_days:
        missing_intervals = expected_intervals_2_days - actual_intervals
        console.print(f"[red]❌ Missing {missing_intervals} intervals ({missing_intervals/expected_intervals_2_days*100:.1f}% short)[/red]")
    elif actual_intervals == expected_intervals_2_days:
        console.print(f"[green]✅ Correct number of intervals[/green]")
    
    return {
        'actual_total': actual_total,
        'realistic_total': realistic_total,
        'error_percentage': error_pct,
        'actual_rate': actual_rate,
        'realistic_rate': realistic_rate,
        'intervals': actual_intervals
    }

if __name__ == "__main__":
    result = analyze_funding_error()
    
    console.print(f"\n[bold green]📋 SUMMARY[/bold green]")
    console.print(f"The {result['error_percentage']:.1f}% calculation error is caused by:")
    console.print(f"1. Unrealistically low funding rates: {result['actual_rate']*100:.3f}% vs {result['realistic_rate']*100:.3f}%")
    console.print(f"2. Using {result['intervals']} intervals (may be missing some)")
    console.print(f"3. Total underestimation: ${result['realistic_total'] - result['actual_total']:.6f}")