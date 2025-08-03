#!/usr/bin/env python3
"""
🎯 NaN Location Finder - Pinpoint exact corrupt rows

Identifies the specific row indices and values causing data quality issues.
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add SAGE-Forge to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sage_forge.data.manager import ArrowDataManager

console = Console()


def find_nan_locations():
    """Find exact locations of NaN values in raw data."""
    console.print(Panel.fit("🎯 Finding exact NaN locations", style="bold blue"))
    
    # Fetch raw data
    data_manager = ArrowDataManager()
    
    try:
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
        
        raw_data = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            timeframe="1m", 
            limit=500,
            start_time=start_time,
            end_time=end_time
        )
        
        if raw_data is None:
            console.print("[red]❌ No data fetched[/red]")
            return
            
        console.print(f"[green]📊 Analyzing {len(raw_data)} rows...[/green]")
        
        # Convert to pandas for easier analysis
        if hasattr(raw_data, "to_pandas"):
            df = raw_data.to_pandas()
        else:
            df = raw_data
            
        # Core columns
        core_columns = ["open", "high", "low", "close", "volume"]
        
        # Find rows with any NaN in core columns
        nan_mask = df[core_columns].isna().any(axis=1)
        nan_rows = df[nan_mask]
        
        console.print(f"[yellow]🔍 Found {len(nan_rows)} rows with NaN values[/yellow]")
        
        # Display detailed info about NaN rows
        if len(nan_rows) > 0:
            table = Table(title="NaN Row Details")
            table.add_column("Row Index", style="red")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Open", style="yellow")
            table.add_column("High", style="yellow")
            table.add_column("Low", style="yellow")
            table.add_column("Close", style="yellow")
            table.add_column("Volume", style="yellow")
            
            for idx, row in nan_rows.iterrows():
                timestamp_str = str(row.get('timestamp', 'N/A'))[:19]  # Truncate timestamp
                table.add_row(
                    str(idx),
                    timestamp_str,
                    str(row['open'])[:10],
                    str(row['high'])[:10], 
                    str(row['low'])[:10],
                    str(row['close'])[:10],
                    str(row['volume'])[:10]
                )
                
            console.print(table)
            
            # Show surrounding rows for context
            for idx in nan_rows.index:
                console.print(f"\n[yellow]📍 Context around row {idx}:[/yellow]")
                start_idx = max(0, idx - 2)
                end_idx = min(len(df), idx + 3)
                context_df = df.iloc[start_idx:end_idx]
                
                context_table = Table(title=f"Rows {start_idx} to {end_idx-1}")
                context_table.add_column("Index", style="blue")
                context_table.add_column("Timestamp", style="cyan")
                context_table.add_column("Open", style="green")
                context_table.add_column("High", style="green")
                context_table.add_column("Low", style="green")
                context_table.add_column("Close", style="green")
                context_table.add_column("Volume", style="green")
                
                for i, (ctx_idx, ctx_row) in enumerate(context_df.iterrows()):
                    style = "bold red" if ctx_idx == idx else "white"
                    timestamp_str = str(ctx_row.get('timestamp', 'N/A'))[:19]
                    context_table.add_row(
                        str(ctx_idx),
                        timestamp_str,
                        str(ctx_row['open'])[:8],
                        str(ctx_row['high'])[:8],
                        str(ctx_row['low'])[:8],
                        str(ctx_row['close'])[:8],
                        str(ctx_row['volume'])[:8],
                        style=style
                    )
                    
                console.print(context_table)
        else:
            console.print("[green]✅ No NaN rows found in core columns[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    find_nan_locations()