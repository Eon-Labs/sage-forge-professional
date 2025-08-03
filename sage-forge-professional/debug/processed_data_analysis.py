#!/usr/bin/env python3
"""
🔍 Processed Data Analysis - Investigate remaining NaN after cleaning

Analyzes the processed data after cleaning to understand why NaN still appears during bar creation.
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
from sage_forge.data.enhanced_provider import EnhancedModernBarDataProvider
from sage_forge.market.binance_specs import BinanceSpecificationManager

console = Console()


def analyze_processed_data():
    """Analyze processed data after cleaning step."""
    console.print(Panel.fit("🔍 Analyzing processed data after cleaning", style="bold blue"))
    
    # Initialize components
    data_manager = ArrowDataManager()
    specs_manager = BinanceSpecificationManager()
    provider = EnhancedModernBarDataProvider(specs_manager)
    
    try:
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
        
        # Fetch raw data
        raw_data = data_manager.fetch_real_market_data(
            symbol="BTCUSDT",
            timeframe="1m", 
            limit=500,
            start_time=start_time,
            end_time=end_time
        )
        
        if raw_data is None:
            console.print("[red]❌ No raw data fetched[/red]")
            return
            
        console.print(f"[green]📊 Raw data: {len(raw_data)} rows[/green]")
        
        # Process data 
        processed_data = data_manager.process_ohlcv_data(raw_data)
        console.print(f"[green]📊 Processed data: {len(processed_data)} rows[/green]")
        
        # Clean corrupt rows
        cleaned_data = provider._clean_corrupt_rows(processed_data)
        console.print(f"[green]📊 Cleaned data: {len(cleaned_data)} rows[/green]")
        
        # Convert to pandas for analysis
        if hasattr(cleaned_data, "to_pandas"):
            df = cleaned_data.to_pandas()
        else:
            df = cleaned_data
            
        # Check for any remaining NaN values
        core_columns = ["open", "high", "low", "close", "volume"]
        
        console.print("\n[yellow]🔍 Checking for remaining NaN values...[/yellow]")
        
        total_nans = 0
        for col in core_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                total_nans += nan_count
                console.print(f"  {col}: {nan_count} NaN values")
                
        console.print(f"[blue]📊 Total core NaN values after cleaning: {total_nans}[/blue]")
        
        if total_nans > 0:
            # Find remaining NaN rows
            nan_mask = df[core_columns].isna().any(axis=1)
            nan_rows = df[nan_mask]
            
            console.print(f"[red]⚠️ Found {len(nan_rows)} rows with remaining NaN values![/red]")
            
            # Show details
            table = Table(title="Remaining NaN Rows After Cleaning")
            table.add_column("Row Index", style="red")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Open", style="yellow")
            table.add_column("High", style="yellow")
            table.add_column("Low", style="yellow")
            table.add_column("Close", style="yellow")
            table.add_column("Volume", style="yellow")
            
            for idx, row in nan_rows.iterrows():
                timestamp_str = str(row.get('timestamp', 'N/A'))[:19]
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
            
        else:
            console.print("[green]✅ No remaining NaN values in core columns![/green]")
            
        # Check other columns for NaN (especially derived indicators)
        console.print("\n[yellow]🔍 Checking all columns for NaN values...[/yellow]")
        
        all_nan_table = Table(title="NaN Count by Column")
        all_nan_table.add_column("Column", style="cyan")
        all_nan_table.add_column("NaN Count", style="red")
        all_nan_table.add_column("% Complete", style="green")
        
        for col in df.columns:
            nan_count = df[col].isna().sum()
            completeness = ((len(df) - nan_count) / len(df) * 100) if len(df) > 0 else 0
            
            style = "bold red" if col in core_columns and nan_count > 0 else "white"
            all_nan_table.add_row(
                col,
                str(nan_count),
                f"{completeness:.2f}%",
                style=style
            )
            
        console.print(all_nan_table)
        
        # Check data types to see if type conversion could be causing issues
        console.print("\n[yellow]🔍 Checking data types...[/yellow]")
        
        dtype_table = Table(title="Data Types")
        dtype_table.add_column("Column", style="cyan")
        dtype_table.add_column("Data Type", style="yellow")
        dtype_table.add_column("Sample Value", style="green")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_val = str(df[col].iloc[0] if len(df) > 0 and not df[col].isna().all() else "N/A")[:20]
            dtype_table.add_row(col, dtype, sample_val)
            
        console.print(dtype_table)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_processed_data()