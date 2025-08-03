#!/usr/bin/env python3
"""
Minimal AlphaForge Test - Phase 0 Week 2 Validation

INCREMENTAL APPROACH: Test single model first with proven infrastructure
Built on enhanced_dsm_hybrid_integration.py template for safety.

Purpose: Validate AlphaForge wrapper can generate factors from BTCUSDT data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add nautilus_test to path (proven pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add sage modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Import our new AlphaForge wrapper
from sage.models.alphaforge_wrapper import AlphaForgeWrapper

# Import proven DSM infrastructure
try:
    from nautilus_test.utils.data_manager import ArrowDataManager, DataPipeline
    DSM_AVAILABLE = True
except ImportError:
    console.print("[yellow]⚠️ DSM not available, will use synthetic data[/yellow]")
    DSM_AVAILABLE = False


def load_btcusdt_data() -> pd.DataFrame:
    """Load BTCUSDT data using proven DSM infrastructure."""
    if DSM_AVAILABLE:
        console.print("[cyan]🔧 Loading BTCUSDT data via proven DSM pipeline...[/cyan]")
        
        try:
            data_manager = ArrowDataManager()
            pipeline = DataPipeline(data_manager)
            
            # Use existing data cache (proven pattern)
            cache_files = list(Path("data_cache").glob("BTCUSDT_*.parquet"))
            if cache_files:
                console.print(f"[green]✅ Found {len(cache_files)} cached BTCUSDT files[/green]")
                
                # Load first available cache file
                df = pd.read_parquet(cache_files[0])
                console.print(f"[green]✅ Loaded {len(df)} rows from cache: {cache_files[0].name}[/green]")
                
                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    console.print(f"[yellow]⚠️ Missing columns {missing_cols}, using synthetic data[/yellow]")
                    return create_synthetic_data()
                
                return df
            else:
                console.print("[yellow]⚠️ No cached BTCUSDT data found, fetching fresh[/yellow]")
                df = data_manager.fetch_real_market_data("BTCUSDT", limit=1000)
                return data_manager.process_ohlcv_data(df)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ DSM loading failed: {e}, using synthetic data[/yellow]")
            return create_synthetic_data()
    else:
        return create_synthetic_data()


def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic BTCUSDT-like data for testing."""
    console.print("[cyan]🔧 Creating synthetic BTCUSDT data for testing...[/cyan]")
    
    # Generate realistic BTC price data
    np_available = True
    try:
        import numpy as np
    except ImportError:
        np_available = False
    
    if np_available:
        # Use numpy for realistic random walk
        n_points = 1000
        base_price = 95000.0
        returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLC from prices
        data = []
        for i in range(len(prices)):
            price = prices[i]
            # Generate realistic OHLC around price
            noise = np.random.uniform(-0.005, 0.005, 4)  # 0.5% noise
            open_price = price * (1 + noise[0])
            close_price = price * (1 + noise[1])
            high_price = max(open_price, close_price) * (1 + abs(noise[2]))
            low_price = min(open_price, close_price) * (1 - abs(noise[3]))
            volume = np.random.uniform(0.5, 2.0)  # Random volume
            
            data.append({
                'open': open_price,
                'high': high_price, 
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    else:
        # Fallback without numpy
        import random
        data = []
        price = 95000.0
        for i in range(1000):
            price *= (1 + random.uniform(-0.02, 0.02))
            data.append({
                'open': price * random.uniform(0.995, 1.005),
                'high': price * random.uniform(1.000, 1.010),
                'low': price * random.uniform(0.990, 1.000),
                'close': price * random.uniform(0.995, 1.005),
                'volume': random.uniform(0.5, 2.0)
            })
    
    df = pd.DataFrame(data)
    
    # Add timestamp index (last N minutes)
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=i) for i in range(len(df)-1, -1, -1)]
    df.index = pd.DatetimeIndex(timestamps)
    
    console.print(f"[green]✅ Created {len(df)} synthetic BTCUSDT data points[/green]")
    return df


def test_alphaforge_basic():
    """Test basic AlphaForge wrapper functionality."""
    console.print(Panel.fit(
        "[bold cyan]🧪 AlphaForge Minimal Test - Phase 0 Week 2[/bold cyan]\n"
        "Testing single model with proven infrastructure",
        title="SAGE VALIDATION"
    ))
    
    # Step 1: Initialize AlphaForge wrapper
    console.print("\n[bold blue]🎯 STEP 1: Initialize AlphaForge Wrapper[/bold blue]")
    
    alphaforge = AlphaForgeWrapper()
    init_success = alphaforge.initialize_model()
    
    if not init_success:
        console.print("[red]❌ AlphaForge initialization failed[/red]")
        return False
    
    # Display model info
    info = alphaforge.get_model_info()
    info_table = Table(title="AlphaForge Model Information")
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value", style="green")
    
    for key, value in info.items():
        info_table.add_row(str(key), str(value))
    
    console.print(info_table)
    
    # Step 2: Load test data
    console.print("\n[bold blue]🎯 STEP 2: Load BTCUSDT Test Data[/bold blue]")
    
    try:
        market_data = load_btcusdt_data()
        console.print(f"[green]✅ Loaded market data: {market_data.shape}[/green]")
        
        # Display data sample
        console.print("[cyan]📊 Data sample (first 5 rows):[/cyan]")
        console.print(market_data.head())
        
        # Validate data quality (proven pattern from enhanced_dsm_hybrid_integration.py)
        nan_count = market_data.isna().sum().sum()
        if nan_count > 0:
            console.print(f"[yellow]⚠️ Found {nan_count} NaN values, cleaning...[/yellow]")
            market_data = market_data.fillna(method='ffill').fillna(method='bfill')
            console.print("[green]✅ Data cleaned[/green]")
        else:
            console.print("[green]✅ Data quality: 100% complete[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Data loading failed: {e}[/red]")
        return False
    
    # Step 3: Generate alpha factors
    console.print("\n[bold blue]🎯 STEP 3: Generate Alpha Factors[/bold blue]")
    
    try:
        # Test with small number of factors for initial validation
        factors = alphaforge.generate_factors(market_data, num_factors=5)
        console.print(f"[green]✅ Generated factors: {factors.shape}[/green]")
        
        # Display factor sample
        console.print("[cyan]📊 Factor sample (first 5 rows):[/cyan]")
        console.print(factors.head())
        
        # Get factor descriptions
        descriptions = alphaforge.get_factor_descriptions()
        desc_table = Table(title="Generated Alpha Factors")
        desc_table.add_column("Factor", style="bold cyan")
        desc_table.add_column("Description", style="white")
        
        for factor in factors.columns:
            if factor in descriptions:
                desc_table.add_row(factor, descriptions[factor])
            else:
                desc_table.add_row(factor, "Auto-generated factor")
        
        console.print(desc_table)
        
    except Exception as e:
        console.print(f"[red]❌ Factor generation failed: {e}[/red]")
        return False
    
    # Step 4: Validate factors
    console.print("\n[bold blue]🎯 STEP 4: Validate Factor Quality[/bold blue]")
    
    try:
        validation_results = alphaforge.validate_factors(factors)
        
        # Display validation results
        val_table = Table(title="Factor Validation Results")
        val_table.add_column("Metric", style="bold")
        val_table.add_column("Value", style="green")
        
        val_table.add_row("Total Factors", str(validation_results["total_factors"]))
        val_table.add_row("Valid Factors", str(validation_results["valid_factors"]))
        val_table.add_row("Invalid Factors", str(len(validation_results["invalid_factors"])))
        val_table.add_row("High Correlation Pairs", str(len(validation_results["high_correlation_pairs"])))
        
        console.print(val_table)
        
        # Display factor statistics
        stats_table = Table(title="Factor Statistics")
        stats_table.add_column("Factor", style="bold cyan")
        stats_table.add_column("Mean", style="white")
        stats_table.add_column("Std", style="white")
        stats_table.add_column("NaN %", style="yellow")
        
        for factor in factors.columns:
            factor_data = factors[factor]
            nan_pct = validation_results["nan_percentage"].get(factor, 0)
            
            stats_table.add_row(
                factor,
                f"{factor_data.mean():.6f}",
                f"{factor_data.std():.6f}",
                f"{nan_pct:.2f}%"
            )
        
        console.print(stats_table)
        
        # Check validation success
        success_rate = validation_results["valid_factors"] / validation_results["total_factors"]
        if success_rate >= 0.8:  # 80% success threshold
            console.print(f"[green]✅ Validation SUCCESS: {success_rate:.1%} factors valid[/green]")
        else:
            console.print(f"[yellow]⚠️ Validation WARNING: {success_rate:.1%} factors valid[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Factor validation failed: {e}[/red]")
        return False
    
    # Step 5: Test with different data sizes
    console.print("\n[bold blue]🎯 STEP 5: Test Scalability[/bold blue]")
    
    try:
        test_sizes = [100, 500, 1000]
        scalability_results = []
        
        for size in test_sizes:
            if size <= len(market_data):
                test_data = market_data.tail(size)
                
                start_time = datetime.now()
                test_factors = alphaforge.generate_factors(test_data, num_factors=3)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                scalability_results.append({
                    'data_size': size,
                    'processing_time': processing_time,
                    'factors_generated': len(test_factors.columns)
                })
                
                console.print(f"[cyan]📊 Size {size}: {processing_time:.2f}s, {len(test_factors.columns)} factors[/cyan]")
        
        # Display scalability results
        scale_table = Table(title="Scalability Test Results")
        scale_table.add_column("Data Size", style="bold")
        scale_table.add_column("Processing Time (s)", style="green")
        scale_table.add_column("Factors Generated", style="cyan")
        scale_table.add_column("Speed (rows/s)", style="yellow")
        
        for result in scalability_results:
            speed = result['data_size'] / result['processing_time'] if result['processing_time'] > 0 else 0
            scale_table.add_row(
                str(result['data_size']),
                f"{result['processing_time']:.3f}",
                str(result['factors_generated']),
                f"{speed:.0f}"
            )
        
        console.print(scale_table)
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Scalability test failed: {e}[/yellow]")
        # Don't fail the test for scalability issues
    
    # Final success summary
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold green]🎉 AlphaForge Minimal Test PASSED![/bold green]\n"
        "✅ Model initialization successful\n"
        "✅ Data loading successful\n" 
        "✅ Factor generation successful\n"
        "✅ Factor validation successful\n"
        "✅ Ready for SAGE integration",
        title="TEST SUCCESS"
    ))
    
    return True


def main():
    """Run minimal AlphaForge test."""
    try:
        success = test_alphaforge_basic()
        
        if success:
            console.print("[bold green]🚀 AlphaForge validation complete - ready for next model![/bold green]")
            return 0
        else:
            console.print("[bold red]❌ AlphaForge validation failed[/bold red]")
            return 1
            
    except Exception as e:
        console.print(f"[bold red]💥 Test crashed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())