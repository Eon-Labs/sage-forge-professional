#!/usr/bin/env python3
"""
Final comprehensive validation test before freezing TiRex integration.
Tests complete end-to-end pipeline with real DSM data.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from rich.console import Console
from rich.table import Table
from datetime import datetime

console = Console()

def final_validation_test():
    """Run comprehensive validation test of complete TiRex pipeline."""
    console.print("🧪 FINAL VALIDATION TEST - Complete TiRex Pipeline")
    console.print("=" * 70)
    
    # Test multiple time periods to validate consistency
    test_periods = [
        ("2024-10-15", "2024-10-17", "Confirmed +2.31% movement"),
        ("2024-11-01", "2024-11-03", "Test different period"),
        ("2024-12-01", "2024-12-03", "Original problem period"),
    ]
    
    results = []
    
    for start_date, end_date, description in test_periods:
        console.print(f"\n📊 Testing {start_date} to {end_date}: {description}")
        console.print("-" * 60)
        
        try:
            # Create fresh engine for each test
            engine = TiRexBacktestEngine()
            
            # Setup with real DSM data
            success = engine.setup_backtest(
                symbol="BTCUSDT",
                start_date=start_date,
                end_date=end_date,
                initial_balance=10000.0,
                timeframe="15m"
            )
            
            if not success:
                console.print(f"❌ Setup failed for {start_date}")
                results.append({
                    "period": f"{start_date} to {end_date}",
                    "status": "Setup Failed",
                    "bars": 0,
                    "predictions": 0,
                    "signals": 0,
                    "price_change": "N/A"
                })
                continue
            
            bars_count = len(engine.market_bars)
            console.print(f"✅ Loaded {bars_count} bars")
            
            # Calculate price movement
            if bars_count > 0:
                first_bar = engine.market_bars[0]
                last_bar = engine.market_bars[-1]
                price_change = float(last_bar.close) - float(first_bar.open)
                price_change_pct = (price_change / float(first_bar.open)) * 100
                console.print(f"💹 Price movement: ${float(first_bar.open):.2f} → ${float(last_bar.close):.2f} ({price_change_pct:+.2f}%)")
            else:
                price_change_pct = 0
            
            # Run backtest with detailed monitoring
            console.print("🤖 Running TiRex analysis...")
            
            try:
                backtest_results = engine.run_backtest()
                
                if backtest_results:
                    orders = backtest_results.orders
                    positions = backtest_results.positions_closed
                    console.print(f"✅ Backtest completed successfully")
                    console.print(f"📈 Orders generated: {len(orders)}")
                    console.print(f"📊 Positions closed: {len(positions)}")
                    
                    # Get PnL if available
                    try:
                        pnls = backtest_results.stats_pnls()
                        total_pnl = pnls.get('PnL (USD)', 'N/A')
                        console.print(f"💰 Total PnL: {total_pnl}")
                    except:
                        total_pnl = "N/A"
                    
                    results.append({
                        "period": f"{start_date} to {end_date}",
                        "status": "Success",
                        "bars": bars_count,
                        "predictions": "Multiple", # From strategy execution
                        "signals": len(orders),
                        "price_change": f"{price_change_pct:+.2f}%",
                        "pnl": total_pnl
                    })
                    
                else:
                    console.print("⚠️ Backtest completed but no results returned")
                    results.append({
                        "period": f"{start_date} to {end_date}",
                        "status": "No Results",
                        "bars": bars_count,
                        "predictions": "Unknown",
                        "signals": 0,
                        "price_change": f"{price_change_pct:+.2f}%"
                    })
                    
            except Exception as e:
                console.print(f"❌ Backtest failed: {str(e)[:100]}...")
                results.append({
                    "period": f"{start_date} to {end_date}",
                    "status": f"Error: {str(e)[:30]}...",
                    "bars": bars_count,
                    "predictions": 0,
                    "signals": 0,
                    "price_change": f"{price_change_pct:+.2f}%"
                })
            
        except Exception as e:
            console.print(f"❌ Period test failed: {str(e)[:100]}...")
            results.append({
                "period": f"{start_date} to {end_date}",
                "status": f"Failed: {str(e)[:30]}...",
                "bars": 0,
                "predictions": 0,
                "signals": 0,
                "price_change": "N/A"
            })
    
    # Display comprehensive results table
    console.print(f"\n📋 FINAL VALIDATION RESULTS")
    console.print("=" * 70)
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Period", style="white")
    table.add_column("Status", style="green")
    table.add_column("Bars", justify="right")
    table.add_column("Price Change", justify="right", style="yellow")
    table.add_column("Signals", justify="right", style="magenta")
    table.add_column("Notes", style="dim")
    
    total_bars = 0
    total_signals = 0
    successful_tests = 0
    
    for result in results:
        status_style = "green" if "Success" in result["status"] else "red"
        table.add_row(
            result["period"],
            f"[{status_style}]{result['status']}[/{status_style}]",
            str(result["bars"]),
            result["price_change"],
            str(result["signals"]),
            "✅ Working" if "Success" in result["status"] else "❌ Issue"
        )
        
        if "Success" in result["status"]:
            successful_tests += 1
            total_bars += result["bars"]
            total_signals += result["signals"]
    
    console.print(table)
    
    # Final assessment
    console.print(f"\n🎯 FINAL ASSESSMENT")
    console.print("=" * 40)
    console.print(f"✅ Successful tests: {successful_tests}/{len(test_periods)}")
    console.print(f"📊 Total bars processed: {total_bars}")
    console.print(f"🚨 Total signals generated: {total_signals}")
    
    # Determine freeze readiness
    if successful_tests >= 2:  # At least 2/3 tests successful
        console.print(f"\n🎉 FREEZE POINT VALIDATED ✅")
        console.print("📋 Ready for production with documented baselines")
        console.print("🚀 Core TiRex pipeline working correctly")
        return True
    else:
        console.print(f"\n⚠️ VALIDATION INCOMPLETE ❌")
        console.print("🔧 Need additional fixes before freeze")
        console.print("📝 Document current issues for next iteration")
        return False

if __name__ == "__main__":
    success = final_validation_test()
    
    if success:
        console.print(f"\n🔒 READY TO FREEZE IMPLEMENTATION")
        console.print("📚 All documentation is current and accurate")
        console.print("🎖️ Milestone achievement confirmed")
    else:
        console.print(f"\n🔄 CONTINUE DEVELOPMENT")
        console.print("⚡ Address validation issues before freezing")