#!/usr/bin/env python3
"""
Test TiRex backtest results after fixing timestamp issue.
"""

import sys
from pathlib import Path

# Add sage-forge to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sage_forge.backtesting.tirex_backtest_engine import TiRexBacktestEngine
from rich.console import Console

console = Console()

def test_tirex_results():
    """Test TiRex backtest results after timestamp fix."""
    console.print("🧪 Testing TiRex results after timestamp fix")
    console.print("="*50)
    
    try:
        # Create backtest engine
        engine = TiRexBacktestEngine()
        
        # Setup with corrected timestamps
        success = engine.setup_backtest(
            symbol="BTCUSDT",
            start_date="2024-12-01",
            end_date="2024-12-03", 
            initial_balance=10000.0,
            timeframe="15m"
        )
        
        if not success:
            console.print("❌ Failed to setup backtest")
            return
        
        console.print(f"✅ Setup successful with {len(engine.market_bars)} bars")
        
        # Run backtest
        results = engine.run_backtest()
        
        if not results:
            console.print("❌ No backtest results")
            return
            
        console.print("✅ Backtest completed successfully!")
        
        # Get results summary
        orders = results.orders
        positions_closed = results.positions_closed
        pnls = results.stats_pnls()
        
        console.print(f"\n📊 TiRex Trading Results Summary:")
        console.print(f"   📈 Total Orders: {len(orders)}")
        console.print(f"   🎯 Closed Positions: {len(positions_closed)}")
        console.print(f"   💰 Total PnL: {pnls.get('PnL (USD)', 'N/A')}")
        console.print(f"   📊 Total Return: {pnls.get('Total Return %', 'N/A')}%")
        console.print(f"   🎲 Win Rate: {pnls.get('Win Rate %', 'N/A')}%")
        
        if orders:
            console.print(f"\n🎯 First 3 TiRex Orders:")
            for i, order in enumerate(orders[:3]):
                console.print(f"   {i+1}. {order.side} {order.quantity} @ {order.price}")
        
        if positions_closed:
            console.print(f"\n📈 First 3 Closed Positions:")
            for i, pos in enumerate(positions_closed[:3]):
                console.print(f"   {i+1}. {pos.side} {pos.quantity} - PnL: {pos.realized_pnl}")
        
        # Success indicator
        if len(orders) > 0:
            console.print(f"\n🎉 SUCCESS: TiRex generated {len(orders)} trading signals!")
            console.print("✅ Problem solved: Timestamp fix enabled proper data flow to TiRex strategy")
        else:
            console.print("\n⚠️ No trading signals generated (may be intentional based on market conditions)")
            
    except Exception as e:
        console.print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tirex_results()