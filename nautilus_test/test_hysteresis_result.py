#!/usr/bin/env python3
"""Quick test to run enhanced strategy and analyze results."""

import subprocess
import sys
import pandas as pd
from pathlib import Path
import os

def run_enhanced_strategy():
    """Run the enhanced strategy and capture results."""
    print("🚀 Running Enhanced Strategy with Hysteresis Bands...")
    
    try:
        # Run the strategy
        result = subprocess.run([
            sys.executable, "strategies/backtests/enhanced_sota_strategy_2025.py"
        ], capture_output=True, text=True, timeout=120)
        
        print("Strategy execution completed!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return True
    except Exception as e:
        print(f"Error running strategy: {e}")
        return False

def analyze_latest_results():
    """Analyze the latest trade logs."""
    print("\n📊 Analyzing Latest Results...")
    
    # Find latest trade logs
    trade_logs_dir = Path("trade_logs")
    if not trade_logs_dir.exists():
        print("❌ No trade_logs directory found")
        return
    
    # Get latest files
    trade_files = list(trade_logs_dir.glob("trades_*.csv"))
    signal_files = list(trade_logs_dir.glob("signals_*.csv"))
    
    if not trade_files:
        print("❌ No trade files found")
        return
    
    latest_trade_file = max(trade_files, key=lambda x: x.stat().st_mtime)
    latest_signal_file = max(signal_files, key=lambda x: x.stat().st_mtime) if signal_files else None
    
    print(f"📈 Latest trade file: {latest_trade_file.name}")
    if latest_signal_file:
        print(f"📊 Latest signal file: {latest_signal_file.name}")
    
    try:
        # Analyze trades
        trades_df = pd.read_csv(latest_trade_file)
        if not trades_df.empty:
            print(f"\n💰 Final PnL: ${trades_df['total_pnl'].iloc[-1]:.2f}")
            print(f"📊 Total trades: {len(trades_df)}")
            print(f"🎯 Final equity: ${trades_df['equity'].iloc[-1]:.2f}")
        
        # Analyze signals if available
        if latest_signal_file:
            signals_df = pd.read_csv(latest_signal_file)
            executed_signals = signals_df[signals_df['executed'] == True]
            total_signals = len(signals_df)
            executed_count = len(executed_signals)
            
            if total_signals > 0:
                efficiency = (executed_count / total_signals) * 100
                print(f"⚡ Signal efficiency: {efficiency:.1f}% ({executed_count}/{total_signals})")
            
            # Check regime distribution
            if 'regime' in signals_df.columns:
                regime_counts = signals_df['regime'].value_counts()
                print(f"🧠 Regime distribution: {dict(regime_counts)}")
    
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    if run_enhanced_strategy():
        analyze_latest_results()
    else:
        print("❌ Strategy execution failed")