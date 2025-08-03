"""
SAGE-Forge Backtesting Module

NT-native backtesting framework with DSM integration for TiRex SAGE strategies.
Provides comprehensive backtesting capabilities using real market data.
"""

from .tirex_backtest_engine import TiRexBacktestEngine, create_sample_backtest

__all__ = [
    "TiRexBacktestEngine",
    "create_sample_backtest"
]