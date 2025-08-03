#!/usr/bin/env python3
"""
TiRex SAGE Strategy Configuration
Configuration class for the TiRex SAGE Strategy to be used with NautilusTrader backtesting.
"""

from nautilus_trader.config import StrategyConfig


class TiRexSageStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for TiRex SAGE Strategy."""
    instrument_id: str
    min_confidence: float = 0.6
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    model_name: str = "NX-AI/TiRex"
    device: str = "cuda"