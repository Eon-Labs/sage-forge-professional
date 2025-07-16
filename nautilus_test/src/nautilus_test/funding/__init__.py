"""
Production-ready funding rate system for crypto perpetual futures.

This module provides comprehensive funding rate handling for cryptocurrency
perpetual futures contracts with enhanced features:

- Enhanced funding rate data collection (DSM + Direct API)
- Real-time funding rate monitoring with robust error handling
- Production-ready backtesting integration
- Mathematical validation and temporal accuracy
- Native NautilusTrader integration patterns

Designed following NautilusTrader's native architecture with zero redundancy.
"""

from nautilus_test.funding.actor import (
    FundingActor,
    FundingActorConfig,
    add_funding_actor_to_engine,
)
from nautilus_test.funding.backtest_integrator import BacktestFundingIntegrator
from nautilus_test.funding.calculator import FundingPaymentCalculator
from nautilus_test.funding.data import FundingPaymentEvent, FundingRateUpdate
from nautilus_test.funding.provider import FundingRateProvider

__all__ = [
    "BacktestFundingIntegrator",
    "FundingActor",
    "FundingActorConfig",
    "FundingPaymentCalculator",
    "FundingPaymentEvent",
    "FundingRateProvider",
    "FundingRateUpdate",
    "add_funding_actor_to_engine",
]
