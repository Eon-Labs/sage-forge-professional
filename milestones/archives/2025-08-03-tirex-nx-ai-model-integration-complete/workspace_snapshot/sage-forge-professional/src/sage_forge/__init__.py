"""
🔥 SAGE-Forge: Self-Adaptive Generative Evaluation Framework

A professional, bulletproof infrastructure for developing adaptive trading strategies
with NautilusTrader, real market data, and zero trial-and-error setup.

Core Components:
- SAGE Meta-Framework: Self-adaptive model ensemble with dynamic weighting
- NT-Native Integration: Proper NautilusTrader patterns and conventions
- Real Data Pipeline: DSM integration with 100% real market data
- Enhanced Funding Integration: Professional funding rate tracking and analytics
- Professional Visualization: Enhanced FinPlot actors with real-time updates
- Production Infrastructure: Bulletproof UV-native setup with comprehensive CLI
"""

__version__ = "0.1.0"
__author__ = "SAGE Development Team"
__email__ = "dev@sage-forge.com"

# Core configuration
from sage_forge.core.config import get_config

# Data management with real market data integration
from sage_forge.data.manager import ArrowDataManager, DataPipeline
from sage_forge.data.enhanced_provider import EnhancedModernBarDataProvider

# Market specifications and risk management
from sage_forge.market.binance_specs import BinanceSpecificationManager
from sage_forge.risk.position_sizer import RealisticPositionSizer

# Enhanced funding integration
from sage_forge.funding.actor import FundingActor
from sage_forge.funding.data import FundingPaymentEvent, FundingRateUpdate
from sage_forge.funding.provider import FundingRateProvider

# SAGE meta-framework and models
from sage_forge.models.base import BaseSAGEModel
from sage_forge.models.ensemble import SAGEEnsemble

# Adaptive trading strategies
from sage_forge.strategies.adaptive_regime import AdaptiveRegimeStrategy

# Professional visualization
from sage_forge.visualization.actors import EnhancedFinPlotActor
from sage_forge.visualization.native_finplot_actor import FinplotActor

# Enhanced performance reporting
from sage_forge.reporting.performance import (
    display_ultimate_performance_summary,
    create_performance_report_dataframe,
    export_performance_summary,
)

# Version info and core exports
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Core
    "get_config",
    # Data
    "ArrowDataManager",
    "DataPipeline", 
    "EnhancedModernBarDataProvider",
    # Market & Risk
    "BinanceSpecificationManager",
    "RealisticPositionSizer",
    # Funding
    "FundingActor",
    "FundingPaymentEvent", 
    "FundingRateProvider",
    "FundingRateUpdate",
    # Models
    "BaseSAGEModel",
    "SAGEEnsemble",
    # Strategies
    "AdaptiveRegimeStrategy",
    # Visualization
    "EnhancedFinPlotActor",
    "FinplotActor",
    # Reporting
    "display_ultimate_performance_summary",
    "create_performance_report_dataframe", 
    "export_performance_summary",
]