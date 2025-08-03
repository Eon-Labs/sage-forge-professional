"""
SAGE (Self-Adaptive Generative Evaluation) Meta-Framework

Modular integration of multiple SOTA models for financial time series forecasting:
- AlphaForge: Formulaic alpha factor generation
- TiRex: Zero-shot forecasting with uncertainty quantification  
- catch22: Canonical time series features
- tsfresh: Automated feature extraction

Built upon the proven nautilus_test infrastructure with 100% data quality validation.
"""

# Import only implemented models for Phase 0
from .models import AlphaForgeWrapper, TiRexWrapper, Catch22Wrapper, TSFreshWrapper

# Validation and meta-framework will be added in later phases
# from .validation import ComprehensiveBenchmarkValidator, SAGEEnsembleValidator  
# from .meta_framework import SAGEMetaFramework

__all__ = [
    "AlphaForgeWrapper",
    "TiRexWrapper", 
    "Catch22Wrapper",
    "TSFreshWrapper",
    # "ComprehensiveBenchmarkValidator",
    # "SAGEEnsembleValidator", 
    # "SAGEMetaFramework",
]