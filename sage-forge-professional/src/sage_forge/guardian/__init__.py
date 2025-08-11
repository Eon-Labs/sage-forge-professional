"""
🛡️ TiRex Guardian System: Comprehensive Protective Middleware

PROTECTIVE MIDDLEWARE: Acts as security barrier between application and TiRex model,
guarding against all empirically-validated vulnerabilities while ensuring business continuity.

The Guardian System shields against:
- NaN injection attacks (100% NaN input acceptance vulnerability)
- Infinity propagation attacks (causes model output corruption) 
- Extreme value injection (produces unrealistic forecasts)
- DoS/rate-limit attacks (overwhelming inference requests)
- Model failures and circuit breaking scenarios

Usage Pattern (PRODUCTION REQUIRED):
    from sage_forge.guardian import TiRexGuardian
    
    guardian = TiRexGuardian()  # The protective middleware
    quantiles, mean = guardian.safe_forecast(context, prediction_length)

Architecture:
    - shields/: Multi-layer protection components
    - detectors/: Threat and anomaly detection systems  
    - fallbacks/: Graceful degradation strategies
    - audit/: Comprehensive forensic logging

Empirical Basis:
    Based on comprehensive validation findings in:
    sage-forge-professional/docs/implementation/tirex/empirical-validation/
    
    Confirmed vulnerabilities: 8/8 attack vectors successful against raw TiRex
    Protection effectiveness: All known attacks mitigated through Guardian
"""

from .core import TiRexGuardian
from .shields.input_shield import InputShield
from .shields.circuit_shield import CircuitShield
from .shields.data_pipeline_shield import DataPipelineShield
from .exceptions import (
    GuardianError,
    ShieldViolation, 
    ThreatDetected,
    TiRexServiceUnavailableError
)

# Main entry point for LLM agents - immediately recognizable as protective interface
__all__ = [
    'TiRexGuardian',         # 🛡️ Main protective interface
    'InputShield',           # 🛡️ Input validation protection  
    'CircuitShield',         # 🛡️ Failure handling protection
    'DataPipelineShield',    # 🛡️ Data pipeline safety protection
    'GuardianError',         # ⚠️ Guardian system errors
    'ShieldViolation',       # ⚠️ Protection boundary violations
    'ThreatDetected',        # 🚨 Security threat detection
    'TiRexServiceUnavailableError'  # 🚨 Service failure conditions
]

# Version tracking for audit purposes
__version__ = "1.0.0"
__guardian_version__ = "1.0.0"  # Specific guardian system version