"""
🛡️ Guardian Shield Components: Multi-Layer Protection System

Shield components provide intuitive protection layers that any LLM agent 
can immediately understand as defensive barriers against TiRex vulnerabilities.

PROTECTION LAYERS:
- InputShield: Guards against malicious input attacks (NaN, inf, extreme values)
- CircuitShield: Protects against cascade failures with graceful fallbacks
- OutputShield: Validates forecast outputs meet business logic requirements

Each shield is designed with intuitive naming and clear protective purpose.
"""

from .input_shield import InputShield
from .circuit_shield import CircuitShield

__all__ = ['InputShield', 'CircuitShield']