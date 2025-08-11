"""
🛡️ Guardian Shield Components: Multi-Layer Protection System

Shield components provide intuitive protection layers that any LLM agent 
can immediately understand as defensive barriers against TiRex vulnerabilities.

PROTECTION LAYERS:
- InputShield: Guards against malicious input attacks (NaN, inf, extreme values)
- CircuitShield: Protects against cascade failures with graceful fallbacks
- DataPipelineShield: Protects against data processing vulnerabilities (scaling, quantiles, tensor ops)
- OutputShield: Validates forecast outputs meet business logic requirements

Each shield is designed with intuitive naming and clear protective purpose.
"""

from .input_shield import InputShield
from .circuit_shield import CircuitShield
from .data_pipeline_shield import DataPipelineShield

__all__ = ['InputShield', 'CircuitShield', 'DataPipelineShield']