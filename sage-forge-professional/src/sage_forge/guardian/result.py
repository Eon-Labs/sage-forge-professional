"""
🛡️ Guardian Result Types
========================

Result objects returned by Guardian protection layers.
"""

import torch
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class GuardianResult:
    """
    Result object returned by Guardian protected inference.
    
    Attributes:
        is_blocked: True if inference was blocked by security measures
        quantiles: TiRex quantile predictions (None if blocked)
        mean: TiRex mean prediction (None if blocked)  
        block_reason: Reason for blocking (None if not blocked)
        threat_level: Detected threat level
        processing_time_ms: Time taken for protected inference
        shield_activations: Which shields were activated
    """
    is_blocked: bool
    quantiles: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None
    block_reason: Optional[str] = None
    threat_level: str = "none"
    processing_time_ms: float = 0.0
    shield_activations: Optional[dict] = None
    
    @classmethod
    def success(cls, quantiles: torch.Tensor, mean: torch.Tensor, 
                processing_time_ms: float = 0.0, shield_activations: dict = None):
        """Create successful result."""
        return cls(
            is_blocked=False,
            quantiles=quantiles,
            mean=mean,
            processing_time_ms=processing_time_ms,
            shield_activations=shield_activations or {}
        )
    
    @classmethod
    def blocked(cls, reason: str, threat_level: str = "medium",
                processing_time_ms: float = 0.0, shield_activations: dict = None):
        """Create blocked result."""
        return cls(
            is_blocked=True,
            block_reason=reason,
            threat_level=threat_level,
            processing_time_ms=processing_time_ms,
            shield_activations=shield_activations or {}
        )