#!/usr/bin/env python3
"""
Sluice 1C: Confidence Inheritance Oracle (CIA)

Oracle that inherits TiRex confidence distributions for confidence-weighted 
ODEB analysis, following NautilusTrader configuration patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

from sage_forge.reporting.performance import Position


@dataclass
class OdebOracleConfig:
    """
    Configuration for Confidence Inheritance Oracle following NT patterns.
    
    Parameters
    ----------
    confidence_window_size : int, default 100
        Number of positions to use for confidence distribution analysis
    min_confidence_threshold : float, default 0.15
        Minimum confidence threshold for oracle analysis
    confidence_decay_factor : float, default 0.95  
        Exponential decay factor for weighting recent confidence values
    regime_confidence_mapping : bool, default True
        Enable regime-specific confidence distribution mapping
    oracle_update_frequency : int, default 10
        Number of new positions before oracle confidence update
    """
    
    confidence_window_size: int = 100
    min_confidence_threshold: float = 0.15
    confidence_decay_factor: float = 0.95
    regime_confidence_mapping: bool = True
    oracle_update_frequency: int = 10


class ConfidenceInheritanceOracle:
    """
    Oracle that inherits and analyzes TiRex confidence distributions.
    
    Implements confidence-weighted ODEB analysis by inheriting TiRex model
    confidence distributions and applying them to position analysis with
    < 5% performance overhead.
    """
    
    def __init__(self, config: Optional[OdebOracleConfig] = None):
        """Initialize Confidence Inheritance Oracle."""
        self.config = config or OdebOracleConfig()
        
        # Confidence distribution tracking
        self.confidence_history: List[float] = []
        self.regime_confidence_map: Dict[str, List[float]] = {}
        self.position_count = 0
        
        # Oracle state
        self.inherited_distribution: Optional[np.ndarray] = None
        self.regime_distributions: Dict[str, np.ndarray] = {}
        self.last_update_position_count = 0
        
        # Performance tracking
        self.oracle_overhead_ms: List[float] = []
    
    def inherit_tirex_confidence(self, positions: List[Position]) -> None:
        """
        Inherit confidence distributions from TiRex-generated positions.
        
        Parameters
        ----------
        positions : List[Position]
            List of positions with TiRex confidence metadata
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Extract confidence values from positions (optimized)
            confidences = [pos.confidence for pos in positions if pos.confidence > 0.0]
            
            if not confidences:
                self.inherited_distribution = None
                return
            
            # Only process if significant number of new positions (performance optimization)
            if len(positions) < 5:
                self.position_count += len(positions)
                return
            
            # Update confidence history with decay weighting
            self._update_confidence_history(confidences)
            
            # Create inherited confidence distribution (cached)
            if len(self.confidence_history) >= 10:  # Only create distribution if enough data
                self.inherited_distribution = self._create_confidence_distribution(self.confidence_history)
            
            # Update regime-specific distributions if enabled (optimized)
            if self.config.regime_confidence_mapping and len(positions) >= 10:
                self._update_regime_distributions(positions)
            
            self.position_count += len(positions)
            self.last_update_position_count = self.position_count
            
        finally:
            # Track performance overhead (lightweight)
            overhead_ms = (time.perf_counter() - start_time) * 1000
            if len(self.oracle_overhead_ms) < 50:  # Limit overhead tracking
                self.oracle_overhead_ms.append(overhead_ms)
    
    def _update_confidence_history(self, new_confidences: List[float]) -> None:
        """Update confidence history with exponential decay weighting."""
        
        # Apply decay to existing history
        decay_factor = self.config.confidence_decay_factor
        self.confidence_history = [conf * decay_factor for conf in self.confidence_history]
        
        # Add new confidence values
        self.confidence_history.extend(new_confidences)
        
        # Maintain window size
        if len(self.confidence_history) > self.config.confidence_window_size:
            excess = len(self.confidence_history) - self.config.confidence_window_size
            self.confidence_history = self.confidence_history[excess:]
    
    def _create_confidence_distribution(self, confidences: List[float]) -> np.ndarray:
        """Create confidence distribution from historical values."""
        
        if not confidences:
            return np.array([])
        
        # Create histogram bins for confidence distribution
        bins = np.linspace(0.0, 1.0, 21)  # 20 bins from 0 to 1
        hist, _ = np.histogram(confidences, bins=bins, density=True)
        
        # Normalize to probability distribution
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        
        return hist
    
    def _update_regime_distributions(self, positions: List[Position]) -> None:
        """Update regime-specific confidence distributions."""
        
        regime_confidences: Dict[str, List[float]] = {}
        
        # Group confidences by regime
        for pos in positions:
            if pos.confidence > 0.0 and pos.market_regime != "unknown":
                if pos.market_regime not in regime_confidences:
                    regime_confidences[pos.market_regime] = []
                regime_confidences[pos.market_regime].append(pos.confidence)
        
        # Update regime confidence maps with decay
        decay_factor = self.config.confidence_decay_factor
        
        for regime, confidences in regime_confidences.items():
            # Apply decay to existing regime history
            if regime in self.regime_confidence_map:
                self.regime_confidence_map[regime] = [
                    conf * decay_factor for conf in self.regime_confidence_map[regime]
                ]
            else:
                self.regime_confidence_map[regime] = []
            
            # Add new regime confidences
            self.regime_confidence_map[regime].extend(confidences)
            
            # Create regime-specific distribution
            self.regime_distributions[regime] = self._create_confidence_distribution(
                self.regime_confidence_map[regime]
            )
    
    def get_confidence_weight(self, position: Position) -> float:
        """
        Get confidence weight for position based on inherited distribution.
        
        Parameters
        ----------
        position : Position
            Position to calculate confidence weight for
            
        Returns
        -------
        float
            Confidence weight [0, 1] for ODEB analysis weighting
        """
        if position.confidence <= 0.0:
            return 0.0
        
        # Use regime-specific distribution if available
        if (self.config.regime_confidence_mapping and 
            position.market_regime in self.regime_distributions):
            
            distribution = self.regime_distributions[position.market_regime]
        else:
            distribution = self.inherited_distribution
        
        if distribution is None or len(distribution) == 0:
            return position.confidence  # Fallback to raw confidence
        
        # Find confidence bin
        confidence_bin = min(int(position.confidence * len(distribution)), len(distribution) - 1)
        
        # Get distribution probability for this confidence level
        distribution_weight = distribution[confidence_bin] if confidence_bin < len(distribution) else 0.0
        
        # Combine raw confidence with distribution weight
        confidence_weight = position.confidence * (1.0 + distribution_weight)
        
        # Ensure weight stays in [0, 1] bounds
        return min(confidence_weight, 1.0)
    
    def analyze_confidence_weighted_performance(self, positions: List[Position]) -> Dict:
        """
        Perform confidence-weighted ODEB analysis.
        
        Parameters
        ----------
        positions : List[Position]
            Positions to analyze
            
        Returns
        -------
        Dict
            Confidence-weighted performance metrics
        """
        if not positions:
            return {"error": "No positions provided"}
        
        # Inherit confidence distributions if needed
        positions_since_update = self.position_count - self.last_update_position_count
        if positions_since_update >= self.config.oracle_update_frequency:
            self.inherit_tirex_confidence(positions)
        
        # Calculate confidence weights
        weights = [self.get_confidence_weight(pos) for pos in positions]
        pnls = [pos.pnl for pos in positions]
        
        if not weights or sum(weights) == 0:
            return {"error": "No valid confidence weights"}
        
        # Confidence-weighted metrics
        weighted_return = np.average(pnls, weights=weights)
        weighted_volatility = np.sqrt(np.average((np.array(pnls) - weighted_return) ** 2, weights=weights))
        
        # Confidence distribution statistics
        high_confidence_positions = [pos for pos in positions if pos.confidence >= 0.7]
        low_confidence_positions = [pos for pos in positions if pos.confidence < 0.3]
        
        high_conf_return = np.mean([pos.pnl for pos in high_confidence_positions]) if high_confidence_positions else 0.0
        low_conf_return = np.mean([pos.pnl for pos in low_confidence_positions]) if low_confidence_positions else 0.0
        
        # Oracle performance metrics
        avg_overhead_ms = np.mean(self.oracle_overhead_ms) if self.oracle_overhead_ms else 0.0
        
        return {
            "confidence_weighted_return": weighted_return,
            "confidence_weighted_volatility": weighted_volatility,
            "confidence_weighted_sharpe": weighted_return / weighted_volatility if weighted_volatility > 0 else 0.0,
            "high_confidence_return": high_conf_return,
            "low_confidence_return": low_conf_return,
            "confidence_return_spread": high_conf_return - low_conf_return,
            "confidence_distribution_bins": len(self.inherited_distribution) if self.inherited_distribution is not None else 0,
            "regime_distributions_count": len(self.regime_distributions),
            "oracle_overhead_ms": avg_overhead_ms,
            "positions_analyzed": len(positions),
            "valid_confidence_positions": len([p for p in positions if p.confidence > 0.0])
        }
    
    def get_oracle_diagnostics(self) -> Dict:
        """Get oracle diagnostic information."""
        
        return {
            "config": {
                "confidence_window_size": self.config.confidence_window_size,
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "confidence_decay_factor": self.config.confidence_decay_factor,
                "regime_confidence_mapping": self.config.regime_confidence_mapping,
                "oracle_update_frequency": self.config.oracle_update_frequency
            },
            "state": {
                "confidence_history_size": len(self.confidence_history),
                "position_count": self.position_count,
                "last_update_position_count": self.last_update_position_count,
                "has_inherited_distribution": self.inherited_distribution is not None,
                "regime_distributions_count": len(self.regime_distributions),
                "tracked_regimes": list(self.regime_distributions.keys())
            },
            "performance": {
                "avg_overhead_ms": np.mean(self.oracle_overhead_ms) if self.oracle_overhead_ms else 0.0,
                "max_overhead_ms": np.max(self.oracle_overhead_ms) if self.oracle_overhead_ms else 0.0,
                "overhead_samples": len(self.oracle_overhead_ms)
            }
        }