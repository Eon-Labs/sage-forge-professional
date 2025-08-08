#!/usr/bin/env python3
"""
Sluice 1D: Regime-Aware ODEB Weighting (RAEW)

Implements regime stability scoring and confidence-weighted ODEB analysis
using existing regime classification patterns from TiRex SAGE strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sage_forge.reporting.performance import Position, OdebResult
from sage_forge.benchmarking.confidence_inheritance_oracle import ConfidenceInheritanceOracle


class RegimeStability(Enum):
    """Regime stability classification levels."""
    VERY_HIGH = 0.9
    HIGH = 0.75
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2
    UNSTABLE = 0.1


@dataclass
class RegimeAnalysisResult:
    """Results from regime-aware ODEB analysis."""
    regime: str
    stability_score: float
    position_count: int
    confidence_weighted_return: float
    regime_efficiency_ratio: float
    regime_weight: float
    
    
class RegimeAwareOdebAnalyzer:
    """
    Implements regime-aware ODEB weighting using existing strategy patterns.
    
    Leverages TiRex SAGE strategy's _get_regime_multiplier() pattern to weight
    ODEB analysis by regime stability × confidence, preserving regime detection
    accuracy while adding sophisticated weighting capabilities.
    """
    
    def __init__(self, oracle: Optional[ConfidenceInheritanceOracle] = None):
        """Initialize regime-aware ODEB analyzer."""
        self.oracle = oracle or ConfidenceInheritanceOracle()
        
        # Regime stability mapping (follows TiRex SAGE pattern)
        self.regime_stability_map = {
            'low_vol_trending': RegimeStability.VERY_HIGH.value,    # Most stable - trending + low vol
            'medium_vol_trending': RegimeStability.HIGH.value,     # High stability - trending
            'low_vol_ranging': RegimeStability.MEDIUM.value,       # Medium - ranging but low vol
            'high_vol_trending': RegimeStability.MEDIUM.value,     # Medium - trending but high vol  
            'medium_vol_ranging': RegimeStability.LOW.value,       # Low stability - ranging + medium vol
            'high_vol_ranging': RegimeStability.VERY_LOW.value,    # Very low - ranging + high vol
            'insufficient_data': RegimeStability.UNSTABLE.value,   # Unstable - insufficient data
            'unknown': RegimeStability.UNSTABLE.value              # Default unstable
        }
        
        # Analysis cache for performance
        self.regime_analysis_cache: Dict[str, RegimeAnalysisResult] = {}
        self.cache_invalidation_threshold = 50  # Positions before cache refresh
        self.positions_since_cache_update = 0
    
    def calculate_regime_stability(self, market_regime: str) -> float:
        """
        Calculate regime stability score using existing strategy patterns.
        
        Parameters
        ----------
        market_regime : str
            Market regime classification from TiRex model
            
        Returns
        -------
        float
            Regime stability score [0, 1]
        """
        return self.regime_stability_map.get(market_regime, RegimeStability.UNSTABLE.value)
    
    def get_regime_odeb_weight(self, position: Position) -> float:
        """
        Calculate regime-aware ODEB weight: regime_stability × confidence.
        
        Parameters
        ----------
        position : Position
            Position with regime and confidence metadata
            
        Returns
        -------
        float
            Combined regime-confidence weight [0, 1]
        """
        if position.market_regime == "unknown" or position.confidence <= 0.0:
            return 0.0  # Zero weight for unknown/invalid data
        
        # Get regime stability score
        regime_stability = self.calculate_regime_stability(position.market_regime)
        
        # Get confidence weight from oracle
        confidence_weight = self.oracle.get_confidence_weight(position)
        
        # Multiplicative weighting: regime_stability × confidence
        combined_weight = regime_stability * confidence_weight
        
        # Ensure bounds [0, 1]
        return min(max(combined_weight, 0.0), 1.0)
    
    def analyze_regime_performance(self, positions: List[Position]) -> Dict[str, RegimeAnalysisResult]:
        """
        Perform regime-specific ODEB performance analysis.
        
        Parameters
        ----------
        positions : List[Position]
            Positions to analyze by regime
            
        Returns
        -------
        Dict[str, RegimeAnalysisResult]
            Performance analysis results by regime
        """
        if not positions:
            return {}
        
        # Group positions by regime
        regime_positions: Dict[str, List[Position]] = {}
        for pos in positions:
            regime = pos.market_regime if pos.market_regime != "unknown" else "unclassified"
            if regime not in regime_positions:
                regime_positions[regime] = []
            regime_positions[regime].append(pos)
        
        results = {}
        
        for regime, regime_pos_list in regime_positions.items():
            if not regime_pos_list:
                continue
                
            # Calculate regime-specific metrics
            regime_stability = self.calculate_regime_stability(regime)
            
            # Get confidence-weighted returns
            weights = [self.get_regime_odeb_weight(pos) for pos in regime_pos_list]
            returns = [pos.pnl for pos in regime_pos_list]
            
            if sum(weights) > 0:
                confidence_weighted_return = np.average(returns, weights=weights)
            else:
                confidence_weighted_return = np.mean(returns) if returns else 0.0
            
            # Calculate regime efficiency ratio
            total_regime_pnl = sum(returns)
            total_regime_size = sum(abs(pos.size_usd) for pos in regime_pos_list)
            regime_efficiency = total_regime_pnl / total_regime_size if total_regime_size > 0 else 0.0
            
            # Calculate regime weight (normalized by total positions)
            regime_weight = len(regime_pos_list) / len(positions) if positions else 0.0
            
            results[regime] = RegimeAnalysisResult(
                regime=regime,
                stability_score=regime_stability,
                position_count=len(regime_pos_list),
                confidence_weighted_return=confidence_weighted_return,
                regime_efficiency_ratio=regime_efficiency,
                regime_weight=regime_weight
            )
        
        return results
    
    def calculate_weighted_odeb_metrics(self, positions: List[Position]) -> Dict:
        """
        Calculate regime-aware weighted ODEB metrics.
        
        Parameters
        ----------
        positions : List[Position]
            Positions for ODEB analysis
            
        Returns
        -------
        Dict
            Comprehensive regime-weighted ODEB metrics
        """
        if not positions:
            return {"error": "No positions provided"}
        
        # Update oracle if needed
        self.positions_since_cache_update += len(positions)
        if self.positions_since_cache_update >= self.cache_invalidation_threshold:
            self.oracle.inherit_tirex_confidence(positions)
            self.regime_analysis_cache.clear()
            self.positions_since_cache_update = 0
        
        # Get regime-aware weights
        regime_weights = [self.get_regime_odeb_weight(pos) for pos in positions]
        confidence_weights = [self.oracle.get_confidence_weight(pos) for pos in positions]
        
        # Calculate weighted metrics
        returns = [pos.pnl for pos in positions]
        durations = [pos.duration_days for pos in positions]
        
        if sum(regime_weights) == 0:
            return {"error": "All regime weights are zero"}
        
        # Primary weighted metrics
        regime_weighted_return = np.average(returns, weights=regime_weights)
        
        # Handle zero confidence weights case
        if sum(confidence_weights) > 0:
            confidence_weighted_return = np.average(returns, weights=confidence_weights)
        else:
            confidence_weighted_return = np.mean(returns) if returns else 0.0
        
        # Weighted volatility
        regime_weighted_vol = np.sqrt(
            np.average((np.array(returns) - regime_weighted_return) ** 2, weights=regime_weights)
        )
        
        # Weighted Sharpe ratio
        regime_weighted_sharpe = (regime_weighted_return / regime_weighted_vol 
                                  if regime_weighted_vol > 0 else 0.0)
        
        # Regime-specific analysis
        regime_analysis = self.analyze_regime_performance(positions)
        
        # Best and worst performing regimes
        if regime_analysis:
            best_regime = max(regime_analysis.keys(), 
                            key=lambda r: regime_analysis[r].confidence_weighted_return)
            worst_regime = min(regime_analysis.keys(),
                             key=lambda r: regime_analysis[r].confidence_weighted_return)
        else:
            best_regime = worst_regime = "none"
        
        # Regime stability distribution
        regime_stabilities = [self.calculate_regime_stability(pos.market_regime) for pos in positions]
        avg_regime_stability = np.mean(regime_stabilities)
        
        # Efficiency comparison
        traditional_efficiency = sum(returns) / sum(abs(pos.size_usd) for pos in positions)
        regime_weighted_efficiency = regime_weighted_return / np.average(
            [abs(pos.size_usd) for pos in positions], weights=regime_weights
        ) if sum(regime_weights) > 0 else 0.0
        
        return {
            # Core weighted metrics
            "regime_weighted_return": regime_weighted_return,
            "confidence_weighted_return": confidence_weighted_return,
            "regime_weighted_volatility": regime_weighted_vol,
            "regime_weighted_sharpe": regime_weighted_sharpe,
            
            # Efficiency metrics
            "traditional_odeb_efficiency": traditional_efficiency,
            "regime_weighted_efficiency": regime_weighted_efficiency,
            "efficiency_improvement": regime_weighted_efficiency - traditional_efficiency,
            
            # Regime distribution
            "avg_regime_stability": avg_regime_stability,
            "regime_count": len(regime_analysis),
            "best_performing_regime": best_regime,
            "worst_performing_regime": worst_regime,
            
            # Weight statistics
            "avg_regime_weight": np.mean(regime_weights),
            "max_regime_weight": np.max(regime_weights),
            "min_regime_weight": np.min(regime_weights),
            "zero_weight_positions": sum(1 for w in regime_weights if w == 0),
            
            # Performance comparison
            "regime_vs_confidence_return_diff": regime_weighted_return - confidence_weighted_return,
            
            # Regime breakdown
            "regime_analysis": {
                regime: {
                    "stability_score": analysis.stability_score,
                    "position_count": analysis.position_count,
                    "weighted_return": analysis.confidence_weighted_return,
                    "efficiency_ratio": analysis.regime_efficiency_ratio,
                    "regime_weight": analysis.regime_weight
                }
                for regime, analysis in regime_analysis.items()
            },
            
            # Meta
            "total_positions": len(positions),
            "valid_regime_positions": len([p for p in positions if p.market_regime != "unknown"]),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
    
    def get_regime_recommendations(self, analysis_result: Dict) -> List[str]:
        """
        Generate regime-based trading recommendations from analysis.
        
        Parameters
        ----------
        analysis_result : Dict
            Result from calculate_weighted_odeb_metrics
            
        Returns
        -------
        List[str]
            List of actionable regime recommendations
        """
        recommendations = []
        
        if "regime_analysis" not in analysis_result:
            return ["No regime analysis available for recommendations"]
        
        regime_data = analysis_result["regime_analysis"]
        
        # Find best performing regimes
        best_regimes = sorted(regime_data.items(), 
                            key=lambda x: x[1]["weighted_return"], reverse=True)[:3]
        
        # Find most stable regimes
        stable_regimes = sorted(regime_data.items(),
                              key=lambda x: x[1]["stability_score"], reverse=True)[:3]
        
        # Generate recommendations
        if best_regimes:
            best_regime_name, best_metrics = best_regimes[0]
            recommendations.append(
                f"Focus on {best_regime_name} regime: {best_metrics['weighted_return']:.2f} weighted return, "
                f"{best_metrics['stability_score']:.1%} stability"
            )
        
        if analysis_result.get("efficiency_improvement", 0) > 0:
            recommendations.append(
                f"Regime weighting improves efficiency by {analysis_result['efficiency_improvement']:.4f}"
            )
        else:
            recommendations.append("Consider revising regime classification - weighting shows no improvement")
        
        # Stability recommendations
        avg_stability = analysis_result.get("avg_regime_stability", 0)
        if avg_stability < 0.5:
            recommendations.append("Low average regime stability - consider more conservative position sizing")
        elif avg_stability > 0.8:
            recommendations.append("High regime stability - potential for increased position sizing")
        
        # Zero weight warning
        zero_weights = analysis_result.get("zero_weight_positions", 0)
        if zero_weights > 0:
            recommendations.append(f"{zero_weights} positions have zero regime weight - review regime classification")
        
        return recommendations