#!/usr/bin/env python3
"""
Sluice 1F: Context Boundary Phase Management (CBPP)

Manages prediction phases for accurate ODEB analysis with TiRex's 1-768+ bar
prediction capability, implementing NT-compatible state management patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sage_forge.reporting.performance import Position


class PredictionPhase(Enum):
    """TiRex prediction phases following NT enum patterns."""
    WARM_UP_PERIOD = "WARM_UP_PERIOD"
    CONTEXT_BOUNDARY = "CONTEXT_BOUNDARY"  
    STABLE_WINDOW = "STABLE_WINDOW"
    

@dataclass
class PhaseTransition:
    """Records phase transition events."""
    from_phase: PredictionPhase
    to_phase: PredictionPhase
    transition_time: pd.Timestamp
    trigger_reason: str
    confidence_at_transition: float
    positions_processed: int


@dataclass
class PhaseAnalysisResult:
    """Results from phase-aware ODEB analysis."""
    phase: PredictionPhase
    position_count: int
    avg_confidence: float
    confidence_weighted_return: float
    phase_duration_minutes: float
    reliability_score: float  # >95% for STABLE_WINDOW requirement


class ContextBoundaryPhaseManager:
    """
    Manages TiRex prediction phases for accurate ODEB analysis.
    
    Implements phase detection and transitions (WARM_UP → CONTEXT_BOUNDARY → STABLE_WINDOW)
    with appropriate confidence scaling and NT-compatible state management.
    """
    
    def __init__(self, 
                 warm_up_bars: int = 50,
                 context_boundary_bars: int = 100, 
                 stable_window_threshold: float = 0.95):
        """
        Initialize Context Boundary Phase Manager.
        
        Parameters
        ----------
        warm_up_bars : int, default 50
            Number of bars for warm-up phase
        context_boundary_bars : int, default 100
            Number of bars for context boundary establishment
        stable_window_threshold : float, default 0.95
            Reliability threshold for stable window (>95% requirement)
        """
        self.warm_up_bars = warm_up_bars
        self.context_boundary_bars = context_boundary_bars
        self.stable_window_threshold = stable_window_threshold
        
        # State management (NT-compatible pattern)
        self.current_phase = PredictionPhase.WARM_UP_PERIOD
        self.phase_start_time: Optional[pd.Timestamp] = None
        self.bars_processed = 0
        self.positions_in_current_phase = 0
        
        # Phase history and transitions
        self.phase_history: List[PhaseTransition] = []
        self.phase_statistics: Dict[PredictionPhase, Dict] = {}
        
        # Reliability tracking for STABLE_WINDOW validation
        self.reliability_window: List[float] = []  # Recent prediction accuracies
        self.reliability_window_size = 20
        
        # Performance tracking
        self.phase_positions: Dict[PredictionPhase, List[Position]] = {
            PredictionPhase.WARM_UP_PERIOD: [],
            PredictionPhase.CONTEXT_BOUNDARY: [],
            PredictionPhase.STABLE_WINDOW: []
        }
    
    def process_new_bar(self, bar_timestamp: pd.Timestamp, prediction_confidence: float = None):
        """
        Process new bar and manage phase transitions.
        
        Parameters
        ----------
        bar_timestamp : pd.Timestamp
            Timestamp of new bar
        prediction_confidence : float, optional
            TiRex prediction confidence for this bar
        """
        self.bars_processed += 1
        
        # Initialize phase timing on first bar
        if self.phase_start_time is None:
            self.phase_start_time = bar_timestamp
        
        # Update reliability tracking if confidence provided
        if prediction_confidence is not None:
            # Convert confidence to reliability score (simplified)
            reliability = prediction_confidence if prediction_confidence > 0.5 else 0.0
            self.reliability_window.append(reliability)
            
            # Maintain rolling window
            if len(self.reliability_window) > self.reliability_window_size:
                self.reliability_window.pop(0)
        
        # Check for phase transitions
        self._check_phase_transitions(bar_timestamp, prediction_confidence or 0.0)
    
    def _check_phase_transitions(self, current_time: pd.Timestamp, confidence: float):
        """Check and execute phase transitions based on current state."""
        
        old_phase = self.current_phase
        transition_triggered = False
        trigger_reason = ""
        
        if self.current_phase == PredictionPhase.WARM_UP_PERIOD:
            # Transition to CONTEXT_BOUNDARY after warm-up bars
            if self.bars_processed >= self.warm_up_bars:
                self.current_phase = PredictionPhase.CONTEXT_BOUNDARY
                transition_triggered = True
                trigger_reason = f"Completed warm-up period ({self.warm_up_bars} bars)"
        
        elif self.current_phase == PredictionPhase.CONTEXT_BOUNDARY:
            # Transition to STABLE_WINDOW after context establishment
            total_bars_needed = self.warm_up_bars + self.context_boundary_bars
            if self.bars_processed >= total_bars_needed:
                
                # Check if reliability threshold is met
                current_reliability = self._calculate_current_reliability()
                if current_reliability >= self.stable_window_threshold:
                    self.current_phase = PredictionPhase.STABLE_WINDOW
                    transition_triggered = True
                    trigger_reason = f"Context established and reliability {current_reliability:.3f} >= {self.stable_window_threshold}"
                else:
                    # Stay in CONTEXT_BOUNDARY until reliability improves
                    trigger_reason = f"Reliability {current_reliability:.3f} below threshold {self.stable_window_threshold}"
        
        # Record transition if occurred
        if transition_triggered:
            transition = PhaseTransition(
                from_phase=old_phase,
                to_phase=self.current_phase,
                transition_time=current_time,
                trigger_reason=trigger_reason,
                confidence_at_transition=confidence,
                positions_processed=self.positions_in_current_phase
            )
            
            self.phase_history.append(transition)
            
            # Reset phase counters
            self.phase_start_time = current_time
            self.positions_in_current_phase = 0
    
    def _calculate_current_reliability(self) -> float:
        """Calculate current prediction reliability score."""
        if not self.reliability_window:
            return 0.0
        
        # Calculate mean reliability with emphasis on recent values
        weights = np.exp(np.linspace(-1, 0, len(self.reliability_window)))  # Exponential weighting
        weighted_reliability = np.average(self.reliability_window, weights=weights)
        
        return min(weighted_reliability, 1.0)
    
    def assign_position_phase(self, position: Position) -> Position:
        """
        Assign current prediction phase to position.
        
        Parameters
        ----------
        position : Position
            Position to assign phase to
            
        Returns
        -------
        Position
            Position with prediction_phase field updated
        """
        # Update position phase
        position.prediction_phase = self.current_phase.value
        
        # Track position in current phase
        self.phase_positions[self.current_phase].append(position)
        self.positions_in_current_phase += 1
        
        return position
    
    def get_phase_confidence_scaling(self, base_confidence: float) -> float:
        """
        Apply phase-appropriate confidence scaling.
        
        Parameters
        ----------
        base_confidence : float
            Raw TiRex confidence [0, 1]
            
        Returns
        -------
        float
            Phase-scaled confidence [0, 1]
        """
        if self.current_phase == PredictionPhase.WARM_UP_PERIOD:
            # Reduce confidence during warm-up (model still learning)
            scaling_factor = 0.7
            
        elif self.current_phase == PredictionPhase.CONTEXT_BOUNDARY:
            # Moderate confidence scaling (context being established)
            scaling_factor = 0.85
            
        elif self.current_phase == PredictionPhase.STABLE_WINDOW:
            # Full confidence (model has established stable context)
            current_reliability = self._calculate_current_reliability()
            if current_reliability >= self.stable_window_threshold:
                scaling_factor = 1.0  # No scaling in stable window
            else:
                scaling_factor = 0.9  # Slight reduction if reliability dropping
        else:
            scaling_factor = 0.5  # Conservative default
        
        scaled_confidence = base_confidence * scaling_factor
        return min(max(scaled_confidence, 0.0), 1.0)  # Ensure [0, 1] bounds
    
    def analyze_phase_performance(self) -> Dict[PredictionPhase, PhaseAnalysisResult]:
        """
        Analyze ODEB performance by prediction phase.
        
        Returns
        -------
        Dict[PredictionPhase, PhaseAnalysisResult]
            Performance analysis for each phase
        """
        results = {}
        
        for phase, positions in self.phase_positions.items():
            if not positions:
                continue
            
            # Calculate phase metrics
            confidences = [pos.confidence for pos in positions if pos.confidence > 0]
            returns = [pos.pnl for pos in positions]
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Confidence-weighted return
            if confidences and len(confidences) == len(returns):
                confidence_weighted_return = np.average(returns, weights=confidences)
            else:
                confidence_weighted_return = np.mean(returns) if returns else 0.0
            
            # Phase duration (approximate from positions)
            if len(positions) > 1:
                phase_duration = (positions[-1].open_time - positions[0].open_time).total_seconds() / 60
            else:
                phase_duration = 0.0
            
            # Reliability score calculation
            if phase == PredictionPhase.STABLE_WINDOW:
                reliability_score = self._calculate_current_reliability()
            else:
                # For non-stable phases, base on confidence distribution
                reliability_score = avg_confidence * 0.8  # Conservative estimate
            
            results[phase] = PhaseAnalysisResult(
                phase=phase,
                position_count=len(positions),
                avg_confidence=avg_confidence,
                confidence_weighted_return=confidence_weighted_return,
                phase_duration_minutes=phase_duration,
                reliability_score=reliability_score
            )
        
        return results
    
    def get_stable_window_validation(self) -> Dict:
        """
        Validate STABLE_WINDOW phase meets >95% reliability requirement.
        
        Returns
        -------
        Dict
            Validation results for stable window phase
        """
        current_reliability = self._calculate_current_reliability()
        stable_positions = self.phase_positions[PredictionPhase.STABLE_WINDOW]
        
        # Check if currently in stable window
        in_stable_window = self.current_phase == PredictionPhase.STABLE_WINDOW
        
        # Validate reliability requirement
        meets_reliability_requirement = current_reliability >= self.stable_window_threshold
        
        # Calculate stable window statistics
        if stable_positions:
            stable_confidences = [pos.confidence for pos in stable_positions if pos.confidence > 0]
            stable_returns = [pos.pnl for pos in stable_positions]
            
            avg_stable_confidence = np.mean(stable_confidences) if stable_confidences else 0.0
            avg_stable_return = np.mean(stable_returns) if stable_returns else 0.0
        else:
            avg_stable_confidence = 0.0
            avg_stable_return = 0.0
        
        return {
            "current_phase": self.current_phase.value,
            "in_stable_window": in_stable_window,
            "current_reliability": current_reliability,
            "reliability_threshold": self.stable_window_threshold,
            "meets_reliability_requirement": meets_reliability_requirement,
            "stable_window_positions": len(stable_positions),
            "avg_stable_confidence": avg_stable_confidence,
            "avg_stable_return": avg_stable_return,
            "bars_processed": self.bars_processed,
            "phase_transitions": len(self.phase_history),
            "reliability_samples": len(self.reliability_window),
            "validation_passed": meets_reliability_requirement and in_stable_window
        }
    
    def get_phase_diagnostics(self) -> Dict:
        """Get comprehensive phase manager diagnostics."""
        
        phase_analysis = self.analyze_phase_performance()
        stable_validation = self.get_stable_window_validation()
        
        return {
            "current_state": {
                "current_phase": self.current_phase.value,
                "phase_start_time": self.phase_start_time.isoformat() if self.phase_start_time else None,
                "bars_processed": self.bars_processed,
                "positions_in_current_phase": self.positions_in_current_phase
            },
            "configuration": {
                "warm_up_bars": self.warm_up_bars,
                "context_boundary_bars": self.context_boundary_bars,
                "stable_window_threshold": self.stable_window_threshold,
                "reliability_window_size": self.reliability_window_size
            },
            "phase_transitions": [
                {
                    "from": t.from_phase.value,
                    "to": t.to_phase.value,
                    "time": t.transition_time.isoformat(),
                    "reason": t.trigger_reason,
                    "confidence": t.confidence_at_transition
                }
                for t in self.phase_history
            ],
            "phase_analysis": {
                phase.value: {
                    "position_count": result.position_count,
                    "avg_confidence": result.avg_confidence,
                    "weighted_return": result.confidence_weighted_return,
                    "duration_minutes": result.phase_duration_minutes,
                    "reliability_score": result.reliability_score
                }
                for phase, result in phase_analysis.items()
            },
            "stable_window_validation": stable_validation,
            "mhtfc_integration": {
                "max_prediction_horizon": 768,  # TiRex capability
                "context_establishment_bars": self.warm_up_bars + self.context_boundary_bars,
                "multi_horizon_ready": stable_validation["validation_passed"]
            }
        }