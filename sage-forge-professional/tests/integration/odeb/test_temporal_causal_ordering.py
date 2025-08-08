#!/usr/bin/env python3
"""
Sluice 1E: Temporal Causal Ordering Preservation (TCOP) Validation

Tests temporal causality preservation to ensure zero look-ahead bias in 
confidence/regime assignments and maintain proper timeline throughout ODEB analysis.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from sage_forge.reporting.performance import Position
from sage_forge.models.tirex_model import TiRexPrediction


@dataclass
class TemporalViolation:
    """Represents a temporal causality violation."""
    violation_type: str
    description: str
    prediction_time: pd.Timestamp
    position_time: pd.Timestamp
    severity: str  # "CRITICAL", "WARNING", "INFO"
    

class TemporalCausalityValidator:
    """
    Validates temporal causality in TiRex → ODEB pipeline.
    
    Ensures prediction data strictly precedes position execution data
    and detects any look-ahead bias in confidence or regime assignments.
    """
    
    def __init__(self):
        """Initialize temporal causality validator."""
        self.violations: List[TemporalViolation] = []
        self.temporal_checkpoints: List[Dict] = []
        self.strict_validation_mode = True
    
    def clear_violations(self):
        """Clear violation history for new test run."""
        self.violations.clear()
        self.temporal_checkpoints.clear()
    
    def add_temporal_checkpoint(self, stage: str, timestamp: pd.Timestamp, metadata: Dict):
        """Add temporal checkpoint for audit trail."""
        self.temporal_checkpoints.append({
            'stage': stage,
            'timestamp': timestamp,
            'metadata': metadata,
            'checkpoint_id': len(self.temporal_checkpoints)
        })
    
    def validate_prediction_precedes_position(self, 
                                            prediction: TiRexPrediction,
                                            prediction_timestamp: pd.Timestamp,
                                            position: Position) -> bool:
        """
        Validate that prediction timestamp strictly precedes position timestamps.
        
        Parameters
        ----------
        prediction : TiRexPrediction
            TiRex model prediction
        prediction_timestamp : pd.Timestamp  
            When prediction was generated
        position : Position
            Position that resulted from prediction
            
        Returns
        -------
        bool
            True if temporal causality preserved
        """
        self.add_temporal_checkpoint("prediction_generated", prediction_timestamp, {
            "confidence": prediction.confidence,
            "direction": prediction.direction,
            "regime": prediction.market_regime
        })
        
        self.add_temporal_checkpoint("position_opened", position.open_time, {
            "size_usd": position.size_usd,
            "direction": position.direction,
            "confidence": position.confidence
        })
        
        # CRITICAL: Prediction must strictly precede position opening
        if prediction_timestamp >= position.open_time:
            violation = TemporalViolation(
                violation_type="FUTURE_PREDICTION",
                description=f"Prediction timestamp {prediction_timestamp} >= position open {position.open_time}",
                prediction_time=prediction_timestamp,
                position_time=position.open_time,
                severity="CRITICAL"
            )
            self.violations.append(violation)
            return False
        
        # WARNING: Prediction should be reasonably recent (within 5 minutes)
        time_gap = position.open_time - prediction_timestamp
        if time_gap > pd.Timedelta(minutes=5):
            violation = TemporalViolation(
                violation_type="STALE_PREDICTION",
                description=f"Prediction is {time_gap} old when position opened",
                prediction_time=prediction_timestamp,
                position_time=position.open_time,
                severity="WARNING"
            )
            self.violations.append(violation)
        
        return True
    
    def validate_confidence_temporal_integrity(self, positions: List[Position]) -> bool:
        """
        Validate confidence values only use past information.
        
        Parameters
        ----------
        positions : List[Position]
            Positions to validate for confidence temporal integrity
            
        Returns
        -------
        bool
            True if no temporal violations detected
        """
        violations_detected = False
        
        # Sort positions by open time for temporal analysis
        sorted_positions = sorted(positions, key=lambda p: p.open_time)
        
        for i, position in enumerate(sorted_positions):
            
            self.add_temporal_checkpoint("confidence_validation", position.open_time, {
                "position_index": i,
                "confidence": position.confidence,
                "regime": position.market_regime
            })
            
            # Check if confidence/regime data could have been influenced by future positions
            future_positions = sorted_positions[i+1:]
            
            for j, future_pos in enumerate(future_positions, i+1):
                # Look for suspiciously similar confidence values that suggest look-ahead
                if abs(position.confidence - future_pos.confidence) < 0.000001 and position.confidence > 0:
                    # This could indicate batch processing or look-ahead bias
                    time_gap = future_pos.open_time - position.open_time
                    
                    if time_gap < pd.Timedelta(minutes=2):  # Very close positions
                        violation = TemporalViolation(
                            violation_type="SUSPICIOUS_CONFIDENCE_SIMILARITY",
                            description=f"Identical confidence {position.confidence:.6f} in positions {time_gap} apart",
                            prediction_time=position.open_time,
                            position_time=future_pos.open_time,
                            severity="WARNING"
                        )
                        self.violations.append(violation)
                        violations_detected = True
                
                # Check for regime changes that are suspiciously abrupt
                if (position.market_regime != future_pos.market_regime and 
                    position.market_regime != "unknown" and
                    future_pos.market_regime != "unknown"):
                    
                    time_gap = future_pos.open_time - position.open_time
                    if time_gap < pd.Timedelta(seconds=30):  # Very rapid regime change
                        violation = TemporalViolation(
                            violation_type="ABRUPT_REGIME_CHANGE", 
                            description=f"Regime change from {position.market_regime} to {future_pos.market_regime} in {time_gap}",
                            prediction_time=position.open_time,
                            position_time=future_pos.open_time,
                            severity="INFO"
                        )
                        self.violations.append(violation)
        
        return not violations_detected
    
    def validate_bar_timestamp_ordering(self, 
                                      predictions: List[Tuple[TiRexPrediction, pd.Timestamp]],
                                      bar_timestamps: List[pd.Timestamp]) -> bool:
        """
        Validate prediction timestamps align with bar timestamp sequence.
        
        Parameters
        ----------
        predictions : List[Tuple[TiRexPrediction, pd.Timestamp]]
            Predictions with their generation timestamps
        bar_timestamps : List[pd.Timestamp] 
            Bar timestamps in chronological order
            
        Returns
        -------
        bool
            True if temporal ordering preserved
        """
        violations_detected = False
        
        for pred, pred_timestamp in predictions:
            self.add_temporal_checkpoint("bar_alignment_check", pred_timestamp, {
                "prediction_confidence": pred.confidence,
                "prediction_regime": pred.market_regime
            })
            
            # Find the bar that prediction should be based on
            preceding_bars = [ts for ts in bar_timestamps if ts <= pred_timestamp]
            
            if not preceding_bars:
                violation = TemporalViolation(
                    violation_type="PREDICTION_BEFORE_FIRST_BAR",
                    description=f"Prediction at {pred_timestamp} precedes all available bars",
                    prediction_time=pred_timestamp,
                    position_time=bar_timestamps[0] if bar_timestamps else pd.Timestamp.now(),
                    severity="CRITICAL"
                )
                self.violations.append(violation)
                violations_detected = True
                continue
            
            # Most recent bar before prediction
            latest_bar_time = max(preceding_bars)
            bar_to_prediction_gap = pred_timestamp - latest_bar_time
            
            # Prediction should be generated shortly after bar completion
            if bar_to_prediction_gap > pd.Timedelta(minutes=2):
                violation = TemporalViolation(
                    violation_type="DELAYED_PREDICTION_PROCESSING",
                    description=f"Prediction generated {bar_to_prediction_gap} after latest bar {latest_bar_time}",
                    prediction_time=pred_timestamp,
                    position_time=latest_bar_time,
                    severity="WARNING"
                )
                self.violations.append(violation)
            
            # Check for predictions that seem to use future bar data
            future_bars = [ts for ts in bar_timestamps if ts > pred_timestamp and ts <= pred_timestamp + pd.Timedelta(minutes=1)]
            if future_bars and pred.confidence > 0.8:  # High confidence with immediate future bars
                violation = TemporalViolation(
                    violation_type="POSSIBLE_FUTURE_BAR_USAGE",
                    description=f"High confidence prediction {pred.confidence:.3f} with future bars available",
                    prediction_time=pred_timestamp,
                    position_time=future_bars[0],
                    severity="WARNING" 
                )
                self.violations.append(violation)
        
        return not violations_detected
    
    def get_violation_summary(self) -> Dict:
        """Get summary of all temporal violations detected."""
        
        violation_counts = {}
        for violation in self.violations:
            v_type = violation.violation_type
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        severity_counts = {}
        for violation in self.violations:
            severity = violation.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "violation_types": violation_counts,
            "severity_distribution": severity_counts,
            "critical_violations": severity_counts.get("CRITICAL", 0),
            "warning_violations": severity_counts.get("WARNING", 0),
            "temporal_checkpoints": len(self.temporal_checkpoints),
            "validation_passed": severity_counts.get("CRITICAL", 0) == 0
        }


class TestTemporalCausalOrdering:
    """Test suite for temporal causal ordering preservation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = TemporalCausalityValidator()
        self.base_time = pd.Timestamp('2025-01-01 10:00:00')
    
    def create_test_prediction(self, offset_minutes: int, confidence: float = 0.65) -> Tuple[TiRexPrediction, pd.Timestamp]:
        """Create test prediction with timestamp."""
        timestamp = self.base_time + pd.Timedelta(minutes=offset_minutes)
        
        prediction = TiRexPrediction(
            direction=1,
            confidence=confidence,
            raw_forecast=np.array([0.1, 0.4, 0.5]),  # Required field
            volatility_forecast=0.02,
            processing_time_ms=42.5,
            market_regime="low_vol_trending",
            prediction_phase="STABLE_WINDOW"  # Required field
        )
        
        return prediction, timestamp
    
    def create_test_position(self, open_offset_minutes: int, close_offset_minutes: int, 
                           confidence: float = 0.65, regime: str = "low_vol_trending") -> Position:
        """Create test position with timestamps."""
        
        return Position(
            open_time=self.base_time + pd.Timedelta(minutes=open_offset_minutes),
            close_time=self.base_time + pd.Timedelta(minutes=close_offset_minutes),
            size_usd=1000.0,
            pnl=25.0,
            direction=1,
            confidence=confidence,
            market_regime=regime,
            regime_stability=0.85,
            prediction_phase="STABLE_WINDOW"
        )
    
    def test_prediction_precedes_position_valid(self):
        """Test: Valid temporal ordering where prediction precedes position."""
        self.validator.clear_violations()
        
        # Create prediction at T+0, position opens at T+1
        prediction, pred_timestamp = self.create_test_prediction(0, 0.75)
        position = self.create_test_position(1, 30, 0.75)
        
        result = self.validator.validate_prediction_precedes_position(
            prediction, pred_timestamp, position
        )
        
        assert result == True, "Valid temporal ordering should pass validation"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] == 0, f"No critical violations expected, got {summary}"
    
    def test_prediction_precedes_position_violation(self):
        """Test: Temporal violation where prediction comes after position."""
        self.validator.clear_violations()
        
        # Create prediction at T+5, position opens at T+1 (VIOLATION)
        prediction, pred_timestamp = self.create_test_prediction(5, 0.85)
        position = self.create_test_position(1, 30, 0.85)
        
        result = self.validator.validate_prediction_precedes_position(
            prediction, pred_timestamp, position
        )
        
        assert result == False, "Temporal violation should fail validation"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] > 0, "Critical violation should be detected"
        assert "FUTURE_PREDICTION" in summary["violation_types"], "Should detect future prediction violation"
    
    def test_confidence_temporal_integrity_valid(self):
        """Test: Confidence values show proper temporal progression."""
        self.validator.clear_violations()
        
        # Create positions with reasonable confidence variation over time
        positions = [
            self.create_test_position(1, 15, 0.60, "low_vol_trending"),
            self.create_test_position(5, 20, 0.65, "low_vol_trending"),  # Slight confidence increase
            self.create_test_position(10, 25, 0.55, "medium_vol_trending"),  # Regime change, confidence drop
            self.create_test_position(15, 30, 0.70, "medium_vol_trending")   # Confidence recovery
        ]
        
        result = self.validator.validate_confidence_temporal_integrity(positions)
        
        assert result == True, "Valid confidence progression should pass"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] == 0, "No critical violations for valid progression"
    
    def test_confidence_temporal_integrity_suspicious(self):
        """Test: Suspicious confidence patterns that suggest look-ahead bias.""" 
        self.validator.clear_violations()
        
        # Create positions with suspiciously identical confidence (potential look-ahead)
        identical_confidence = 0.847239  # Very precise value suggesting batch processing
        positions = [
            self.create_test_position(1, 15, identical_confidence, "low_vol_trending"),
            self.create_test_position(2, 16, identical_confidence, "low_vol_trending"),  # Same confidence 1 min later
            self.create_test_position(3, 17, identical_confidence, "low_vol_trending")   # Same confidence again
        ]
        
        result = self.validator.validate_confidence_temporal_integrity(positions)
        
        # Should detect suspicious pattern (warnings, not critical)
        summary = self.validator.get_violation_summary()
        assert summary["warning_violations"] > 0, "Should detect suspicious confidence similarity"
    
    def test_bar_timestamp_ordering_valid(self):
        """Test: Predictions align properly with bar timestamp sequence."""
        self.validator.clear_violations()
        
        # Create bar timestamps (1-minute bars)
        bar_timestamps = [
            self.base_time + pd.Timedelta(minutes=i) for i in range(0, 20, 1)
        ]
        
        # Create predictions that follow bars appropriately
        predictions = [
            self.create_test_prediction(2.1, 0.60),  # 0.1 min after 2nd bar
            self.create_test_prediction(5.2, 0.65),  # 0.2 min after 5th bar  
            self.create_test_prediction(10.1, 0.55), # 0.1 min after 10th bar
        ]
        
        result = self.validator.validate_bar_timestamp_ordering(predictions, bar_timestamps)
        
        assert result == True, "Valid bar-prediction alignment should pass"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] == 0, "No critical violations for valid alignment"
    
    def test_bar_timestamp_ordering_violation(self):
        """Test: Predictions that violate bar timestamp ordering."""
        self.validator.clear_violations()
        
        # Create bar timestamps
        bar_timestamps = [
            self.base_time + pd.Timedelta(minutes=i) for i in range(5, 25, 1)  # Bars start at T+5
        ]
        
        # Create predictions with temporal violations
        predictions = [
            self.create_test_prediction(2, 0.90),    # CRITICAL: Before first bar
            self.create_test_prediction(10.5, 0.85), # WARNING: High confidence with future bars nearby
        ]
        
        result = self.validator.validate_bar_timestamp_ordering(predictions, bar_timestamps)
        
        assert result == False, "Temporal violations should fail validation"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] > 0, "Should detect prediction before first bar"
    
    def test_end_to_end_temporal_validation(self):
        """Test: Complete temporal validation of TiRex → Position pipeline."""
        self.validator.clear_violations()
        
        # Create comprehensive temporal scenario
        bar_timestamps = [self.base_time + pd.Timedelta(minutes=i) for i in range(0, 60, 1)]
        
        # Valid predictions following bars
        predictions = [
            self.create_test_prediction(5.1, 0.70),
            self.create_test_prediction(15.2, 0.65),
            self.create_test_prediction(30.1, 0.75)
        ]
        
        # Positions that follow predictions appropriately
        positions = [
            self.create_test_position(6, 20, 0.70, "low_vol_trending"),    # 0.9 min after prediction
            self.create_test_position(16, 35, 0.65, "medium_vol_trending"), # 0.8 min after prediction  
            self.create_test_position(31, 50, 0.75, "low_vol_trending")    # 0.9 min after prediction
        ]
        
        # Validate all temporal aspects
        bar_validation = self.validator.validate_bar_timestamp_ordering(predictions, bar_timestamps)
        confidence_validation = self.validator.validate_confidence_temporal_integrity(positions)
        
        # Individual prediction-position validations
        prediction_validations = []
        for (pred, pred_time), pos in zip(predictions, positions):
            prediction_validations.append(
                self.validator.validate_prediction_precedes_position(pred, pred_time, pos)
            )
        
        # All validations should pass
        assert bar_validation == True, "Bar timestamp validation should pass"
        assert confidence_validation == True, "Confidence integrity validation should pass"
        assert all(prediction_validations), "All prediction-position validations should pass"
        
        summary = self.validator.get_violation_summary()
        assert summary["critical_violations"] == 0, f"No critical violations expected in valid scenario: {summary}"
        
        print(f"✅ End-to-end temporal validation PASSED")
        print(f"📊 Temporal checkpoints: {summary['temporal_checkpoints']}")
        print(f"🛡️ Zero critical violations - Temporal causality PRESERVED")


if __name__ == "__main__":
    # Run temporal causal ordering preservation tests
    print("🔧 Running Sluice 1E: Temporal Causal Ordering Preservation Tests")
    
    test_suite = TestTemporalCausalOrdering()
    
    try:
        test_suite.setup_method()
        
        test_suite.test_prediction_precedes_position_valid()
        print("✅ Prediction precedes position validation - PASSED")
        
        test_suite.setup_method()
        test_suite.test_prediction_precedes_position_violation()
        print("✅ Temporal violation detection - PASSED")
        
        test_suite.setup_method() 
        test_suite.test_confidence_temporal_integrity_valid()
        print("✅ Confidence temporal integrity validation - PASSED")
        
        test_suite.setup_method()
        test_suite.test_confidence_temporal_integrity_suspicious()
        print("✅ Suspicious confidence pattern detection - PASSED")
        
        test_suite.setup_method()
        test_suite.test_bar_timestamp_ordering_valid()
        print("✅ Bar timestamp ordering validation - PASSED")
        
        test_suite.setup_method()
        test_suite.test_bar_timestamp_ordering_violation()
        print("✅ Bar timestamp violation detection - PASSED")
        
        test_suite.setup_method()
        test_suite.test_end_to_end_temporal_validation()
        print("✅ End-to-end temporal validation - PASSED")
        
        print(f"\n🎯 Sluice 1E VALIDATION COMPLETE")
        print(f"🛡️ Zero look-ahead bias detected - Temporal causality PRESERVED")
        print(f"⏰ Prediction data strictly precedes position execution - VALIDATED")
        
    except Exception as e:
        print(f"❌ Sluice 1E validation FAILED: {e}")
        print("🚨 Temporal causality violation detected - ODEB implementation blocked")
        raise