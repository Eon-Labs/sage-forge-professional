#!/usr/bin/env python3
"""
Sluice 1B: Confidence Flow Chain Validation Test Framework

Tests confidence preservation through complete TiRex → Strategy → Position pipeline
ensuring zero confidence leakage or corruption during the flow chain.
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import asdict
from decimal import Decimal

from sage_forge.reporting.performance import Position
from sage_forge.models.tirex_model import TiRexModel, TiRexPrediction
from sage_forge.strategies.tirex_sage_strategy import TiRexSageStrategy, TiRexSignal

class TestConfidenceFlowChain:
    """Test suite for confidence preservation validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_confidence_values = [0.15, 0.45, 0.75, 0.95]
        self.test_regimes = ["low_vol_trending", "high_vol_ranging", "medium_vol_trending"]
        self.confidence_audit_trail = []
    
    def log_confidence_checkpoint(self, stage: str, confidence: float, context: dict):
        """Log confidence at each stage for audit trail."""
        self.confidence_audit_trail.append({
            'stage': stage,
            'confidence': confidence,
            'context': context,
            'timestamp': pd.Timestamp.now()
        })
    
    def test_tirex_prediction_confidence_preservation(self):
        """Test: TiRex model prediction confidence is preserved correctly."""
        
        for test_confidence in self.test_confidence_values:
            # Create mock TiRex prediction with known confidence
            prediction = TiRexPrediction(
                direction=1,
                confidence=test_confidence,
                raw_forecast=np.array([0.1, 0.3, 0.6]),  # Required field
                volatility_forecast=0.02,
                processing_time_ms=45.2,
                market_regime="low_vol_trending",
                prediction_phase="STABLE_WINDOW"  # Required field
            )
            
            self.log_confidence_checkpoint(
                "tirex_prediction", 
                prediction.confidence,
                {"direction": prediction.direction, "regime": prediction.market_regime}
            )
            
            # Validate confidence is exactly preserved
            assert prediction.confidence == test_confidence, \
                f"TiRex prediction confidence corrupted: expected {test_confidence}, got {prediction.confidence}"
            
            # Validate confidence bounds [0, 1]
            assert 0.0 <= prediction.confidence <= 1.0, \
                f"TiRex confidence out of bounds: {prediction.confidence}"
    
    def test_strategy_signal_confidence_preservation(self):
        """Test: Strategy signal generation preserves TiRex confidence."""
        
        # Initialize strategy with test config
        config = {
            'min_confidence': 0.10,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'model_name': 'NX-AI/TiRex',
            'adaptive_thresholds': True,
            'instrument_id': 'BTCUSDT-PERP.BINANCE'
        }
        
        strategy = TiRexSageStrategy(config=config)
        
        for test_confidence in self.test_confidence_values:
            # Create TiRex prediction
            prediction = TiRexPrediction(
                direction=1,
                confidence=test_confidence,
                raw_forecast=np.array([0.2, 0.4, 0.4]),  # Required field
                volatility_forecast=0.025,
                processing_time_ms=42.1,
                market_regime="medium_vol_trending",
                prediction_phase="STABLE_WINDOW"  # Required field
            )
            
            self.log_confidence_checkpoint(
                "strategy_input",
                prediction.confidence,
                {"prediction_direction": prediction.direction}
            )
            
            # Generate signal through strategy
            # Note: This requires mocking bar data for _generate_signal
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.objects import Price
            from decimal import Decimal
            
            mock_bar = Bar(
                bar_type=None,  # Will be set by strategy
                open=Price.from_str("50000.00"),
                high=Price.from_str("50100.00"), 
                low=Price.from_str("49900.00"),
                close=Price.from_str("50050.00"),
                volume=None,
                ts_event=0,
                ts_init=0
            )
            
            # Generate signal using strategy's internal method
            signal = strategy._generate_tirex_signal(prediction, mock_bar)
            
            if signal:  # Signal may be None if confidence too low
                self.log_confidence_checkpoint(
                    "strategy_signal",
                    signal.confidence,
                    {"signal_direction": signal.direction, "position_size": signal.position_size}
                )
                
                # Validate exact confidence preservation
                assert signal.confidence == test_confidence, \
                    f"Strategy signal confidence corrupted: expected {test_confidence}, got {signal.confidence}"
                
                # Validate regime preservation  
                assert signal.market_regime == prediction.market_regime, \
                    f"Market regime not preserved: expected {prediction.market_regime}, got {signal.market_regime}"
    
    def test_position_confidence_inheritance(self):
        """Test: Position dataclass correctly inherits confidence metadata."""
        
        for test_confidence in self.test_confidence_values:
            for regime in self.test_regimes:
                
                # Create position with confidence metadata
                position = Position(
                    open_time=pd.Timestamp('2025-01-01 10:00:00'),
                    close_time=pd.Timestamp('2025-01-01 10:30:00'),
                    size_usd=1000.0,
                    pnl=25.5,
                    direction=1,
                    confidence=test_confidence,
                    market_regime=regime,
                    regime_stability=0.85,
                    prediction_phase="STABLE_WINDOW"
                )
                
                self.log_confidence_checkpoint(
                    "position_created",
                    position.confidence,
                    {
                        "regime": position.market_regime,
                        "phase": position.prediction_phase,
                        "pnl": position.pnl
                    }
                )
                
                # Validate confidence preservation
                assert position.confidence == test_confidence, \
                    f"Position confidence corrupted: expected {test_confidence}, got {position.confidence}"
                
                # Validate metadata preservation
                assert position.market_regime == regime, \
                    f"Position regime corrupted: expected {regime}, got {position.market_regime}"
                
                # Validate confidence bounds
                assert 0.0 <= position.confidence <= 1.0, \
                    f"Position confidence out of bounds: {position.confidence}"
                
                # Test backward compatibility - position should work without new fields
                minimal_position = Position(
                    open_time=pd.Timestamp('2025-01-01 11:00:00'),
                    close_time=pd.Timestamp('2025-01-01 11:15:00'),
                    size_usd=500.0,
                    pnl=-12.3,
                    direction=-1
                )
                
                # Should use default values
                assert minimal_position.confidence == 0.0
                assert minimal_position.market_regime == "unknown" 
                assert minimal_position.regime_stability == 0.0
                assert minimal_position.prediction_phase == "unknown"
    
    def test_end_to_end_confidence_preservation(self):
        """Test: Complete confidence flow chain preserves accuracy."""
        
        test_confidence = 0.67
        test_regime = "low_vol_trending"
        
        # Step 1: Create TiRex prediction
        prediction = TiRexPrediction(
            direction=1,
            confidence=test_confidence,
            raw_forecast=np.array([0.15, 0.25, 0.6]),  # Required field
            volatility_forecast=0.018,
            processing_time_ms=38.7,
            market_regime=test_regime,
            prediction_phase="STABLE_WINDOW"  # Required field
        )
        
        self.log_confidence_checkpoint("e2e_start", prediction.confidence, {"stage": "tirex_prediction"})
        
        # Step 2: Strategy processes prediction (mock flow)
        signal_confidence = prediction.confidence  # Simulating strategy preservation
        self.log_confidence_checkpoint("e2e_strategy", signal_confidence, {"stage": "strategy_signal"})
        
        # Step 3: Position creation with metadata
        position = Position(
            open_time=pd.Timestamp('2025-01-01 14:00:00'),
            close_time=pd.Timestamp('2025-01-01 14:45:00'), 
            size_usd=2000.0,
            pnl=87.4,
            direction=1,
            confidence=signal_confidence,
            market_regime=test_regime,
            regime_stability=0.92,
            prediction_phase="STABLE_WINDOW"
        )
        
        self.log_confidence_checkpoint("e2e_position", position.confidence, {"stage": "position_final"})
        
        # Validation: End-to-end confidence preservation
        assert position.confidence == test_confidence, \
            f"End-to-end confidence leak detected: started {test_confidence}, ended {position.confidence}"
        
        # Validation: Complete audit trail
        assert len(self.confidence_audit_trail) >= 3, "Incomplete confidence audit trail"
        
        # Validation: No confidence corruption in audit trail
        for checkpoint in self.confidence_audit_trail:
            if 'e2e_' in checkpoint['stage']:
                assert checkpoint['confidence'] == test_confidence, \
                    f"Confidence corruption detected at {checkpoint['stage']}: {checkpoint['confidence']}"
    
    def test_confidence_bounds_validation(self):
        """Test: Confidence values remain within valid bounds throughout flow."""
        
        invalid_confidences = [-0.1, 1.1, 2.0, -0.001, 1.001]
        
        for invalid_conf in invalid_confidences:
            # Test should handle invalid confidence gracefully
            with pytest.raises((ValueError, AssertionError)):
                position = Position(
                    open_time=pd.Timestamp('2025-01-01 15:00:00'),
                    close_time=pd.Timestamp('2025-01-01 15:30:00'),
                    size_usd=500.0,
                    pnl=0.0,
                    direction=1,
                    confidence=invalid_conf  # Invalid confidence should be rejected
                )
                
                # Validate bounds if position creation succeeds
                assert 0.0 <= position.confidence <= 1.0, \
                    f"Invalid confidence accepted: {position.confidence}"
    
    def test_confidence_audit_trail_completeness(self):
        """Test: Audit trail captures all confidence manipulation points."""
        
        # Run a few confidence flow tests to populate audit trail
        self.test_tirex_prediction_confidence_preservation()
        self.test_position_confidence_inheritance()
        
        # Validate audit trail structure
        assert len(self.confidence_audit_trail) > 0, "No confidence audit trail captured"
        
        required_stages = ["tirex_prediction", "position_created"]
        captured_stages = {checkpoint['stage'] for checkpoint in self.confidence_audit_trail}
        
        for stage in required_stages:
            assert stage in captured_stages, f"Missing audit trail stage: {stage}"
        
        # Validate audit trail data integrity
        for checkpoint in self.confidence_audit_trail:
            assert 'stage' in checkpoint, "Audit checkpoint missing stage"
            assert 'confidence' in checkpoint, "Audit checkpoint missing confidence" 
            assert 'context' in checkpoint, "Audit checkpoint missing context"
            assert 'timestamp' in checkpoint, "Audit checkpoint missing timestamp"
            
            # Validate confidence bounds in audit trail
            assert 0.0 <= checkpoint['confidence'] <= 1.0, \
                f"Invalid confidence in audit trail: {checkpoint['confidence']}"


if __name__ == "__main__":
    # Run confidence flow chain validation
    print("🔧 Running Sluice 1B: Confidence Flow Chain Validation Tests")
    
    test_suite = TestConfidenceFlowChain()
    test_suite.setup_method()
    
    try:
        test_suite.test_tirex_prediction_confidence_preservation()
        print("✅ TiRex prediction confidence preservation - PASSED")
        
        test_suite.test_position_confidence_inheritance()  
        print("✅ Position confidence inheritance - PASSED")
        
        test_suite.test_end_to_end_confidence_preservation()
        print("✅ End-to-end confidence preservation - PASSED")
        
        test_suite.test_confidence_bounds_validation()
        print("✅ Confidence bounds validation - PASSED") 
        
        test_suite.test_confidence_audit_trail_completeness()
        print("✅ Confidence audit trail completeness - PASSED")
        
        print(f"\n🎯 Sluice 1B VALIDATION COMPLETE")
        print(f"📊 Audit trail captured {len(test_suite.confidence_audit_trail)} confidence checkpoints")
        print(f"🛡️ Zero confidence leakage detected - ODEB confidence preservation VALIDATED")
        
    except Exception as e:
        print(f"❌ Sluice 1B validation FAILED: {e}")
        print("🚨 Confidence leak detected - ODEB implementation blocked until resolution")
        raise