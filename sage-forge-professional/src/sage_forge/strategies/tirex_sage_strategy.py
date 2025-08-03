#!/usr/bin/env python3
"""
TiRex SAGE Strategy - NX-AI TiRex Integration with SAGE-Forge Framework
Real-time directional trading strategy using TiRex 35M parameter model for GPU-accelerated forecasting.

SAGE Methodology: Self-Adaptive Generative Evaluation with quantitative assessment
through parameter-free, regime-aware evaluation using the NX-AI TiRex model.
"""

import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, PositionSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.instruments import CryptoFuture, CryptoPerpetual
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.objects import Quantity, Price
from rich.console import Console

from sage_forge.core.config import get_config
from sage_forge.risk.position_sizer import RealisticPositionSizer
from sage_forge.models.tirex_model import TiRexModel, TiRexPrediction

console = Console()


@dataclass
class TiRexSignal:
    """TiRex trading signal with risk management parameters."""
    direction: int  # -1: sell, 0: hold, 1: buy
    confidence: float  # Signal confidence [0, 1]
    position_size: float  # Recommended position size
    stop_loss: Optional[float]  # Stop loss price
    take_profit: Optional[float]  # Take profit price
    market_regime: str  # Current market regime
    processing_time_ms: float  # Model inference time


class TiRexSageStrategy(Strategy):
    """
    TiRex SAGE Strategy - Real-time directional trading with NX-AI TiRex model.
    
    Features:
    - GPU-accelerated TiRex model inference (35M parameters)
    - Adaptive position sizing based on model confidence
    - Market regime-aware risk management
    - Parameter-free signal generation
    - Real-time performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize TiRex SAGE Strategy."""
        
        # Load configuration
        strategy_config = config or get_config().get('tirex_strategy', {})
        
        # Strategy parameters
        self.min_confidence = strategy_config.get('min_confidence', 0.6)
        self.max_position_size = strategy_config.get('max_position_size', 0.1)  # 10% of account
        self.risk_per_trade = strategy_config.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.model_name = strategy_config.get('model_name', 'NX-AI/TiRex')  # HuggingFace model ID
        
        # Initialize components
        self.tirex_model = None
        self.position_sizer = None
        self.instrument_id = None
        
        # Performance tracking
        self.total_predictions = 0
        self.successful_predictions = 0
        self.total_pnl = 0.0
        self.trade_history = []
        
        # Current state
        self.last_prediction = None
        self.current_position_side = PositionSide.FLAT
        
        super().__init__()
    
    def on_start(self):
        """Strategy startup initialization."""
        console.print("🚀 Starting TiRex SAGE Strategy")
        
        # Initialize TiRex model
        try:
            # Use real TiRex model from HuggingFace
            self.tirex_model = TiRexModel(model_name=self.model_name, prediction_length=1)
            if not self.tirex_model.is_loaded:
                self.log.error("Failed to load TiRex model")
                return
            
            console.print("✅ Real TiRex 35M parameter model loaded successfully")
            
        except Exception as e:
            self.log.error(f"TiRex model initialization failed: {e}")
            return
        
        # Initialize position sizer
        self.position_sizer = RealisticPositionSizer(
            max_position_size=self.max_position_size,
            risk_per_trade=self.risk_per_trade
        )
        
        # Subscribe to data
        for instrument_id in self.cache.instrument_ids():
            self.subscribe_bars(
                bar_type=f"{instrument_id}-1-MINUTE-LAST-INTERNAL",
                await_partial=True
            )
            self.instrument_id = instrument_id  # Store for trading
            break  # Use first instrument for now
        
        console.print(f"✅ Subscribed to {self.instrument_id}")
        
        self.log.info("TiRex SAGE Strategy started successfully")
    
    def on_stop(self):
        """Strategy shutdown."""
        console.print("🛑 Stopping TiRex SAGE Strategy")
        
        # Performance summary
        if self.total_predictions > 0:
            accuracy = self.successful_predictions / self.total_predictions
            console.print(f"📊 Strategy Performance:")
            console.print(f"   Total Predictions: {self.total_predictions}")
            console.print(f"   Accuracy: {accuracy:.2%}")
            console.print(f"   Total PnL: ${self.total_pnl:.2f}")
            
            if self.tirex_model:
                model_stats = self.tirex_model.get_performance_stats()
                console.print(f"   Avg Inference Time: {model_stats.get('avg_inference_time_ms', 0):.1f}ms")
        
        self.log.info("TiRex SAGE Strategy stopped")
    
    def on_bar(self, bar: Bar):
        """Process new market data bar."""
        if not self.tirex_model or not self.tirex_model.is_loaded:
            return
        
        try:
            # Add bar to TiRex model
            self.tirex_model.add_bar(bar)
            
            # Generate prediction
            prediction = self.tirex_model.predict()
            if prediction is None:
                return
            
            self.total_predictions += 1
            self.last_prediction = prediction
            
            # Generate trading signal
            signal = self._generate_trading_signal(prediction, bar)
            
            if signal and signal.confidence >= self.min_confidence:
                self._execute_signal(signal, bar)
            
            # Log performance
            if self.total_predictions % 100 == 0:
                self._log_performance()
        
        except Exception as e:
            self.log.error(f"Error processing bar: {e}")
    
    def _generate_trading_signal(self, prediction: TiRexPrediction, bar: Bar) -> Optional[TiRexSignal]:
        """Convert TiRex prediction to trading signal."""
        
        if prediction.confidence < self.min_confidence:
            return None
        
        # Get current price
        current_price = float(bar.close)
        
        # Calculate position size based on confidence and volatility
        base_size = self.position_sizer.calculate_position_size(
            account_balance=float(self.cache.account_balance_total()),
            risk_amount=current_price * self.risk_per_trade,
            entry_price=current_price,
            confidence=prediction.confidence
        )
        
        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier(prediction.market_regime)
        position_size = base_size * regime_multiplier
        
        # Calculate stop loss and take profit
        volatility = prediction.volatility_forecast
        
        if prediction.direction == 1:  # Bullish
            stop_loss = current_price * (1 - volatility * 2)
            take_profit = current_price * (1 + volatility * 3)
        elif prediction.direction == -1:  # Bearish
            stop_loss = current_price * (1 + volatility * 2)
            take_profit = current_price * (1 - volatility * 3)
        else:
            return None  # No signal
        
        return TiRexSignal(
            direction=prediction.direction,
            confidence=prediction.confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_regime=prediction.market_regime,
            processing_time_ms=prediction.processing_time_ms
        )
    
    def _get_regime_multiplier(self, market_regime: str) -> float:
        """Get position size multiplier based on market regime."""
        regime_multipliers = {
            'low_vol_trending': 1.2,      # High confidence in trending low vol
            'low_vol_ranging': 0.8,       # Reduced size in ranging markets
            'medium_vol_trending': 1.0,   # Normal size
            'medium_vol_ranging': 0.7,    # Further reduced in ranging
            'high_vol_trending': 0.9,     # Slightly reduced due to high vol
            'high_vol_ranging': 0.5,      # Minimal size in high vol ranging
            'insufficient_data': 0.3      # Very conservative with limited data
        }
        
        return regime_multipliers.get(market_regime, 0.8)  # Default conservative
    
    def _execute_signal(self, signal: TiRexSignal, bar: Bar):
        """Execute trading signal."""
        
        # Check if we need to close existing position
        current_position = self.cache.position(self.instrument_id)
        
        if current_position is not None and not current_position.is_closed:
            # Check if signal direction conflicts with current position
            current_side = current_position.side
            
            if (signal.direction == 1 and current_side == PositionSide.SHORT) or \
               (signal.direction == -1 and current_side == PositionSide.LONG):
                # Close conflicting position
                self._close_position(current_position)
        
        # Open new position if signal is strong enough
        if signal.confidence >= self.min_confidence:
            self._open_position(signal, bar)
    
    def _open_position(self, signal: TiRexSignal, bar: Bar):
        """Open new position based on signal."""
        try:
            # Get instrument
            instrument = self.cache.instrument(self.instrument_id)
            if not instrument:
                self.log.error(f"Instrument not found: {self.instrument_id}")
                return
            
            # Calculate order quantity
            order_quantity = Quantity.from_str(f"{abs(signal.position_size):.6f}")
            
            # Determine order side
            order_side = OrderSide.BUY if signal.direction == 1 else OrderSide.SELL
            
            # Create market order
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=self.instrument_id,
                order_side=order_side,
                quantity=order_quantity,
                time_in_force=self.default_time_in_force,
                reduce_only=False,
                client_order_id=self.cache.generate_order_id()
            )
            
            # Submit order
            self.submit_order(order)
            
            # Log trade
            console.print(f"📈 TiRex Signal: {signal.direction} | "
                         f"Confidence: {signal.confidence:.2f} | "
                         f"Size: {signal.position_size:.4f} | "
                         f"Regime: {signal.market_regime}")
            
            # Record trade
            self.trade_history.append({
                'timestamp': bar.ts_event,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'size': signal.position_size,
                'price': float(bar.close),
                'regime': signal.market_regime,
                'processing_time_ms': signal.processing_time_ms
            })
            
        except Exception as e:
            self.log.error(f"Failed to open position: {e}")
    
    def _close_position(self, position):
        """Close existing position."""
        try:
            # Create closing order
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=position.instrument_id,
                order_side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                quantity=position.quantity,
                time_in_force=self.default_time_in_force,
                reduce_only=True,
                client_order_id=self.cache.generate_order_id()
            )
            
            self.submit_order(order)
            
            console.print(f"🔄 Closing position: {position.side} {position.quantity}")
            
        except Exception as e:
            self.log.error(f"Failed to close position: {e}")
    
    def _log_performance(self):
        """Log strategy performance metrics."""
        if self.total_predictions == 0:
            return
        
        accuracy = self.successful_predictions / self.total_predictions
        
        model_stats = self.tirex_model.get_performance_stats() if self.tirex_model else {}
        
        self.log.info(
            f"TiRex Performance | "
            f"Predictions: {self.total_predictions} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Avg Inference: {model_stats.get('avg_inference_time_ms', 0):.1f}ms | "
            f"GPU: {model_stats.get('is_gpu_accelerated', False)}"
        )
    
    def on_order_filled(self, order, fill):
        """Handle order fill events."""
        console.print(f"✅ Order filled: {order.side} {fill.quantity} @ {fill.price}")
        
        # Update performance tracking
        if hasattr(self, '_last_fill_price'):
            # Calculate PnL (simplified)
            if order.side == OrderSide.SELL:
                pnl = (float(fill.price) - self._last_fill_price) * float(fill.quantity)
            else:
                pnl = (self._last_fill_price - float(fill.price)) * float(fill.quantity)
            
            self.total_pnl += pnl
            
            # Check if prediction was successful (simplified)
            if pnl > 0:
                self.successful_predictions += 1
        
        self._last_fill_price = float(fill.price)
    
    def on_position_opened(self, position):
        """Handle position opened events."""
        self.current_position_side = position.side
        console.print(f"📊 Position opened: {position.side} {position.quantity}")
    
    def on_position_closed(self, position):
        """Handle position closed events."""
        self.current_position_side = PositionSide.FLAT
        console.print(f"📊 Position closed: PnL = {position.realized_pnl}")
    
    def get_strategy_stats(self) -> Dict:
        """Get comprehensive strategy statistics."""
        model_stats = self.tirex_model.get_performance_stats() if self.tirex_model else {}
        
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'accuracy': self.successful_predictions / max(1, self.total_predictions),
            'total_pnl': self.total_pnl,
            'total_trades': len(self.trade_history),
            'current_position': str(self.current_position_side),
            'last_prediction': {
                'direction': self.last_prediction.direction if self.last_prediction else None,
                'confidence': self.last_prediction.confidence if self.last_prediction else None,
                'regime': self.last_prediction.market_regime if self.last_prediction else None
            } if self.last_prediction else None,
            'model_performance': model_stats
        }