"""
Advanced Risk Management System
===============================

Parameter-free risk management that adapts to market conditions and performance.
"""



class AdaptiveRiskManager:
    """
    Advanced risk management system that adapts to market conditions
    and trading performance without requiring manual parameter tuning.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Performance tracking
        self.trade_history: list[dict] = []
        self.drawdown_history: list[float] = []
        self.volatility_history: list[float] = []
        
        # Adaptive parameters
        self.base_risk_per_trade = 0.02  # 2% base risk
        self.max_risk_per_trade = 0.05   # 5% maximum risk
        self.min_risk_per_trade = 0.005  # 0.5% minimum risk
        
        # Dynamic risk adjustment
        self.risk_multiplier = 1.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Market condition tracking
        self.market_volatility = 0.0
        self.market_trend_strength = 0.0
        
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float | None = None,
                              market_regime: str = "UNKNOWN") -> float:
        """
        Calculate optimal position size based on current risk parameters
        and market conditions.
        """
        # Adjust base risk based on performance
        current_risk = self._calculate_dynamic_risk()
        
        # Adjust for market regime
        regime_adjustment = self._get_regime_risk_adjustment(market_regime)
        adjusted_risk = current_risk * regime_adjustment
        
        # Ensure within bounds
        adjusted_risk = max(self.min_risk_per_trade, 
                          min(adjusted_risk, self.max_risk_per_trade))
        
        # Calculate position size
        risk_capital = self.current_capital * adjusted_risk
        
        if stop_loss:
            # Position size based on stop loss
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_capital / risk_per_unit
        else:
            # Fixed percentage of capital approach
            position_size = risk_capital / entry_price
            
        return position_size
    
    def _calculate_dynamic_risk(self) -> float:
        """Calculate dynamic risk based on recent performance."""
        base_risk = self.base_risk_per_trade
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= 3:
            loss_penalty = 0.1 * (self.consecutive_losses - 2)
            base_risk *= (1 - loss_penalty)
            
        # Slightly increase risk after consistent wins (but cap it)
        elif self.consecutive_wins >= 5:
            win_bonus = 0.05 * min(self.consecutive_wins - 4, 3)
            base_risk *= (1 + win_bonus)
            
        # Adjust based on current drawdown
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > 0.05:  # 5% drawdown
            drawdown_penalty = current_drawdown * 0.5
            base_risk *= (1 - drawdown_penalty)
            
        return base_risk
    
    def _get_regime_risk_adjustment(self, regime: str) -> float:
        """Get risk adjustment factor based on market regime."""
        adjustments = {
            "TRENDING": 1.2,    # Slightly higher risk in trending markets
            "RANGING": 1.0,     # Normal risk in ranging markets
            "VOLATILE": 0.6,    # Much lower risk in volatile markets
            "UNKNOWN": 0.8      # Conservative in unknown conditions
        }
        return adjustments.get(regime, 0.8)
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self.trade_history:
            return 0.0
            
        # Find peak capital
        peak_capital = self.initial_capital
        for trade in self.trade_history:
            capital_after_trade = trade.get('capital_after', self.initial_capital)
            peak_capital = max(peak_capital, capital_after_trade)
            
        # Calculate drawdown
        drawdown = (peak_capital - self.current_capital) / peak_capital
        return max(0.0, drawdown)
    
    def update_performance(self, trade_pnl: float, trade_outcome: str):
        """Update performance tracking after each trade."""
        self.current_capital += trade_pnl
        
        # Update trade history
        trade_record = {
            'pnl': trade_pnl,
            'outcome': trade_outcome,
            'capital_after': self.current_capital,
            'drawdown': self._calculate_current_drawdown()
        }
        self.trade_history.append(trade_record)
        
        # Update consecutive win/loss tracking
        if trade_outcome == 'WIN':
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # Keep only recent history (last 100 trades)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def should_stop_trading(self) -> bool:
        """Determine if trading should be stopped due to risk conditions."""
        # Stop if drawdown is too large
        if self._calculate_current_drawdown() > 0.20:  # 20% drawdown
            return True
            
        # Stop if too many consecutive losses
        if self.consecutive_losses >= 8:
            return True
            
        # Stop if capital is too low
        if self.current_capital < self.initial_capital * 0.5:  # 50% of initial
            return True
            
        return False
    
    def get_max_positions(self) -> int:
        """Get maximum number of concurrent positions based on risk."""
        # Reduce max positions as drawdown increases
        drawdown = self._calculate_current_drawdown()
        
        if drawdown > 0.15:
            return 1  # Only one position in high drawdown
        elif drawdown > 0.10:
            return 2  # Two positions in moderate drawdown
        elif drawdown > 0.05:
            return 3  # Three positions in small drawdown
        else:
            return 5  # Up to 5 positions when performing well
    
    def get_risk_metrics(self) -> dict:
        """Get current risk metrics for monitoring."""
        return {
            'current_capital': self.current_capital,
            'current_drawdown': self._calculate_current_drawdown(),
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'dynamic_risk': self._calculate_dynamic_risk(),
            'should_stop': self.should_stop_trading(),
            'max_positions': self.get_max_positions(),
            'total_trades': len(self.trade_history)
        }


class TransactionCostOptimizer:
    """
    Optimizes trading frequency to account for transaction costs.
    """
    
    def __init__(self, maker_fee: float = 0.00012, taker_fee: float = 0.00032):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.recent_trades: list[dict] = []
        
    def should_trade(self, expected_profit: float, position_size: float, 
                    entry_price: float, confidence: float = 1.0) -> bool:
        """
        Determine if a trade should be executed based on expected profit
        vs transaction costs.
        """
        # Calculate total transaction cost (entry + exit)
        trade_value = position_size * entry_price
        total_cost = trade_value * (self.taker_fee * 2)  # Assume taker for both
        
        # Required profit to break even
        breakeven_profit = total_cost
        
        # Apply confidence adjustment
        required_profit = breakeven_profit / confidence
        
        # Only trade if expected profit significantly exceeds costs
        return expected_profit > required_profit * 1.5  # 50% buffer
    
    def optimize_exit_timing(self, current_profit: float, 
                           position_size: float, entry_price: float) -> bool:
        """
        Optimize exit timing based on current profit and costs.
        """
        trade_value = position_size * entry_price
        exit_cost = trade_value * self.taker_fee
        
        # Don't exit if profit doesn't cover exit cost plus buffer
        return current_profit > exit_cost * 2.0
