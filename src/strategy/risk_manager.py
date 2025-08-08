"""
Risk management module
"""
from typing import Dict, Optional, Tuple
from src.utils.logger import get_logger
from src.utils.config import config

class RiskManager:
    """Manages risk parameters and position sizing"""
    
    def __init__(self):
        self.logger = get_logger("risk_manager")
        self.max_positions = min(config.symbol_count, 10)  # Limit concurrent positions
        self.position_size_percent = config.position_size_percent
        
    def calculate_position_size(self, account_balance: float, current_positions: int) -> float:
        """
        Calculate position size based on account balance and risk parameters
        
        Args:
            account_balance: Current account balance
            current_positions: Number of current open positions
            
        Returns:
            Position size in USDT
        """
        # Check if we can open more positions
        if current_positions >= self.max_positions:
            self.logger.warning(f"Maximum positions reached: {current_positions}/{self.max_positions}")
            return 0.0
        
        # Calculate base position size
        position_size = account_balance * (self.position_size_percent / 100)
        
        # Adjust for number of positions
        remaining_slots = self.max_positions - current_positions
        if remaining_slots < self.max_positions:
            # Reduce size as we approach max positions
            adjustment_factor = remaining_slots / self.max_positions
            position_size *= (0.5 + 0.5 * adjustment_factor)
        
        # Apply leverage
        position_size *= config.leverage
        
        return round(position_size, 2)
    
    def validate_entry_signal(self, symbol: str, signal: Dict, positions: Dict, 
                            cooldowns: Dict, account_balance: float) -> Tuple[bool, str]:
        """
        Validate if entry signal should be executed
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            positions: Current positions
            cooldowns: Cooldown dictionary
            account_balance: Current account balance
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if already in position
        if symbol in positions:
            return False, "Already in position"
        
        # Check cooldown
        if symbol in cooldowns and cooldowns[symbol]:
            return False, "In cooldown period"
        
        # Check position limit
        if len(positions) >= self.max_positions:
            return False, "Maximum positions reached"
        
        # Check account balance
        min_position_size = 10.0  # Minimum $10 position
        position_size = self.calculate_position_size(account_balance, len(positions))
        
        if position_size < min_position_size:
            return False, "Insufficient balance for position"
        
        # Validate signal quality
        if signal.get('market_type') == 'trending':
            # For trending market, ensure strong trend signals
            indicators = signal.get('indicators', {})
            
            if signal['action'] in ['long_entry', 'short_entry']:
                # Check trend strength
                adx = indicators.get('adx', 0)
        
        elif signal.get('market_type') == 'ranging':
            # For ranging market, ensure strong oscillator signals
            oscillator_score = signal.get('indicators', {}).get('oscillator_score', 0)
            
            if abs(oscillator_score) < 0.5:  # Minimum score threshold
                return False, f"Weak oscillator signal (Score: {oscillator_score:.2f})"
        
        return True, "Valid signal"
    
    def calculate_order_prices(self, current_price: float, side: str) -> Dict:
        """
        Calculate order prices with slippage protection
        
        Args:
            current_price: Current market price
            side: 'long' or 'short'
            
        Returns:
            Dictionary with order prices
        """
        slippage = 0.001  # 0.1% slippage allowance
        
        if side == 'long':
            entry_price = current_price * (1 + slippage)
            take_profit = entry_price * (1 + config.take_profit_percent / 100)
            initial_stop = entry_price * (1 - config.stop_trailing_max_percent / 100)
        else:  # short
            entry_price = current_price * (1 - slippage)
            take_profit = entry_price * (1 - config.take_profit_percent / 100)
            initial_stop = entry_price * (1 + config.stop_trailing_max_percent / 100)
        
        return {
            'entry': round(entry_price, 2),
            'take_profit': round(take_profit, 2),
            'initial_stop': round(initial_stop, 2)
        }
    
    def check_risk_limits(self, positions: Dict, account_balance: float) -> Dict:
        """
        Check overall risk limits
        
        Args:
            positions: Current positions dictionary
            account_balance: Current account balance
            
        Returns:
            Risk status dictionary
        """
        total_exposure = sum(p.get('size', 0) for p in positions.values())
        total_pnl = sum(p.get('pnl', 0) for p in positions.values())
        
        # Calculate risk metrics
        exposure_percent = (total_exposure / account_balance * 100) if account_balance > 0 else 0
        pnl_percent = (total_pnl / account_balance * 100) if account_balance > 0 else 0
        
        # Define risk limits
        max_exposure_percent = 50.0  # Maximum 50% account exposure
        max_drawdown_percent = 10.0  # Maximum 10% drawdown
        
        status = {
            'total_exposure': total_exposure,
            'exposure_percent': exposure_percent,
            'total_pnl': total_pnl,
            'pnl_percent': pnl_percent,
            'position_count': len(positions),
            'max_positions': self.max_positions,
            'is_healthy': True,
            'warnings': []
        }
        
        # Check exposure limit
        if exposure_percent > max_exposure_percent:
            status['is_healthy'] = False
            status['warnings'].append(f"High exposure: {exposure_percent:.1f}%")
        
        # Check drawdown limit
        if pnl_percent < -max_drawdown_percent:
            status['is_healthy'] = False
            status['warnings'].append(f"High drawdown: {pnl_percent:.1f}%")
        
        # Check position concentration
        if len(positions) == 1 and exposure_percent > 20:
            status['warnings'].append("High concentration in single position")
        
        return status