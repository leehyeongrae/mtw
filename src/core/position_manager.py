"""
Position management and tracking
"""
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
from src.utils.logger import get_logger
from src.utils.config import config

class PositionManager:
    """Manages trading positions and cooldowns"""
    
    def __init__(self):
        self.logger = get_logger("position_manager")
        self.positions = {}  # {symbol: position_info}
        self.cooldowns = {}  # {symbol: cooldown_end_time}
        
    def open_position(self, symbol: str, side: str, size: float, entry_price: float) -> Dict:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            
        Returns:
            Position information dict
        """
        position = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'current_price': entry_price,
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'take_profit': self._calculate_take_profit(entry_price, side),
            'stop_loss': None,  # Will be set by trailing stop
            'highest_price': entry_price if side == 'long' else None,
            'lowest_price': entry_price if side == 'short' else None
        }
        
        self.positions[symbol] = position
        self.logger.info(f"Opened {side} position for {symbol} at {entry_price}")
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> Optional[Dict]:
        """
        Close a position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Closed position info or None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate final PnL
        if position['side'] == 'long':
            pnl_percent = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        else:  # short
            pnl_percent = ((position['entry_price'] - exit_price) / position['entry_price']) * 100
        
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['pnl_percent'] = pnl_percent
        position['pnl'] = position['size'] * (pnl_percent / 100)
        position['close_reason'] = reason
        
        # Remove from active positions
        del self.positions[symbol]
        
        # Set cooldown - reason을 전달하도록 수정
        self._set_cooldown(symbol, pnl_percent, reason)  # reason 파라미터 추가
        
        self.logger.info(
            f"Closed {position['side']} position for {symbol} at {exit_price} "
            f"(PnL: {pnl_percent:.2f}%, Reason: {reason})"
        )
        
        return position

    def _set_cooldown(self, symbol: str, pnl_percent: float, reason: str = ""):
        """
        Set cooldown period after closing position
        
        Args:
            symbol: Trading symbol
            pnl_percent: PnL percentage of closed position
            reason: Reason for closing (e.g., 'stop_loss', 'take_profit', 'signal_exit')
        """
        base_cooldown = config.position_cooldown_seconds
        
        # Check if it's a stop loss closure
        if 'stop_loss' in reason.lower() or 'stop' in reason.lower():
            # Stop loss로 청산된 경우 multiplier 적용
            cooldown_seconds = base_cooldown * config.cooldown_multiplier
            self.logger.info(
                f"Stop loss cooldown for {symbol}: {cooldown_seconds}s "
                f"(base: {base_cooldown}s × multiplier: {config.cooldown_multiplier})"
            )
        else:
            # 일반 청산 (TP, 신호 청산 등) - 기본 쿨다운만 적용
            cooldown_seconds = base_cooldown
            self.logger.info(
                f"Normal cooldown for {symbol}: {cooldown_seconds}s "
                f"(Reason: {reason if reason else 'general'})"
            )
        
        self.cooldowns[symbol] = time.time() + cooldown_seconds
        self.logger.debug(
            f"Cooldown set for {symbol} until "
            f"{datetime.fromtimestamp(self.cooldowns[symbol]).strftime('%H:%M:%S')}"
        )
    
    def update_position(self, symbol: str, current_price: float, atr: Optional[float] = None) -> Optional[Dict]:
        """
        Update position with current price and trailing stop
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            atr: Current ATR value for trailing stop
            
        Returns:
            Updated position or None
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position['current_price'] = current_price
        
        # Update PnL
        if position['side'] == 'long':
            pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            # Update highest price
            if position['highest_price'] is None or current_price > position['highest_price']:
                position['highest_price'] = current_price
            
        else:  # short
            pnl_percent = ((position['entry_price'] - current_price) / position['entry_price']) * 100
            
            # Update lowest price
            if position['lowest_price'] is None or current_price < position['lowest_price']:
                position['lowest_price'] = current_price
        
        position['pnl_percent'] = pnl_percent
        position['pnl'] = position['size'] * (pnl_percent / 100)
        
        # Update trailing stop if ATR provided
        if atr and atr > 0:
            self._update_trailing_stop(position, current_price, atr)
        
        return position
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        tp_percent = config.take_profit_percent / 100
        
        if side == 'long':
            return entry_price * (1 + tp_percent)
        else:  # short
            return entry_price * (1 - tp_percent)
    
    def _update_trailing_stop(self, position: Dict, current_price: float, atr: float):
        """Update trailing stop loss"""
        multiplier = config.stop_trailing_atr_multiplier
        min_percent = config.stop_trailing_min_percent / 100
        max_percent = config.stop_trailing_max_percent / 100
        
        # Calculate ATR-based stop distance
        atr_stop_distance = atr * multiplier
        
        # Apply min/max constraints
        min_distance = position['entry_price'] * min_percent
        max_distance = position['entry_price'] * max_percent
        stop_distance = max(min_distance, min(atr_stop_distance, max_distance))
        
        if position['side'] == 'long':
            # Trail from highest price
            new_stop = position['highest_price'] - stop_distance
            
            # Only update if new stop is higher
            if position['stop_loss'] is None or new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                
        else:  # short
            # Trail from lowest price
            new_stop = position['lowest_price'] + stop_distance
            
            # Only update if new stop is lower
            if position['stop_loss'] is None or new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
    
    def _set_cooldown(self, symbol: str, pnl_percent: float):
        """Set cooldown period after closing position"""
        base_cooldown = config.position_cooldown_seconds
        
        # Increase cooldown for losses
        if pnl_percent < 0:
            cooldown_seconds = base_cooldown * config.cooldown_multiplier
        else:
            cooldown_seconds = base_cooldown
        
        self.cooldowns[symbol] = time.time() + cooldown_seconds
        self.logger.debug(f"Set cooldown for {symbol}: {cooldown_seconds} seconds")
    
    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown"""
        if symbol not in self.cooldowns:
            return False
        
        return time.time() < self.cooldowns[symbol]
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Get all open positions"""
        return self.positions.copy()
    
    def should_close_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position should be closed
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Close reason or None
        """
        position = self.get_position(symbol)
        if not position:
            return None
        
        # Check take profit
        if position['side'] == 'long':
            if current_price >= position['take_profit']:
                return "take_profit"
            if position['stop_loss'] and current_price <= position['stop_loss']:
                return "stop_loss"
        else:  # short
            if current_price <= position['take_profit']:
                return "take_profit"
            if position['stop_loss'] and current_price >= position['stop_loss']:
                return "stop_loss"
        
        return None