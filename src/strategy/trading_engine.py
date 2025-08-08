"""
Main trading engine
"""
import asyncio
from typing import Dict, Optional
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.core.position_manager import PositionManager
from src.strategy.risk_manager import RiskManager
from src.exchange.order_executor import OrderExecutor

class TradingEngine:
    """Main trading engine that processes signals and executes trades"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("trading_engine")
        self.data_manager = data_manager
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.running = True
        
    async def process_signal(self, symbol: str, signal: Dict):
        """
        Process a trading signal
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
        """
        try:
            action = signal.get('action')
            
            if not action:
                return
            
            # Get current account balance
            balance = self.order_executor.get_account_balance()
            
            # Handle entry signals
            if 'entry' in action:
                await self._handle_entry_signal(symbol, signal, balance)
            
            # Handle exit signals
            elif 'exit' in action:
                await self._handle_exit_signal(symbol, signal)
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def _handle_entry_signal(self, symbol: str, signal: Dict, balance: float):
        """Handle entry signals"""
        
        # Check if already in position
        if self.position_manager.get_position(symbol):
            self.logger.debug(f"Already in position for {symbol}")
            return
        
        # Check cooldown
        if self.position_manager.is_in_cooldown(symbol):
            self.logger.debug(f"{symbol} is in cooldown")
            return
        
        # Validate signal
        all_positions = self.position_manager.get_all_positions()
        cooldowns = self.position_manager.cooldowns
        
        is_valid, reason = self.risk_manager.validate_entry_signal(
            symbol, signal, all_positions, cooldowns, balance
        )
        
        if not is_valid:
            self.logger.debug(f"Signal rejected for {symbol}: {reason}")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            balance, len(all_positions)
        )
        
        if position_size <= 0:
            self.logger.warning(f"Invalid position size for {symbol}")
            return
        
        # Determine side
        side = 'long' if 'long' in signal['action'] else 'short'
        
        # Execute order
        order = self.order_executor.open_position(symbol, side, position_size)
        
        if order:
            # Get execution price
            exec_price = float(order.get('avgPrice', order.get('price', 0)))
            
            # Record position
            position = self.position_manager.open_position(
                symbol, side, position_size, exec_price
            )
            
            # Update data manager
            self.data_manager.update_position(symbol, position)
            
            self.logger.info(
                f"Opened {side} position for {symbol}: "
                f"Size=${position_size:.2f}, Price={exec_price:.2f}, "
                f"Reason={signal.get('reason', 'Signal triggered')}"
            )
    
    async def _handle_exit_signal(self, symbol: str, signal: Dict):
        """Handle exit signals"""
        
        position = self.position_manager.get_position(symbol)
        if not position:
            self.logger.debug(f"No position to close for {symbol}")
            return
        
        # Execute close order
        order = self.order_executor.close_position(symbol, position['side'])
        
        if order:
            # Get execution price
            exec_price = float(order.get('avgPrice', order.get('price', 0)))
            
            # Record closed position
            closed = self.position_manager.close_position(
                symbol, exec_price, signal.get('reason', 'Signal exit')
            )
            
            # Clear from data manager
            self.data_manager.update_position(symbol, None)
            
            if closed:
                self.logger.info(
                    f"Closed {position['side']} position for {symbol}: "
                    f"Price={exec_price:.2f}, PnL={closed['pnl_percent']:.2f}%, "
                    f"Reason={signal.get('reason', 'Signal triggered')}"
                )
    
    async def check_risk_stops(self):
        """Check and execute risk-based stops"""
        
        positions = self.position_manager.get_all_positions()
        
        for symbol, position in positions.items():
            try:
                # Get current candle
                _, current = self.data_manager.get_candles(symbol)
                if not current:
                    continue
                
                current_price = current.get('c', 0)
                if not current_price:
                    continue
                
                # Get ATR for trailing stop
                indicators = self.data_manager.get_indicators(symbol)
                atr = None
                if indicators and 'atr' in indicators:
                    atr_values = indicators['atr']
                    if len(atr_values) > 0:
                        atr = atr_values[-1]
                
                # Update position with current price
                updated = self.position_manager.update_position(
                    symbol, current_price, atr
                )
                
                if updated:
                    # Check if position should be closed
                    close_reason = self.position_manager.should_close_position(
                        symbol, current_price
                    )
                    
                    if close_reason:
                        # Execute close
                        order = self.order_executor.close_position(
                            symbol, position['side']
                        )
                        
                        if order:
                            exec_price = float(order.get('avgPrice', order.get('price', 0)))
                            closed = self.position_manager.close_position(
                                symbol, exec_price, close_reason
                            )
                            self.data_manager.update_position(symbol, None)
                            
                            if closed:
                                self.logger.info(
                                    f"Risk stop triggered for {symbol}: "
                                    f"{close_reason}, PnL={closed['pnl_percent']:.2f}%"
                                )
                
            except Exception as e:
                self.logger.error(f"Error checking stops for {symbol}: {e}")
    
    async def update_positions_from_exchange(self):
        """Sync positions with exchange"""
        
        try:
            # Get positions from exchange
            exchange_positions = self.order_executor.get_all_positions()
            
            # Update local positions
            for symbol, pos_info in exchange_positions.items():
                local_pos = self.position_manager.get_position(symbol)
                
                if not local_pos:
                    # Position exists on exchange but not locally
                    self.position_manager.open_position(
                        symbol,
                        pos_info['side'],
                        pos_info['quantity'] * pos_info['entry_price'],
                        pos_info['entry_price']
                    )
                    self.logger.info(f"Synced position from exchange: {symbol}")
            
            # Check for positions that exist locally but not on exchange
            for symbol in list(self.position_manager.positions.keys()):
                if symbol not in exchange_positions:
                    self.position_manager.close_position(
                        symbol, 0, "Position not found on exchange"
                    )
                    self.data_manager.update_position(symbol, None)
                    self.logger.warning(f"Removed orphaned position: {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")
    
    async def run(self):
        """Main trading loop"""
        self.logger.info("Starting trading engine")
        
        # Initial position sync
        await self.update_positions_from_exchange()
        
        while self.data_manager.is_running():
            try:
                # Process signal queue
                signal_data = self.data_manager.get_signal_from_queue(timeout=0.1)
                
                if signal_data:
                    symbol, signal = signal_data
                    await self.process_signal(symbol, signal)
                
                # Check risk stops periodically
                await self.check_risk_stops()
                
                # Sync positions periodically (every 30 seconds)
                if int(asyncio.get_event_loop().time()) % 30 == 0:
                    await self.update_positions_from_exchange()
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)
        
        self.logger.info("Trading engine stopped")
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.logger.info("Stopping trading engine")