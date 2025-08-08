"""
Order execution module
"""
from typing import Dict, Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.logger import get_logger
from src.utils.config import config

class OrderExecutor:
    """Handles order execution on Binance"""
    
    def __init__(self):
        self.logger = get_logger("order_executor")
        self.client = self._init_client()
        
    def _init_client(self) -> Client:
        """Initialize Binance client"""
        if config.binance_testnet:
            client = Client(
                config.binance_api_key,
                config.binance_api_secret,
                testnet=True
            )
        else:
            client = Client(
                config.binance_api_key,
                config.binance_api_secret
            )
        return client
    
    def set_leverage(self, symbol: str) -> bool:
        """
        Set leverage for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: Success status
        """
        try:
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=config.leverage
            )
            self.logger.info(f"Set leverage to {config.leverage}x for {symbol}")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def open_position(self, symbol: str, side: str, size: float, 
                     entry_price: Optional[float] = None) -> Optional[Dict]:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size in USDT
            entry_price: Limit price (optional)
            
        Returns:
            Order response or None
        """
        try:
            # Set leverage first
            self.set_leverage(symbol)
            
            # Get symbol info for precision
            symbol_info = self._get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Calculate quantity
            current_price = self._get_current_price(symbol)
            if not current_price:
                return None
            
            quantity = self._calculate_quantity(size, current_price, symbol_info)
            if not quantity:
                return None
            
            # Determine order side
            order_side = 'BUY' if side == 'long' else 'SELL'
            
            # Place market order for immediate execution
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=quantity
            )
            
            self.logger.info(f"Opened {side} position for {symbol}: {quantity} @ market")
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to open position for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, side: str) -> Optional[Dict]:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            side: Current position side ('long' or 'short')
            
        Returns:
            Order response or None
        """
        try:
            # Get current position
            position = self._get_position(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return None
            
            quantity = abs(float(position['positionAmt']))
            if quantity == 0:
                self.logger.warning(f"Position already closed for {symbol}")
                return None
            
            # Determine close order side (opposite of position)
            order_side = 'SELL' if side == 'long' else 'BUY'
            
            # Place market order to close
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=quantity,
                reduceOnly=True
            )
            
            self.logger.info(f"Closed {side} position for {symbol}: {quantity} @ market")
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to close position for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def get_account_balance(self) -> float:
        """
        Get account USDT balance
        
        Returns:
            USDT balance or 0.0
        """
        try:
            account = self.client.futures_account()
            
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_all_positions(self) -> Dict:
        """
        Get all open positions
        
        Returns:
            Dictionary of positions by symbol
        """
        try:
            positions = self.client.futures_position_information()
            
            result = {}
            for pos in positions:
                quantity = float(pos['positionAmt'])
                if quantity != 0:
                    symbol = pos['symbol']
                    result[symbol] = {
                        'symbol': symbol,
                        'side': 'long' if quantity > 0 else 'short',
                        'quantity': abs(quantity),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'pnl': float(pos['unRealizedProfit'])
                    }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading rules"""
        try:
            info = self.client.futures_exchange_info()
            
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return s
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            if positions:
                return positions[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def _calculate_quantity(self, size_usdt: float, price: float, symbol_info: Dict) -> Optional[float]:
        """Calculate order quantity with correct precision"""
        try:
            # Calculate raw quantity
            quantity = size_usdt / price
            
            # Get precision from symbol info
            precision = 3  # Default precision
            for filter in symbol_info.get('filters', []):
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    # Calculate precision from step size
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    break
            
            # Round to correct precision
            quantity = round(quantity, precision)
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating quantity: {e}")
            return None