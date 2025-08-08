"""
Central data management with multiprocessing support
"""
import multiprocessing as mp
from multiprocessing import Manager, Queue, Process
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from src.utils.logger import get_logger
from src.utils.config import config

class DataManager:
    """Central data manager for multiprocessing coordination"""
    
    def __init__(self):
        self.logger = get_logger("data_manager")
        self.manager = Manager()
        
        # Shared data structures
        self.candle_data = self.manager.dict()  # {symbol: {'historical': df, 'current': dict}}
        self.indicators = self.manager.dict()   # {symbol: dict of indicators}
        self.signals = self.manager.dict()      # {symbol: signal_dict}
        self.positions = self.manager.dict()    # {symbol: position_info}
        self.symbol_list = self.manager.list()  # Active symbols
        
        # Control flags
        self.running = mp.Value('b', True)
        self.last_update = self.manager.dict()  # {symbol: timestamp}
        
        # Locks for thread safety
        self.locks = {
            'candle': mp.Lock(),
            'indicator': mp.Lock(),
            'signal': mp.Lock(),
            'position': mp.Lock(),
            'symbol': mp.Lock()
        }
        
        # Queues for communication
        self.ws_queue = Queue()  # WebSocket data queue
        self.rest_queue = Queue()  # REST API data queue
        self.signal_queue = Queue()  # Trading signals queue
        
        self.logger.info("DataManager initialized")
    
    def update_candles(self, symbol: str, historical_df: Optional[pd.DataFrame] = None, 
                      current_candle: Optional[Dict] = None) -> bool:
        """
        Update candle data for a symbol
        
        Args:
            symbol: Trading symbol
            historical_df: Historical candles DataFrame
            current_candle: Current candle dict
            
        Returns:
            bool: Success status
        """
        with self.locks['candle']:
            try:
                if symbol not in self.candle_data:
                    self.candle_data[symbol] = self.manager.dict({
                        'historical': None,
                        'current': None
                    })
                
                data = self.candle_data[symbol]
                
                if historical_df is not None:
                    data['historical'] = historical_df.to_dict()
                    
                if current_candle is not None:
                    data['current'] = current_candle
                
                self.candle_data[symbol] = data
                self.last_update[symbol] = time.time()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating candles for {symbol}: {e}")
                return False
    
    def get_candles(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Get candle data for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (historical_df, current_candle)
        """
        with self.locks['candle']:
            if symbol not in self.candle_data:
                return None, None
            
            data = self.candle_data[symbol]
            
            historical = None
            if data['historical'] is not None:
                historical = pd.DataFrame.from_dict(data['historical'])
            
            return historical, data.get('current')
    
    def update_indicators(self, symbol: str, indicators: Dict) -> bool:
        """
        Update indicators for a symbol
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of calculated indicators
            
        Returns:
            bool: Success status
        """
        with self.locks['indicator']:
            try:
                self.indicators[symbol] = indicators
                return True
            except Exception as e:
                self.logger.error(f"Error updating indicators for {symbol}: {e}")
                return False
    
    def get_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get indicators for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators or None
        """
        with self.locks['indicator']:
            return self.indicators.get(symbol)
    
    def update_signal(self, symbol: str, signal: Dict) -> bool:
        """
        Update trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            
        Returns:
            bool: Success status
        """
        with self.locks['signal']:
            try:
                self.signals[symbol] = signal
                # Also put in signal queue for immediate processing
                self.signal_queue.put((symbol, signal))
                return True
            except Exception as e:
                self.logger.error(f"Error updating signal for {symbol}: {e}")
                return False
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """
        Get current signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Signal dictionary or None
        """
        with self.locks['signal']:
            return self.signals.get(symbol)
    
    def update_position(self, symbol: str, position_info: Optional[Dict]) -> bool:
        """
        Update position information for a symbol
        
        Args:
            symbol: Trading symbol
            position_info: Position information or None to clear
            
        Returns:
            bool: Success status
        """
        with self.locks['position']:
            try:
                if position_info is None:
                    if symbol in self.positions:
                        del self.positions[symbol]
                else:
                    self.positions[symbol] = position_info
                return True
            except Exception as e:
                self.logger.error(f"Error updating position for {symbol}: {e}")
                return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position information for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position dictionary or None
        """
        with self.locks['position']:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """
        Get all open positions
        
        Returns:
            Dictionary of all positions
        """
        with self.locks['position']:
            return dict(self.positions)
    
    def update_symbol_list(self, symbols: List[str]) -> bool:
        """
        Update the active symbol list
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            bool: Success status
        """
        with self.locks['symbol']:
            try:
                self.symbol_list[:] = symbols
                self.logger.info(f"Symbol list updated: {symbols}")
                return True
            except Exception as e:
                self.logger.error(f"Error updating symbol list: {e}")
                return False
    
    def get_symbol_list(self) -> List[str]:
        """
        Get the active symbol list
        
        Returns:
            List of trading symbols
        """
        with self.locks['symbol']:
            return list(self.symbol_list)
    
    def add_ws_data(self, data: Dict) -> None:
        """Add WebSocket data to queue"""
        self.ws_queue.put(data)
    
    def add_rest_data(self, data: Dict) -> None:
        """Add REST API data to queue"""
        self.rest_queue.put(data)
    
    def get_ws_data(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get WebSocket data from queue"""
        try:
            return self.ws_queue.get(timeout=timeout)
        except:
            return None
    
    def get_rest_data(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get REST API data from queue"""
        try:
            return self.rest_queue.get(timeout=timeout)
        except:
            return None
    
    def get_signal_from_queue(self, timeout: float = 0.1) -> Optional[Tuple[str, Dict]]:
        """Get signal from queue"""
        try:
            return self.signal_queue.get(timeout=timeout)
        except:
            return None
    
    def is_running(self) -> bool:
        """Check if system is running"""
        return self.running.value
    
    def stop(self) -> None:
        """Stop the data manager"""
        self.running.value = False
        self.logger.info("DataManager stopped")
    
    def get_status(self, symbol: str) -> Dict:
        """
        Get complete status for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with complete status information
        """
        historical, current = self.get_candles(symbol)
        indicators = self.get_indicators(symbol)
        signal = self.get_signal(symbol)
        position = self.get_position(symbol)
        
        status = {
            'symbol': symbol,
            'has_historical': historical is not None,
            'has_current': current is not None,
            'has_indicators': indicators is not None,
            'has_signal': signal is not None,
            'has_position': position is not None,
            'last_update': self.last_update.get(symbol, 0)
        }
        
        if current:
            status['current_price'] = current.get('close', 0)
            
        if indicators:
            status['market_type'] = 'trending' if indicators.get('is_trending') else 'ranging'
            
        if position:
            status['position_side'] = position.get('side')
            status['position_size'] = position.get('size')
            
        return status
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop()
        self.ws_queue.close()
        self.rest_queue.close()
        self.signal_queue.close()
        self.manager.shutdown()
        self.logger.info("DataManager cleaned up")