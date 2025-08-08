"""
Multiprocessing indicator calculation
"""
import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.indicators.indicators import Indicators
from src.indicators.signals import SignalGenerator
from typing import Dict, Optional, List 

class IndicatorProcessor:
    """Process indicators for a single symbol"""
    
    def __init__(self, symbol: str, data_manager: DataManager):
        self.symbol = symbol
        self.data_manager = data_manager
        self.logger = get_logger(f"indicator_{symbol}")
        self.indicators = Indicators()
        self.signal_generator = SignalGenerator(symbol)
        self.last_closed_candle_time = None
        self.cached_hurst = None
        
    def calculate_indicators(self, df: pd.DataFrame, current_candle: Optional[Dict] = None) -> Dict:
        """
        Calculate all indicators
        
        Args:
            df: Historical candles DataFrame
            current_candle: Current candle dict
            
        Returns:
            Dict of calculated indicators
        """
        try:
            # Merge current candle if available
            if current_candle and not current_candle.get('x', False):
                # Add current candle to DataFrame for calculation
                current_row = pd.DataFrame([{
                    'open_time': current_candle['t'],
                    'open': current_candle['o'],
                    'high': current_candle['h'],
                    'low': current_candle['l'],
                    'close': current_candle['c'],
                    'volume': current_candle['v'],
                    'close_time': current_candle['T'],
                    'quote_volume': current_candle['q'],
                    'trades': current_candle['n']
                }])
                df = pd.concat([df, current_row], ignore_index=True)
            
            # Check if we need to recalculate Hurst (only on closed candles)
            exclude_hurst = True
            if current_candle and current_candle.get('x', False):
                # Candle is closed, calculate Hurst
                if self.last_closed_candle_time != current_candle['T']:
                    exclude_hurst = False
                    self.last_closed_candle_time = current_candle['T']
            
            # Calculate indicators
            result = self.indicators.calculate_all(
                df,
                cci_length=config.cci_length,
                cci_smoothing=config.cci_smoothing,
                rsi_length=config.rsi_length,
                supertrend_atr_length=config.supertrend_atr_length,
                supertrend_multiplier=config.supertrend_multiplier,
                psar_start=config.psar_start,
                psar_increment=config.psar_increment,
                psar_maximum=config.psar_maximum,
                vi_length=config.vi_length,
                mfi_length=config.mfi_length,
                hurst_window=config.hurst_window,
                hurst_rs_lag=config.hurst_rs_lag,
                hurst_smoothing=config.hurst_smoothing,
                vwma_short=config.vwma_short,
                vwma_mid=config.vwma_mid,
                vwma_long=config.vwma_long,
                atr_length=config.atr_length,
                adx_length=config.adx_length,
                adx_smoothing=config.adx_smoothing,
                symbol=self.symbol,
                exclude_hurst=exclude_hurst
            )
            
            # Use cached Hurst if not recalculated
            if exclude_hurst and self.cached_hurst is not None:
                result['hurst'] = self.cached_hurst.get('hurst', np.array([]))
                result['hurst_smoothed'] = self.cached_hurst.get('hurst_smoothed', np.array([]))
            else:
                # Cache the new Hurst values
                self.cached_hurst = {
                    'hurst': result.get('hurst', np.array([])),
                    'hurst_smoothed': result.get('hurst_smoothed', np.array([]))
                }
            
            # Add trending market detection
            if len(result.get('adx', [])) > 0 and len(result.get('hurst_smoothed', [])) > 0:
                result['is_trending'] = self.signal_generator.is_trending_market_by_slope(
                    result['adx'],
                    result['hurst_smoothed']
                )
            else:
                result['is_trending'] = 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def generate_signal(self, indicators: Dict) -> Optional[Dict]:
        """
        Generate trading signal from indicators
        
        Args:
            indicators: Calculated indicators
            
        Returns:
            Signal dict or None
        """
        try:
            # Get current position
            position = self.data_manager.get_position(self.symbol)
            current_position = None
            
            if position:
                current_position = position.get('side')  # 'long' or 'short'
            
            # Generate signal
            signal = self.signal_generator.generate_signal(indicators, current_position)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def process(self):
        """Main processing loop for this symbol"""
        self.logger.info(f"Starting indicator processor for {self.symbol}")
        
        while self.data_manager.is_running():
            try:
                # Get latest candle data
                historical_df, current_candle = self.data_manager.get_candles(self.symbol)
                
                if historical_df is not None and len(historical_df) > 100:
                    # Calculate indicators
                    indicators = self.calculate_indicators(historical_df, current_candle)
                    
                    if indicators:
                        # Update indicators in data manager
                        self.data_manager.update_indicators(self.symbol, indicators)
                        
                        # Generate signal
                        signal = self.generate_signal(indicators)
                        
                        if signal:
                            # Update signal in data manager
                            self.data_manager.update_signal(self.symbol, signal)
                            self.logger.info(f"Signal generated for {self.symbol}: {signal['action']}")
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
        
        self.logger.info(f"Indicator processor stopped for {self.symbol}")


def run_indicator_processor(symbol: str, data_manager: DataManager):
    """
    Worker function for multiprocessing
    
    Args:
        symbol: Trading symbol
        data_manager: Shared data manager
    """
    processor = IndicatorProcessor(symbol, data_manager)
    processor.process()


class IndicatorProcessorManager:
    """Manages multiple indicator processors"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("indicator_manager")
        self.data_manager = data_manager
        self.processes = {}
        
    def start_processor(self, symbol: str):
        """
        Start indicator processor for a symbol
        
        Args:
            symbol: Trading symbol
        """
        if symbol not in self.processes:
            process = mp.Process(
                target=run_indicator_processor,
                args=(symbol, self.data_manager)
            )
            process.start()
            self.processes[symbol] = process
            self.logger.info(f"Started processor for {symbol}")
    
    def stop_processor(self, symbol: str):
        """
        Stop indicator processor for a symbol
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.processes:
            process = self.processes[symbol]
            process.terminate()
            process.join(timeout=5)
            del self.processes[symbol]
            self.logger.info(f"Stopped processor for {symbol}")
    
    def update_symbols(self, symbols: List[str]):
        """
        Update active symbol processors
        
        Args:
            symbols: List of trading symbols
        """
        current_symbols = set(self.processes.keys())
        new_symbols = set(symbols)
        
        # Stop processors for removed symbols
        for symbol in current_symbols - new_symbols:
            self.stop_processor(symbol)
        
        # Start processors for new symbols
        for symbol in new_symbols - current_symbols:
            self.start_processor(symbol)
    
    def stop_all(self):
        """Stop all processors"""
        for symbol in list(self.processes.keys()):
            self.stop_processor(symbol)
        self.logger.info("All processors stopped")