"""
Candle data management and validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from src.utils.logger import get_logger
from src.utils.config import config

class CandleManager:
    """Manages candle data integrity and validation"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = get_logger(f"candle_manager_{symbol}")
        self.timeframe = '15m'
        self.timeframe_seconds = 15 * 60  # 15 minutes in seconds
        self.max_candles = 500
        self.validation_candles = 50
        
    def validate_candle_intervals(self, df: pd.DataFrame) -> bool:
        """
        Validate that recent candles have correct 15-minute intervals
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            bool: True if intervals are valid
        """
        if len(df) < self.validation_candles:
            return True
        
        # Check last 50 candles
        recent_df = df.tail(self.validation_candles).copy()
        recent_df['open_time'] = pd.to_datetime(recent_df['open_time'], unit='ms')
        
        # Calculate time differences
        time_diffs = recent_df['open_time'].diff().dropna()
        expected_diff = timedelta(minutes=15)
        
        # Allow 1 second tolerance
        tolerance = timedelta(seconds=1)
        
        for diff in time_diffs:
            if abs(diff - expected_diff) > tolerance:
                self.logger.warning(
                    f"Invalid interval detected: {diff} (expected {expected_diff})"
                )
                return False
        
        return True
    
    def merge_candle(self, historical_df: pd.DataFrame, new_candle: Dict) -> pd.DataFrame:
        """
        Merge a new completed candle into historical data
        
        Args:
            historical_df: Historical candles DataFrame
            new_candle: New candle to merge
            
        Returns:
            pd.DataFrame: Updated DataFrame
        """
        try:
            # Convert new candle to DataFrame row
            new_row = pd.DataFrame([{
                'open_time': new_candle['t'],
                'open': float(new_candle['o']),
                'high': float(new_candle['h']),
                'low': float(new_candle['l']),
                'close': float(new_candle['c']),
                'volume': float(new_candle['v']),
                'close_time': new_candle['T'],
                'quote_volume': float(new_candle['q']),
                'trades': int(new_candle['n'])
            }])
            
            # Check if candle already exists
            if len(historical_df) > 0:
                last_time = historical_df.iloc[-1]['open_time']
                if new_candle['t'] <= last_time:
                    self.logger.debug(f"Candle already exists: {new_candle['t']}")
                    return historical_df
            
            # Append new candle
            updated_df = pd.concat([historical_df, new_row], ignore_index=True)
            
            # Sort by time and remove duplicates
            updated_df = updated_df.sort_values('open_time').drop_duplicates('open_time')
            
            # Keep only max_candles
            if len(updated_df) > self.max_candles:
                updated_df = updated_df.tail(self.max_candles)
            
            return updated_df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error merging candle: {e}")
            return historical_df
    
    def format_rest_candles(self, klines: List) -> pd.DataFrame:
        """
        Format REST API kline data to DataFrame
        
        Args:
            klines: Raw kline data from Binance
            
        Returns:
            pd.DataFrame: Formatted candle data
        """
        try:
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce', downcast='integer')
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce', downcast='integer')
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce', downcast='integer')
            
            # Drop unnecessary columns
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'quote_volume', 'trades']]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error formatting REST candles: {e}")
            return pd.DataFrame()
    
    def format_ws_candle(self, kline_data: Dict) -> Dict:
        """
        Format WebSocket kline data
        
        Args:
            kline_data: Raw kline data from WebSocket
            
        Returns:
            Dict: Formatted current candle
        """
        try:
            k = kline_data['k']
            return {
                't': k['t'],  # Open time
                'T': k['T'],  # Close time
                'o': float(k['o']),  # Open
                'h': float(k['h']),  # High
                'l': float(k['l']),  # Low
                'c': float(k['c']),  # Close
                'v': float(k['v']),  # Volume
                'q': float(k['q']),  # Quote volume
                'n': int(k['n']),    # Number of trades
                'x': k['x']  # Is closed
            }
        except Exception as e:
            self.logger.error(f"Error formatting WebSocket candle: {e}")
            return {}
    
    def should_refresh_all(self, df: pd.DataFrame) -> bool:
        """
        Check if all candles should be refreshed
        
        Args:
            df: Current candle DataFrame
            
        Returns:
            bool: True if refresh needed
        """
        if df is None or len(df) < self.validation_candles:
            return True
        
        if not self.validate_candle_intervals(df):
            self.logger.warning(f"Invalid intervals detected, refresh needed")
            return True
        
        # Check if we have recent data
        if len(df) > 0:
            last_candle_time = pd.to_datetime(df.iloc[-1]['close_time'], unit='ms')
            current_time = datetime.now()
            
            # If last candle is more than 30 minutes old, refresh
            if (current_time - last_candle_time).total_seconds() > 1800:
                self.logger.warning(f"Stale data detected, refresh needed")
                return True
        
        return False