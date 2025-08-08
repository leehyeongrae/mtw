"""
Binance REST API client
"""
import time
import asyncio
from typing import List, Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.core.candle_manager import CandleManager

class BinanceREST:
    """Binance REST API client with rate limiting"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("binance_rest")
        self.data_manager = data_manager
        self.client = self._init_client()
        self.interval = config.rest_api_interval  # 3 seconds between requests
        self.last_request_time = 0
        self.candle_managers = {}
        
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
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.interval:
            sleep_time = self.interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_klines(self, symbol: str, limit: int = 500) -> Optional[List]:
        """
        Fetch klines for a symbol
        
        Args:
            symbol: Trading symbol
            limit: Number of candles to fetch
            
        Returns:
            List of klines or None
        """
        self._rate_limit()
        
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval='15m',
                limit=limit
            )
            self.logger.debug(f"Fetched {len(klines)} klines for {symbol}")
            return klines
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error fetching klines for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
    
    def fetch_recent_kline(self, symbol: str) -> Optional[Dict]:
        """
        Fetch the most recent closed kline
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict of kline data or None
        """
        self._rate_limit()
        
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval='15m',
                limit=2  # Get last 2 to ensure we get the closed one
            )
            
            if klines and len(klines) >= 2:
                # Return the second-to-last (completed) candle
                kline = klines[-2]
                return {
                    't': kline[0],  # Open time
                    'T': kline[6],  # Close time
                    'o': kline[1],  # Open
                    'h': kline[2],  # High
                    'l': kline[3],  # Low
                    'c': kline[4],  # Close
                    'v': kline[5],  # Volume
                    'q': kline[7],  # Quote volume
                    'n': kline[8],  # Number of trades
                    'x': True       # Is closed
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching recent kline for {symbol}: {e}")
            return None
    
    async def update_all_symbols(self):
        """Update all symbols sequentially with rate limiting"""
        symbols = self.data_manager.get_symbol_list()
        
        for symbol in symbols:
            if not self.data_manager.is_running():
                break
            
            # Initialize candle manager if needed
            if symbol not in self.candle_managers:
                self.candle_managers[symbol] = CandleManager(symbol)
            
            candle_manager = self.candle_managers[symbol]
            
            # Fetch full data
            klines = self.fetch_klines(symbol)
            
            if klines:
                df = candle_manager.format_rest_candles(klines)
                
                if not df.empty:
                    # Update data manager
                    self.data_manager.update_candles(symbol, historical_df=df)
                    self.logger.info(f"Updated {symbol} with {len(df)} candles")
    
    async def handle_completed_candle(self, symbol: str):
        """
        Handle a completed candle notification
        
        Args:
            symbol: Trading symbol
        """
        # Wait for candle to be fully closed
        await asyncio.sleep(config.candle_confirmation_delay)
        
        # Fetch the completed candle
        recent_kline = self.fetch_recent_kline(symbol)
        
        if recent_kline:
            # Get current historical data
            historical_df, _ = self.data_manager.get_candles(symbol)
            
            if historical_df is not None:
                if symbol not in self.candle_managers:
                    self.candle_managers[symbol] = CandleManager(symbol)
                
                candle_manager = self.candle_managers[symbol]
                
                # Merge the new candle
                updated_df = candle_manager.merge_candle(historical_df, recent_kline)
                
                # Check if full refresh needed
                if candle_manager.should_refresh_all(updated_df):
                    self.logger.warning(f"Full refresh needed for {symbol}")
                    klines = self.fetch_klines(symbol)
                    if klines:
                        updated_df = candle_manager.format_rest_candles(klines)
                
                # Update data manager
                self.data_manager.update_candles(symbol, historical_df=updated_df)
                self.logger.debug(f"Updated {symbol} with completed candle")
    
    async def run(self):
        """Main REST API loop - 간소화"""
        self.logger.info("Starting REST API client")
        
        # Initial load of all symbols
        await self.update_all_symbols()
        
        # 큐 방식 제거하고 단순 주기적 업데이트로 변경
        while self.data_manager.is_running():
            try:
                # 주기적으로 모든 심볼 업데이트 (큐 대신)
                await asyncio.sleep(10)  # 10초마다 업데이트
                
            except Exception as e:
                self.logger.error(f"Error in REST API loop: {e}")
                await asyncio.sleep(1)
        
        self.logger.info("REST API client stopped")