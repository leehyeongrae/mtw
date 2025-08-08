"""
Symbol list management based on volume and listing age
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import ccxt
from src.utils.logger import get_logger
from src.utils.config import config

class SymbolManager:
    """Manages trading symbol selection"""
    
    def __init__(self):
        self.logger = get_logger("symbol_manager")
        self.exchange = self._init_exchange()
        self.min_listing_days = 7
        self.quote_currency = 'USDT'
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange instance"""
        exchange_config = {
            'apiKey': config.binance_api_key,
            'secret': config.binance_api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Use futures market
            }
        }
        
        if config.binance_testnet:
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            }
        
        return ccxt.binance(exchange_config)
    
    def get_top_symbols(self) -> List[str]:
        """
        Get top symbols by 24h volume
        
        Returns:
            List of symbol strings
        """
        try:
            # Load markets
            markets = self.exchange.load_markets()
            
            # Get tickers for volume data
            tickers = self.exchange.fetch_tickers()
            
            # Filter and sort symbols
            eligible_symbols = []
            
            for symbol, ticker in tickers.items():
                # Check if USDT pair
                if not symbol.endswith(f'/{self.quote_currency}'):
                    continue
                
                # Check if futures market
                market = markets.get(symbol)
                if not market or market['type'] != 'future':
                    continue
                
                # Check listing age (if available)
                if not self._check_listing_age(symbol):
                    continue
                
                # Get 24h volume in USDT
                volume_usdt = ticker.get('quoteVolume', 0)
                if volume_usdt > 0:
                    eligible_symbols.append({
                        'symbol': symbol.replace('/', ''),  # Remove slash for Binance format
                        'volume': volume_usdt
                    })
            
            # Sort by volume
            eligible_symbols.sort(key=lambda x: x['volume'], reverse=True)
            
            # Get top N symbols
            top_symbols = [s['symbol'] for s in eligible_symbols[:config.symbol_count]]
            
            self.logger.info(f"Selected top {len(top_symbols)} symbols: {top_symbols}")
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting top symbols: {e}")
            # Return default symbols as fallback
            return self._get_default_symbols()
    
    def _check_listing_age(self, symbol: str) -> bool:
        """
        Check if symbol has been listed for minimum days
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: True if old enough
        """
        try:
            # For now, implement a simple check
            # In production, you would query the actual listing date
            
            # Skip new listings (simplified check)
            excluded_new = ['NOTUSDT', 'AIUSDT', 'ACEUSDT']  # Example new listings
            base_symbol = symbol.split('/')[0] + self.quote_currency
            
            if base_symbol in excluded_new:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking listing age for {symbol}: {e}")
            return True  # Default to including the symbol
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbol list as fallback"""
        defaults = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT'
        ]
        return defaults[:config.symbol_count]
    
    def update_symbol_list(self) -> List[str]:
        """
        Update and return the current symbol list
        
        Returns:
            List of trading symbols
        """
        return self.get_top_symbols()