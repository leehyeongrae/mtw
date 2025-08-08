"""
Binance WebSocket client for real-time data
"""
import json
import asyncio
import websocket
from typing import List, Dict, Optional
from threading import Thread
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.core.candle_manager import CandleManager

class BinanceWebSocket:
    """Binance WebSocket client for multiple symbols"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("binance_ws")
        self.data_manager = data_manager
        self.ws = None
        self.running = False
        self.symbols = []
        self.candle_managers = {}
        self.reconnect_delay = config.websocket_reconnect_delay
        
    def _get_ws_url(self, symbols: List[str]) -> str:
        """
        Build WebSocket URL for multiple symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            WebSocket URL string
        """
        streams = []
        for symbol in symbols:
            # Convert to lowercase for Binance WebSocket
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@kline_15m")
        
        stream_names = '/'.join(streams)
        return f"{config.binance_ws_url}/stream?streams={stream_names}"
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'data' in data:
                stream_data = data['data']
                
                # Extract symbol from stream name
                if 's' in stream_data:
                    symbol = stream_data['s']  # Symbol in uppercase
                    
                    # Initialize candle manager if needed
                    if symbol not in self.candle_managers:
                        self.candle_managers[symbol] = CandleManager(symbol)
                    
                    candle_manager = self.candle_managers[symbol]
                    
                    # Format the candle data
                    current_candle = candle_manager.format_ws_candle(stream_data)
                    
                    if current_candle:
                        # Update current candle in data manager
                        self.data_manager.update_candles(symbol, current_candle=current_candle)
                        
                        # Check if candle is closed
                        if current_candle.get('x', False):
                            self.logger.info(f"Candle closed for {symbol}")
                            # Notify REST API to fetch the completed candle
                            self.data_manager.add_rest_data({
                                'type': 'candle_closed',
                                'symbol': symbol,
                                'time': current_candle['T']
                            })
                        
                        # Add to WebSocket queue for indicator processing
                        self.data_manager.add_ws_data({
                            'symbol': symbol,
                            'candle': current_candle,
                            'timestamp': current_candle['T']
                        })
            
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.running = False
    
    def _on_open(self, ws):
        """Handle WebSocket open"""
        self.logger.info("WebSocket connection established")
    
    def connect(self, symbols: List[str]):
        """
        Connect to WebSocket with given symbols
        
        Args:
            symbols: List of trading symbols
        """
        self.symbols = symbols
        
        if not symbols:
            self.logger.warning("No symbols to connect")
            return
        
        url = self._get_ws_url(symbols)
        self.logger.info(f"Connecting to WebSocket for {len(symbols)} symbols")
        
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.running = True
    
    def start(self):
        """Start WebSocket connection in a thread"""
        if self.ws:
            ws_thread = Thread(target=self._run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            self.logger.info("WebSocket thread started")
    
    def _run_forever(self):
        """Run WebSocket with reconnection logic"""
        while self.data_manager.is_running():
            try:
                if self.ws:
                    self.ws.run_forever()
                
                if self.data_manager.is_running():
                    self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    asyncio.sleep(self.reconnect_delay)
                    
                    # Reconnect with current symbols
                    symbols = self.data_manager.get_symbol_list()
                    if symbols:
                        self.connect(symbols)
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                asyncio.sleep(self.reconnect_delay)
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
            self.logger.info("WebSocket connection closed")
    
    def update_symbols(self, symbols: List[str]):
        """
        Update symbol list and reconnect
        
        Args:
            symbols: New list of trading symbols
        """
        if set(symbols) != set(self.symbols):
            self.logger.info(f"Updating symbols from {self.symbols} to {symbols}")
            self.stop()
            asyncio.sleep(1)
            self.connect(symbols)
            self.start()
    
    async def run(self):
        """Main WebSocket loop"""
        self.logger.info("Starting WebSocket client")
        
        # Get initial symbol list
        symbols = self.data_manager.get_symbol_list()
        
        if symbols:
            self.connect(symbols)
            self.start()
        
        # Monitor for symbol list changes
        last_symbols = symbols
        
        while self.data_manager.is_running():
            try:
                # Check for symbol list updates
                current_symbols = self.data_manager.get_symbol_list()
                
                if current_symbols != last_symbols:
                    self.update_symbols(current_symbols)
                    last_symbols = current_symbols
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket loop: {e}")
                await asyncio.sleep(1)
        
        self.stop()
        self.logger.info("WebSocket client stopped")