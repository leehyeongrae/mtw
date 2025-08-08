"""
Binance WebSocket client for real-time data - 개선된 버전
"""
import json
import asyncio
import websocket
import time
from typing import List, Dict, Optional
from threading import Thread
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.core.candle_manager import CandleManager

class BinanceWebSocket:
    """Binance WebSocket client for multiple symbols - 안정성 개선"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("binance_ws")
        self.data_manager = data_manager
        self.ws = None
        self.running = False
        self.symbols = []
        self.candle_managers = {}
        self.reconnect_delay = config.websocket_reconnect_delay
        self.max_symbols_per_connection = 200  # 바이낸스 제한
        
    def _get_ws_url(self, symbols: List[str]) -> str:
        """
        Build WebSocket URL for multiple symbols - 바이낸스 공식 형식
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            WebSocket URL string
        """
        if not symbols:
            raise ValueError("No symbols provided for WebSocket connection")
        
        # 심볼 수 제한 확인
        if len(symbols) > self.max_symbols_per_connection:
            self.logger.warning(f"Too many symbols ({len(symbols)}), limiting to {self.max_symbols_per_connection}")
            symbols = symbols[:self.max_symbols_per_connection]
        
        # 스트림 이름 생성 (소문자로 변환)
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@kline_15m")
        
        # 바이낸스 공식 다중 스트림 URL 형식
        if config.binance_testnet:
            base_url = "wss://testnet.binance.vision/ws-api/v3"
        else:
            base_url = "wss://stream.binance.com:9443/ws"
        
        # 다중 스트림을 위한 올바른 URL 구성
        stream_names = '/'.join(streams)
        if len(streams) == 1:
            # 단일 스트림
            url = f"{base_url}/{stream_names}"
        else:
            # 다중 스트림 - combined stream 방식 사용
            combined_streams = '/'.join(streams)
            if config.binance_testnet:
                url = f"wss://testnet.binance.vision/stream?streams={combined_streams}"
            else:
                url = f"wss://stream.binance.com:9443/stream?streams={combined_streams}"
        
        self.logger.info(f"WebSocket URL: {url}")
        return url
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # 다중 스트림 형식 처리
            if 'stream' in data and 'data' in data:
                # Combined stream format
                stream_data = data['data']
                stream_name = data['stream']
                
                # 스트림 이름에서 심볼 추출
                if '@kline_15m' in stream_name:
                    symbol = stream_name.replace('@kline_15m', '').upper()
                else:
                    self.logger.warning(f"Unknown stream format: {stream_name}")
                    return
                    
            elif 'k' in data:
                # 단일 스트림 형식
                stream_data = data
                symbol = data['k']['s']  # 심볼은 대문자로 제공됨
            else:
                self.logger.debug(f"Unknown message format: {data}")
                return
            
            # 심볼 검증
            if symbol not in self.symbols:
                self.logger.debug(f"Received data for unsubscribed symbol: {symbol}")
                return
            
            # 캔들 매니저 초기화
            if symbol not in self.candle_managers:
                self.candle_managers[symbol] = CandleManager(symbol)
            
            candle_manager = self.candle_managers[symbol]
            
            # 캔들 데이터 포맷팅
            current_candle = candle_manager.format_ws_candle(stream_data)
            
            if current_candle:
                # 데이터 매니저 업데이트
                self.data_manager.update_candles(symbol, current_candle=current_candle)
                
                # 캔들 완료 확인
                if current_candle.get('x', False):
                    self.logger.info(f"Candle closed for {symbol} at {current_candle.get('T')}")
                    # REST API에 완료된 캔들 처리 요청
                    self.data_manager.add_rest_data({
                        'type': 'candle_closed',
                        'symbol': symbol,
                        'time': current_candle['T']
                    })
                
                # WebSocket 큐에 추가 (지표 처리용)
                self.data_manager.add_ws_data({
                    'symbol': symbol,
                    'candle': current_candle,
                    'timestamp': current_candle.get('T', 0)
                })
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}, message: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        self.logger.error(f"WebSocket error: {error}")
        # 연결 오류 시 재연결 준비
        self.running = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        if close_status_code or close_msg:
            self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        else:
            self.logger.info("WebSocket connection closed normally")
        self.running = False
    
    def _on_open(self, ws):
        """Handle WebSocket open"""
        self.logger.info(f"WebSocket connection established for {len(self.symbols)} symbols")
        self.running = True
    
    def connect(self, symbols: List[str]):
        """
        Connect to WebSocket with given symbols
        
        Args:
            symbols: List of trading symbols
        """
        if not symbols:
            self.logger.warning("No symbols to connect")
            return False
        
        self.symbols = symbols
        
        try:
            url = self._get_ws_url(symbols)
            self.logger.info(f"Connecting to WebSocket for {len(symbols)} symbols")
            
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create WebSocket connection: {e}")
            return False
    
    def start(self):
        """Start WebSocket connection in a thread"""
        if self.ws:
            ws_thread = Thread(target=self._run_forever, daemon=True)
            ws_thread.start()
            self.logger.info("WebSocket thread started")
            return True
        return False
    
    def _run_forever(self):
        """Run WebSocket with reconnection logic - 동기 함수로 수정"""
        while self.data_manager.is_running():
            try:
                if self.ws:
                    self.logger.info("Starting WebSocket connection...")
                    self.ws.run_forever()
                
                # 재연결이 필요한 경우
                if self.data_manager.is_running():
                    self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)  # asyncio.sleep 대신 time.sleep 사용
                    
                    # 새 연결 생성
                    symbols = self.data_manager.get_symbol_list()
                    if symbols and self.connect(symbols):
                        continue
                    else:
                        self.logger.error("Failed to reconnect, retrying...")
                        time.sleep(5)
                        
            except Exception as e:
                self.logger.error(f"WebSocket error in run_forever: {e}")
                time.sleep(self.reconnect_delay)
    
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
            self.logger.info(f"Updating symbols from {len(self.symbols)} to {len(symbols)}")
            self.stop()
            time.sleep(2)  # 연결 정리 대기
            
            if self.connect(symbols):
                self.start()
            else:
                self.logger.error("Failed to reconnect with new symbols")
    
    async def run(self):
        """Main WebSocket loop"""
        self.logger.info("Starting WebSocket client")
        
        # 초기 심볼 리스트 가져오기
        symbols = self.data_manager.get_symbol_list()
        
        if symbols:
            if self.connect(symbols):
                self.start()
            else:
                self.logger.error("Failed to establish initial WebSocket connection")
        else:
            self.logger.warning("No symbols available for WebSocket connection")
        
        # 심볼 리스트 변경 모니터링
        last_symbols = symbols
        
        while self.data_manager.is_running():
            try:
                # 심볼 리스트 업데이트 확인
                current_symbols = self.data_manager.get_symbol_list()
                
                if current_symbols != last_symbols:
                    self.logger.info("Symbol list changed, updating WebSocket connection")
                    self.update_symbols(current_symbols)
                    last_symbols = current_symbols
                
                # 연결 상태 확인
                if not self.running and current_symbols:
                    self.logger.warning("WebSocket not running, attempting to reconnect")
                    if self.connect(current_symbols):
                        self.start()
                
                await asyncio.sleep(10)  # 10초마다 확인
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket main loop: {e}")
                await asyncio.sleep(5)
        
        self.stop()
        self.logger.info("WebSocket client stopped")