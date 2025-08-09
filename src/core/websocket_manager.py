"""
웹소켓 중앙 관리 모듈 (KISS 원칙 적용)
"""
import asyncio
import json
from typing import Dict, List, Callable, Optional
import websockets
from src.utils.logger import get_logger
from src.utils.config import config

class WebSocketManager:
    """웹소켓 중앙 관리 클래스"""
    
    def __init__(self, candle_manager):
        self.logger = get_logger("websocket_manager")
        self.candle_manager = candle_manager
        self.symbols: List[str] = []
        self.ws = None
        self.running = False
        self.callbacks: Dict[str, List[Callable]] = {}
        
    async def start(self, symbols: List[str]) -> None:
        """
        웹소켓 연결 시작
        
        Args:
            symbols: 구독할 심볼 리스트
        """
        self.symbols = symbols
        self.running = True
        
        # 스트림 생성 (kline_15m)
        streams = [f"{symbol.lower()}@kline_{config.timeframe}" for symbol in symbols]
        url = f"{config.get_binance_ws_url()}/stream?streams={'/'.join(streams)}"
        
        self.logger.info(f"웹소켓 연결 시작: {len(symbols)}개 심볼")
        
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    self.logger.info("웹소켓 연결 성공")
                    
                    # 핑퐁 처리 태스크
                    ping_task = asyncio.create_task(self._ping_loop())
                    
                    try:
                        async for message in ws:
                            if not self.running:
                                break
                            
                            await self._handle_message(message)
                    finally:
                        ping_task.cancel()
                        
            except Exception as e:
                self.logger.error(f"웹소켓 오류: {e}")
                if self.running:
                    await asyncio.sleep(5)  # 재연결 대기
                    self.logger.info("웹소켓 재연결 시도...")
    
    async def _ping_loop(self) -> None:
        """핑퐁 처리 (연결 유지)"""
        while self.running:
            try:
                if self.ws:
                    pong = await self.ws.ping()
                    await asyncio.wait_for(pong, timeout=10)
                await asyncio.sleep(30)  # 30초마다 핑
            except Exception as e:
                self.logger.debug(f"핑 실패: {e}")
                break
    
    async def _handle_message(self, message: str) -> None:
        """
        웹소켓 메시지 처리 - 실시간 신호 생성 추가
        
        Args:
            message: 수신된 메시지
        """
        try:
            data = json.loads(message)
            
            # 스트림 데이터 처리
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                kline_data = data['data']
                
                if 'k' in kline_data:
                    candle = kline_data['k']
                    symbol = candle['s']
                    
                    # 캔들 매니저 업데이트
                    await self.candle_manager.update_current_candle(symbol, candle)
                    
                    # 캔들 종료 확인
                    if candle.get('x', False):
                        self.logger.info(f"{symbol}: 캔들 종료 감지")
                        await self._trigger_callbacks('candle_closed', {
                            'symbol': symbol,
                            'candle': candle
                        })
                    else:
                        # 실시간 업데이트 콜백 (진입/청산 신호 검사용)
                        await self._trigger_callbacks('realtime_update', {
                            'symbol': symbol,
                            'candle': candle,
                            'current_price': float(candle.get('c', 0))
                        })
                    
                    # 기존 콜백 유지
                    await self._trigger_callbacks('candle_update', {
                        'symbol': symbol,
                        'candle': candle
                    })
                    
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")
    
    async def _trigger_callbacks(self, event: str, data: Dict) -> None:
        """
        이벤트 콜백 실행
        
        Args:
            event: 이벤트 타입
            data: 이벤트 데이터
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"콜백 실행 오류: {e}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        이벤트 콜백 등록
        
        Args:
            event: 이벤트 타입
            callback: 콜백 함수
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    async def stop(self) -> None:
        """웹소켓 연결 종료"""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.logger.info("웹소켓 연결 종료")