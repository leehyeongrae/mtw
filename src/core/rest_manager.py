"""
REST API 중앙 관리 모듈 (DRY 원칙 적용)
"""
import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode
from src.utils.logger import get_logger
from src.utils.config import config

class RestManager:
    """REST API 중앙 관리 클래스"""
    
    def __init__(self):
        self.logger = get_logger("rest_manager")
        self.base_url = config.get_binance_base_url()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """세션 초기화"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            self.logger.info("REST API 세션 초기화 완료")
    
    async def close(self) -> None:
        """세션 종료"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _generate_signature(self, params: Dict) -> str:
        """
        API 서명 생성
        
        Args:
            params: 요청 파라미터
            
        Returns:
            str: 서명
        """
        query_string = urlencode(params)
        return hmac.new(
            config.binance_api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _rate_limit(self) -> None:
        """레이트 리밋 관리 (3초 간격)"""
        async with self.request_lock:
            current_time = time.time()
            time_diff = current_time - self.last_request_time
            
            if time_diff < config.rest_api_interval:
                await asyncio.sleep(config.rest_api_interval - time_diff)
            
            self.last_request_time = time.time()
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                       signed: bool = False) -> Optional[Any]:
        """
        API 요청 실행
        
        Args:
            method: HTTP 메서드
            endpoint: API 엔드포인트
            params: 요청 파라미터
            signed: 서명 필요 여부
            
        Returns:
            Optional[Any]: 응답 데이터
        """
        if not self.session:
            await self.initialize()
        
        # 레이트 리밋 적용
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        # 서명 처리
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        headers = {
            'X-MBX-APIKEY': config.binance_api_key
        }
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"API 오류 {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"요청 실패: {e}")
            return None
    
    async def get_klines(self, symbol: str, limit: int = 500) -> Optional[List]:
        """
        캔들 데이터 조회
        
        Args:
            symbol: 심볼명
            limit: 조회할 캔들 수
            
        Returns:
            Optional[List]: 캔들 데이터
        """
        params = {
            'symbol': symbol,
            'interval': config.timeframe,
            'limit': limit
        }
        
        result = await self._request('GET', '/fapi/v1/klines', params)
        return result
    
    async def get_latest_closed_candle(self, symbol: str) -> Optional[List]:
        """
        최근 완성된 캔들 1개 조회
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[List]: 캔들 데이터
        """
        # 지연 시간 적용 (완전히 닫힌 캔들 확보)
        await asyncio.sleep(config.rest_api_delay)
        
        params = {
            'symbol': symbol,
            'interval': config.timeframe,
            'limit': 2  # 현재 진행중 + 완성된 캔들
        }
        
        result = await self._request('GET', '/fapi/v1/klines', params)
        
        if result and len(result) >= 2:
            # 두 번째가 완성된 캔들
            return result[-2]
        
        return None
    
    async def get_exchange_info(self) -> Optional[Dict]:
        """거래소 정보 조회"""
        return await self._request('GET', '/fapi/v1/exchangeInfo')
    
    async def get_24hr_ticker(self) -> Optional[List]:
        """24시간 티커 정보 조회"""
        return await self._request('GET', '/fapi/v1/ticker/24hr')
    
    async def get_account_balance(self) -> Optional[Dict]:
        """계정 잔고 조회"""
        return await self._request('GET', '/fapi/v2/balance', signed=True)
    
    async def get_positions(self) -> Optional[List]:
        """현재 포지션 조회"""
        return await self._request('GET', '/fapi/v2/positionRisk', signed=True)
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                          order_type: str = 'MARKET') -> Optional[Dict]:
        """
        주문 실행
        
        Args:
            symbol: 심볼명
            side: BUY/SELL
            quantity: 수량
            order_type: 주문 타입
            
        Returns:
            Optional[Dict]: 주문 결과
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if order_type == 'MARKET':
            params['newOrderRespType'] = 'RESULT'
        
        return await self._request('POST', '/fapi/v1/order', params, signed=True)
    
    async def set_leverage(self, symbol: str, leverage: int) -> Optional[Dict]:
        """
        레버리지 설정
        
        Args:
            symbol: 심볼명
            leverage: 레버리지 배수
            
        Returns:
            Optional[Dict]: 설정 결과
        """
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        
        return await self._request('POST', '/fapi/v1/leverage', params, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> Optional[Dict]:
        """
        심볼의 모든 주문 취소
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[Dict]: 취소 결과
        """
        params = {'symbol': symbol}
        return await self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)