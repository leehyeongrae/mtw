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
        self.time_offset = 0  # 시간 오프셋 추가
        
    async def initialize(self) -> None:
        """세션 초기화 및 시간 동기화"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            self.logger.info("REST API 세션 초기화 완료")
            
            # 시간 동기화
            offset = await self.sync_time()
            if offset is not None:
                self.time_offset = offset
    
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
                    signed: bool = False, retry_count: int = 0) -> Optional[Any]:
        """
        API 요청 실행 - 시간 동기화 문제 해결
        
        Args:
            method: HTTP 메서드
            endpoint: API 엔드포인트
            params: 요청 파라미터
            signed: 서명 필요 여부
            retry_count: 재시도 횟수
            
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
            # 타임스탬프 생성 (밀리초)
            params['timestamp'] = int(time.time() * 1000)
            # recvWindow 추가 (10초로 설정 - 네트워크 지연 고려)
            params['recvWindow'] = 10000
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
                    
                    # 타임스탬프 에러 처리
                    if response.status == 400 and '-1021' in error_text and retry_count < 3:
                        self.logger.warning(f"타임스탬프 에러 감지, 재시도 {retry_count + 1}/3")
                        await asyncio.sleep(1)
                        # 재귀 호출로 재시도
                        return await self._request(method, endpoint, params, signed, retry_count + 1)
                    
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
        최근 완성된 캔들 1개 조회 - 수정 버전
        
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
            'limit': 3  # 여유있게 3개 가져옴
        }
        
        result = await self._request('GET', '/fapi/v1/klines', params)
        
        if result and len(result) >= 2:
            # 현재 시간 확인
            import time
            current_time_ms = int(time.time() * 1000)
            
            # 뒤에서부터 확인하여 완성된 캔들 찾기
            for i in range(len(result) - 1, -1, -1):
                candle = result[i]
                close_time = candle[6]  # close_time index
                
                # close_time이 현재 시간보다 이전이면 완성된 캔들
                if close_time < current_time_ms:
                    return candle
            
            # 모든 캔들이 진행 중이면 두 번째 캔들 반환 (보통 완성됨)
            if len(result) >= 2:
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

    async def set_margin_mode(self, symbol: str, margin_type: str = 'ISOLATED') -> Optional[Dict]:
        """
        마진 모드 설정 (새로운 메서드)
        
        Args:
            symbol: 심볼명
            margin_type: 'ISOLATED' or 'CROSSED'
            
        Returns:
            Optional[Dict]: 설정 결과
        """
        params = {
            'symbol': symbol,
            'marginType': margin_type
        }
        
        # 마진 모드 변경은 실패해도 괜찮음 (이미 설정되어 있을 수 있음)
        result = await self._request('POST', '/fapi/v1/marginType', params, signed=True)
        
        # 400 에러인 경우 이미 설정되어 있는 것
        if result is None:
            self.logger.debug(f"{symbol}: 마진 모드 이미 {margin_type}로 설정됨")
        
        return result    
    
    async def get_server_time(self) -> Optional[int]:
        """
        Binance 서버 시간 조회
        
        Returns:
            Optional[int]: 서버 시간 (밀리초)
        """
        result = await self._request('GET', '/fapi/v1/time')
        if result:
            return result.get('serverTime')
        return None    

    async def sync_time(self) -> Optional[int]:
        """
        시간 동기화 확인 및 오프셋 계산
        
        Returns:
            Optional[int]: 시간 오프셋 (밀리초)
        """
        try:
            import time
            
            # 로컬 시간
            local_time = int(time.time() * 1000)
            
            # 서버 시간
            server_time = await self.get_server_time()
            if not server_time:
                self.logger.error("서버 시간 조회 실패")
                return None
            
            # 시간 차이 계산
            time_offset = server_time - local_time
            
            self.logger.info(f"시간 동기화 - 로컬: {local_time}, 서버: {server_time}, 차이: {time_offset}ms")
            
            # 시간 차이가 5초 이상이면 경고
            if abs(time_offset) > 5000:
                self.logger.warning(f"시간 차이가 큽니다: {time_offset}ms. 시스템 시간을 확인하세요.")
            
            return time_offset
            
        except Exception as e:
            self.logger.error(f"시간 동기화 실패: {e}")
            return None