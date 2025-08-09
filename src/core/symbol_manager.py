"""
심볼 관리 모듈 (KISS 원칙 적용)
거래량 기준 상위 심볼 관리
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.utils.config import config

class SymbolManager:
    """심볼 관리 클래스"""
    
    def __init__(self, rest_manager):
        self.logger = get_logger("symbol_manager")
        self.rest_manager = rest_manager
        self.active_symbols: List[str] = []
        self.symbol_info: Dict[str, Dict] = {}
        
    async def update_top_symbols(self) -> List[str]:
        """
        거래량 기준 상위 심볼 업데이트
        
        Returns:
            List[str]: 업데이트된 심볼 리스트
        """
        try:
            # 거래소 정보 조회
            exchange_info = await self.rest_manager.get_exchange_info()
            if not exchange_info:
                self.logger.error("거래소 정보 조회 실패")
                return self.active_symbols
            
            # USDT 거래 가능 심볼 필터링
            usdt_symbols = {}
            for symbol_info in exchange_info.get('symbols', []):
                if (symbol_info.get('quoteAsset') == config.base_currency and 
                    symbol_info.get('status') == 'TRADING'):
                    
                    symbol = symbol_info['symbol']
                    # 상장일 확인 (contractType으로 판단)
                    contract_type = symbol_info.get('contractType', 'PERPETUAL')
                    
                    if contract_type == 'PERPETUAL':
                        usdt_symbols[symbol] = {
                            'symbol': symbol,
                            'pricePrecision': symbol_info.get('pricePrecision', 2),
                            'quantityPrecision': symbol_info.get('quantityPrecision', 3),
                            'minQty': float(next((f['minQty'] for f in symbol_info.get('filters', []) 
                                                 if f['filterType'] == 'LOT_SIZE'), 0.001))
                        }
            
            # 24시간 티커 정보로 거래량 확인
            tickers = await self.rest_manager.get_24hr_ticker()
            if not tickers:
                self.logger.error("티커 정보 조회 실패")
                return self.active_symbols
            
            # 거래량 기준 정렬
            volume_data = []
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                if symbol in usdt_symbols:
                    volume = float(ticker.get('quoteVolume', 0))
                    
                    # 최소 거래량 체크 (선택적)
                    if volume > 0:
                        volume_data.append({
                            **usdt_symbols[symbol],
                            'volume': volume
                        })
            
            # 거래량 기준 정렬
            volume_data.sort(key=lambda x: x['volume'], reverse=True)
            
            # 상위 N개 선택
            top_symbols = []
            for data in volume_data[:config.symbol_count]:
                symbol = data['symbol']
                top_symbols.append(symbol)
                self.symbol_info[symbol] = data
            
            # 변경 사항 로깅
            if set(top_symbols) != set(self.active_symbols):
                self.logger.info(f"심볼 리스트 업데이트: {top_symbols}")
            
            self.active_symbols = top_symbols
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"심볼 업데이트 오류: {e}")
            return self.active_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        심볼 정보 반환
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[Dict]: 심볼 정보
        """
        return self.symbol_info.get(symbol)
    
    def is_active_symbol(self, symbol: str) -> bool:
        """
        활성 심볼 여부 확인
        
        Args:
            symbol: 심볼명
            
        Returns:
            bool: 활성 여부
        """
        return symbol in self.active_symbols
    
    async def periodic_update(self, interval: int = 3600) -> None:
        """
        주기적 심볼 리스트 업데이트
        
        Args:
            interval: 업데이트 주기 (초)
        """
        while True:
            try:
                await asyncio.sleep(interval)
                await self.update_top_symbols()
                self.logger.info("심볼 리스트 주기적 업데이트 완료")
            except Exception as e:
                self.logger.error(f"주기적 업데이트 오류: {e}")