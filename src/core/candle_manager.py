"""
캔들 데이터 중앙 관리 모듈 (DRY, KISS 원칙 적용)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from src.utils.logger import get_logger
from src.utils.config import config

class CandleManager:
    """캔들 데이터 중앙 관리 클래스"""
    
    def __init__(self):
        self.logger = get_logger("candle_manager")
        self.candles: Dict[str, pd.DataFrame] = {}  # 심볼별 캔들 데이터
        self.current_candles: Dict[str, Dict] = {}  # 심볼별 현재 진행 캔들
        self.lock = asyncio.Lock()  # 스레드 안전성
        
    async def initialize_candles(self, symbol: str, candle_data: List[List]) -> bool:
        """
        캔들 데이터 초기화
        
        Args:
            symbol: 심볼명
            candle_data: REST API로 받은 캔들 데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            async with self.lock:
                # 데이터프레임 생성
                df = pd.DataFrame(candle_data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # 타입 변환
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # 최근 N개만 유지
                df = df.tail(config.candle_limit).reset_index(drop=True)
                self.candles[symbol] = df
                
                self.logger.info(f"{symbol}: 캔들 데이터 초기화 완료 ({len(df)}개)")
                return True
                
        except Exception as e:
            self.logger.error(f"{symbol}: 캔들 초기화 실패 - {e}")
            return False
    
    async def update_current_candle(self, symbol: str, candle_data: Dict) -> None:
        """
        현재 진행 캔들 업데이트 (웹소켓 데이터)
        
        Args:
            symbol: 심볼명
            candle_data: 웹소켓 캔들 데이터
        """
        async with self.lock:
            self.current_candles[symbol] = {
                'open_time': candle_data.get('t'),
                'open': float(candle_data.get('o', 0)),
                'high': float(candle_data.get('h', 0)),
                'low': float(candle_data.get('l', 0)),
                'close': float(candle_data.get('c', 0)),
                'volume': float(candle_data.get('v', 0)),
                'close_time': candle_data.get('T'),
                'is_closed': candle_data.get('x', False)
            }
    
    async def add_completed_candle(self, symbol: str, candle_data: List) -> bool:
        """
        완성된 캔들 추가 (REST API 데이터)
        
        Args:
            symbol: 심볼명
            candle_data: REST API로 받은 완성된 캔들
            
        Returns:
            bool: 성공 여부
        """
        try:
            async with self.lock:
                if symbol not in self.candles:
                    return False
                
                # 새 캔들 데이터 생성
                new_candle = pd.DataFrame([candle_data], columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # 타입 변환
                new_candle['open_time'] = pd.to_datetime(new_candle['open_time'], unit='ms')
                new_candle['close_time'] = pd.to_datetime(new_candle['close_time'], unit='ms')
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    new_candle[col] = new_candle[col].astype(float)
                
                # 기존 데이터와 병합
                self.candles[symbol] = pd.concat([
                    self.candles[symbol], new_candle
                ], ignore_index=True)
                
                # 최근 N개만 유지
                self.candles[symbol] = self.candles[symbol].tail(config.candle_limit).reset_index(drop=True)
                
                # 무결성 검증
                if not await self.validate_candle_integrity(symbol):
                    self.logger.warning(f"{symbol}: 캔들 무결성 검증 실패")
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"{symbol}: 캔들 추가 실패 - {e}")
            return False
    
    async def validate_candle_integrity(self, symbol: str) -> bool:
        """
        캔들 데이터 무결성 검증 (15분 간격 확인)
        
        Args:
            symbol: 심볼명
            
        Returns:
            bool: 무결성 여부
        """
        try:
            if symbol not in self.candles:
                return False
            
            df = self.candles[symbol]
            
            # 최근 N개 캔들 검증
            check_count = min(config.candle_check_count, len(df) - 1)
            
            if check_count <= 0:
                return True
            
            # 시간 간격 검증 (15분 = 900000ms)
            expected_interval = 15 * 60 * 1000  # milliseconds
            
            for i in range(len(df) - check_count, len(df) - 1):
                time_diff = (df.iloc[i + 1]['open_time'] - df.iloc[i]['close_time']).total_seconds() * 1000
                
                # 허용 오차 1초
                if abs(time_diff - 1) > 1000:
                    self.logger.debug(f"{symbol}: 캔들 간격 이상 감지 (인덱스 {i}, 차이: {time_diff}ms)")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"{symbol}: 무결성 검증 오류 - {e}")
            return False
    
    async def get_candles_for_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        분석용 캔들 데이터 반환
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[pd.DataFrame]: 캔들 데이터 또는 None
        """
        async with self.lock:
            if symbol not in self.candles:
                return None
            
            # 현재 진행 캔들이 있으면 임시로 추가
            df = self.candles[symbol].copy()
            
            if symbol in self.current_candles and not self.current_candles[symbol].get('is_closed', False):
                current = self.current_candles[symbol]
                
                # 현재 캔들을 임시로 추가
                temp_row = pd.DataFrame([{
                    'open_time': pd.to_datetime(current['open_time'], unit='ms'),
                    'open': current['open'],
                    'high': current['high'],
                    'low': current['low'],
                    'close': current['close'],
                    'volume': current['volume'],
                    'close_time': pd.to_datetime(current['close_time'], unit='ms')
                }])
                
                df = pd.concat([df, temp_row], ignore_index=True)
            
            return df
    
    async def needs_full_refresh(self, symbol: str) -> bool:
        """
        전체 새로고침 필요 여부 확인
        
        Args:
            symbol: 심볼명
            
        Returns:
            bool: 새로고침 필요 여부
        """
        if symbol not in self.candles:
            return True
        
        # 무결성 검증 실패시 새로고침 필요
        return not await self.validate_candle_integrity(symbol)
    
    def get_latest_candles(self, symbol: str, count: int = 5) -> Optional[pd.DataFrame]:
        """
        최근 N개 캔들 반환 (동기 메서드)
        
        Args:
            symbol: 심볼명
            count: 반환할 캔들 수
            
        Returns:
            Optional[pd.DataFrame]: 최근 캔들 데이터
        """
        if symbol not in self.candles:
            return None
        
        return self.candles[symbol].tail(count)