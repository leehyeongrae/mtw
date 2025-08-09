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
        캔들 데이터 초기화 - 개선 버전
        마지막 캔들이 현재 진행 중이면 제외
        
        Args:
            symbol: 심볼명
            candle_data: REST API로 받은 캔들 데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            async with self.lock:
                if not candle_data:
                    self.logger.error(f"{symbol}: 빈 캔들 데이터")
                    return False
                
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
                
                # 현재 시간 확인
                import time
                current_time_ms = int(time.time() * 1000)
                current_time = pd.to_datetime(current_time_ms, unit='ms')
                
                # 완성된 캔들만 필터링 (close_time이 현재 시간보다 이전)
                completed_candles = []
                for idx in range(len(df)):
                    if df.iloc[idx]['close_time'] < current_time:
                        completed_candles.append(idx)
                
                if completed_candles:
                    df = df.iloc[completed_candles].reset_index(drop=True)
                    self.logger.debug(f"{symbol}: 완성된 캔들 {len(df)}개 로드")
                else:
                    self.logger.warning(f"{symbol}: 완성된 캔들 없음")
                
                # 최근 N개만 유지
                if len(df) > config.candle_limit:
                    df = df.tail(config.candle_limit).reset_index(drop=True)
                
                # 저장
                self.candles[symbol] = df
                
                self.logger.info(
                    f"{symbol}: 캔들 초기화 완료 - "
                    f"캔들 수: {len(df)}, "
                    f"시작: {df.iloc[0]['open_time'].strftime('%Y-%m-%d %H:%M') if len(df) > 0 else 'N/A'}, "
                    f"종료: {df.iloc[-1]['open_time'].strftime('%Y-%m-%d %H:%M') if len(df) > 0 else 'N/A'}"
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"{symbol}: 캔들 초기화 실패 - {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        완성된 캔들 추가 (REST API 데이터) - 완전 수정 버전
        중복 체크 및 시간 순서 보장
        
        Args:
            symbol: 심볼명
            candle_data: REST API로 받은 완성된 캔들
            
        Returns:
            bool: 성공 여부
        """
        try:
            async with self.lock:
                if symbol not in self.candles:
                    self.logger.error(f"{symbol}: 캔들 데이터가 초기화되지 않음")
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
                
                # 기존 데이터프레임
                existing_df = self.candles[symbol]
                
                # 새 캔들의 open_time
                new_open_time = new_candle.iloc[0]['open_time']
                
                # 중복 체크: 이미 존재하는 캔들인지 확인
                if len(existing_df) > 0:
                    # 마지막 캔들의 open_time과 비교
                    last_open_time = existing_df.iloc[-1]['open_time']
                    
                    # 이미 존재하는 캔들이면 무시
                    if new_open_time <= last_open_time:
                        self.logger.debug(f"{symbol}: 이미 존재하는 캔들 (open_time: {new_open_time})")
                        return True
                    
                    # 시간 간격 체크 (15분 = 900초)
                    time_diff = (new_open_time - last_open_time).total_seconds()
                    expected_interval = 15 * 60  # 15분
                    
                    if abs(time_diff - expected_interval) > 60:  # 1분 이상 차이나면 경고
                        self.logger.warning(
                            f"{symbol}: 캔들 간격 이상 - "
                            f"예상: {expected_interval}초, 실제: {time_diff}초"
                        )
                
                # 새 캔들 추가
                self.candles[symbol] = pd.concat([
                    existing_df, new_candle
                ], ignore_index=True)
                
                # 시간순 정렬 (혹시 모를 순서 문제 방지)
                self.candles[symbol] = self.candles[symbol].sort_values('open_time').reset_index(drop=True)
                
                # 최근 N개만 유지
                if len(self.candles[symbol]) > config.candle_limit:
                    self.candles[symbol] = self.candles[symbol].tail(config.candle_limit).reset_index(drop=True)
                
                self.logger.info(
                    f"{symbol}: 새 캔들 추가 완료 - "
                    f"open_time: {new_open_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"전체 캔들 수: {len(self.candles[symbol])}"
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"{symbol}: 캔들 추가 실패 - {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        분석용 캔들 데이터 반환 - 수정 버전
        REST 데이터(완성된 캔들) + WebSocket 데이터(현재 진행 캔들) 조합
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[pd.DataFrame]: 캔들 데이터 또는 None
        """
        async with self.lock:
            if symbol not in self.candles:
                return None
            
            # REST 데이터 복사 (완성된 캔들들만)
            df = self.candles[symbol].copy()
            
            # 현재 진행 캔들이 있고 아직 닫히지 않았으면 추가
            if symbol in self.current_candles:
                current = self.current_candles[symbol]
                
                # WebSocket 캔들이 아직 진행 중인지 확인
                if not current.get('is_closed', False):
                    # 마지막 REST 캔들의 close_time과 현재 WebSocket 캔들의 open_time 비교
                    if len(df) > 0:
                        last_rest_close_time = df.iloc[-1]['close_time']
                        current_open_time = pd.to_datetime(current['open_time'], unit='ms')
                        
                        # REST의 마지막 캔들이 이미 완성된 캔들인지 확인
                        # close_time이 현재 WebSocket 캔들의 open_time보다 이전이면 겹치지 않음
                        if last_rest_close_time < current_open_time:
                            # 현재 진행 중인 WebSocket 캔들 추가
                            temp_row = pd.DataFrame([{
                                'open_time': current_open_time,
                                'open': current['open'],
                                'high': current['high'],
                                'low': current['low'],
                                'close': current['close'],
                                'volume': current['volume'],
                                'close_time': pd.to_datetime(current['close_time'], unit='ms')
                            }])
                            
                            df = pd.concat([df, temp_row], ignore_index=True)
                        else:
                            # REST의 마지막 캔들이 현재 진행 중인 캔들과 같은 시간대
                            # WebSocket 데이터로 교체
                            df.iloc[-1] = {
                                'open_time': current_open_time,
                                'open': current['open'],
                                'high': current['high'],
                                'low': current['low'],
                                'close': current['close'],
                                'volume': current['volume'],
                                'close_time': pd.to_datetime(current['close_time'], unit='ms'),
                                'quote_volume': df.iloc[-1]['quote_volume'] if 'quote_volume' in df.columns else 0,
                                'trades': df.iloc[-1]['trades'] if 'trades' in df.columns else 0,
                                'taker_buy_base': df.iloc[-1]['taker_buy_base'] if 'taker_buy_base' in df.columns else 0,
                                'taker_buy_quote': df.iloc[-1]['taker_buy_quote'] if 'taker_buy_quote' in df.columns else 0,
                                'ignore': df.iloc[-1]['ignore'] if 'ignore' in df.columns else 0
                            }
            
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