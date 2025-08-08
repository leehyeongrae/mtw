"""
Candle data management and validation - 개선된 버전
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from src.utils.logger import get_logger
from src.utils.config import config

class CandleManager:
    """Manages candle data integrity and validation - 강화된 검증"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = get_logger(f"candle_manager_{symbol}")
        self.timeframe = '15m'
        self.timeframe_seconds = 15 * 60  # 15 minutes in seconds
        self.max_candles = 500
        self.validation_candles = 50
        
        # 데이터 품질 검증 강화
        self.min_volume_threshold = 1000  # 최소 거래량
        self.max_price_change_percent = 20  # 최대 가격 변동률 (%)
        self.zero_volume_tolerance = 3  # 연속 0 볼륨 허용 개수
        
    def validate_candle_data_quality(self, df: pd.DataFrame) -> bool:
        """
        캔들 데이터 품질 검증 - 강화된 버전
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            bool: True if data quality is acceptable
        """
        try:
            if df.empty:
                return False
            
            # 1. 0 볼륨 캔들 연속성 검사
            zero_volume_count = 0
            max_consecutive_zero = 0
            
            for _, row in df.iterrows():
                if row['volume'] == 0:
                    zero_volume_count += 1
                    max_consecutive_zero = max(max_consecutive_zero, zero_volume_count)
                else:
                    zero_volume_count = 0
            
            if max_consecutive_zero > self.zero_volume_tolerance:
                self.logger.warning(
                    f"Too many consecutive zero volume candles: {max_consecutive_zero} "
                    f"(threshold: {self.zero_volume_tolerance})"
                )
                return False
            
            # 2. 동일한 OHLC 데이터 연속성 검사
            recent_candles = df.tail(10)  # 최근 10개 캔들 확인
            identical_candles = 0
            
            for i in range(1, len(recent_candles)):
                prev_row = recent_candles.iloc[i-1]
                curr_row = recent_candles.iloc[i]
                
                if (prev_row['open'] == curr_row['open'] and
                    prev_row['high'] == curr_row['high'] and
                    prev_row['low'] == curr_row['low'] and
                    prev_row['close'] == curr_row['close'] and
                    prev_row['volume'] == curr_row['volume']):
                    identical_candles += 1
            
            if identical_candles > 3:
                self.logger.warning(f"Too many identical candles detected: {identical_candles}")
                return False
            
            # 3. 극단적인 가격 변동 검사
            recent_candles = df.tail(20)
            for _, row in recent_candles.iterrows():
                if row['high'] > 0 and row['low'] > 0:
                    price_range_percent = ((row['high'] - row['low']) / row['low']) * 100
                    if price_range_percent > self.max_price_change_percent:
                        self.logger.warning(
                            f"Extreme price movement detected: {price_range_percent:.2f}% "
                            f"(H:{row['high']}, L:{row['low']})"
                        )
                        return False
            
            # 4. 기본 데이터 유효성 검사
            if recent_candles['volume'].sum() < self.min_volume_threshold:
                self.logger.warning(f"Very low total volume: {recent_candles['volume'].sum()}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating candle data quality: {e}")
            return False
    
    def detect_stale_data(self, df: pd.DataFrame) -> bool:
        """
        오래된 데이터 감지
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            bool: True if data is stale
        """
        try:
            if df.empty:
                return True
            
            # 최신 캔들 시간 확인
            last_candle_time = pd.to_datetime(df.iloc[-1]['close_time'], unit='ms')
            current_time = datetime.now()
            
            # 30분 이상 오래된 데이터는 stale로 판단
            time_diff = (current_time - last_candle_time).total_seconds()
            if time_diff > 1800:  # 30분
                self.logger.warning(
                    f"Stale data detected - Last candle: {last_candle_time}, "
                    f"Time diff: {time_diff/60:.1f} minutes"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting stale data: {e}")
            return True
    
    def validate_candle_intervals(self, df: pd.DataFrame) -> bool:
        """
        Validate that recent candles have correct 15-minute intervals
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            bool: True if intervals are valid
        """
        if len(df) < self.validation_candles:
            return True
        
        try:
            # 최근 캔들들 확인
            recent_df = df.tail(self.validation_candles).copy()
            recent_df['open_time'] = pd.to_datetime(recent_df['open_time'], unit='ms')
            
            # 시간 차이 계산
            time_diffs = recent_df['open_time'].diff().dropna()
            expected_diff = timedelta(minutes=15)
            tolerance = timedelta(seconds=30)  # 30초 허용오차
            
            invalid_intervals = 0
            for diff in time_diffs:
                if abs(diff - expected_diff) > tolerance:
                    invalid_intervals += 1
                    self.logger.debug(f"Invalid interval: {diff} (expected {expected_diff})")
            
            # 10% 이상 잘못된 간격이 있으면 실패
            if invalid_intervals > len(time_diffs) * 0.1:
                self.logger.warning(
                    f"Too many invalid intervals: {invalid_intervals}/{len(time_diffs)}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating intervals: {e}")
            return False
    
    def clean_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        잘못된 데이터 정리
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            if df.empty:
                return df
            
            original_length = len(df)
            df_clean = df.copy()
            
            # 1. NaN 값 제거
            df_clean = df_clean.dropna()
            
            # 2. 0 또는 음수 가격 제거
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df_clean = df_clean[df_clean[col] > 0]
            
            # 3. High < Low 같은 논리적 오류 제거
            df_clean = df_clean[df_clean['high'] >= df_clean['low']]
            df_clean = df_clean[df_clean['high'] >= df_clean['open']]
            df_clean = df_clean[df_clean['high'] >= df_clean['close']]
            df_clean = df_clean[df_clean['low'] <= df_clean['open']]
            df_clean = df_clean[df_clean['low'] <= df_clean['close']]
            
            # 4. 시간 순서 정렬
            df_clean = df_clean.sort_values('open_time')
            
            # 5. 중복 시간 제거 (최신 것만 유지)
            df_clean = df_clean.drop_duplicates('open_time', keep='last')
            
            cleaned_count = original_length - len(df_clean)
            if cleaned_count > 0:
                self.logger.info(f"Cleaned {cleaned_count} invalid candles")
            
            return df_clean.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return df
    
    def merge_candle(self, historical_df: pd.DataFrame, new_candle: Dict) -> pd.DataFrame:
        """
        Merge a new completed candle into historical data - 강화된 검증
        
        Args:
            historical_df: Historical candles DataFrame
            new_candle: New candle to merge
            
        Returns:
            pd.DataFrame: Updated DataFrame
        """
        try:
            # 새 캔들 유효성 검사
            if not self._validate_new_candle(new_candle):
                self.logger.warning("Invalid new candle data, skipping merge")
                return historical_df
            
            # 새 캔들을 DataFrame으로 변환
            new_row = pd.DataFrame([{
                'open_time': new_candle['t'],
                'open': float(new_candle['o']),
                'high': float(new_candle['h']),
                'low': float(new_candle['l']),
                'close': float(new_candle['c']),
                'volume': float(new_candle['v']),
                'close_time': new_candle['T'],
                'quote_volume': float(new_candle['q']),
                'trades': int(new_candle['n'])
            }])
            
            # 중복 확인
            if len(historical_df) > 0:
                last_time = historical_df.iloc[-1]['open_time']
                if new_candle['t'] <= last_time:
                    self.logger.debug(f"Candle already exists or is older: {new_candle['t']} <= {last_time}")
                    return historical_df
            
            # 병합
            updated_df = pd.concat([historical_df, new_row], ignore_index=True)
            
            # 데이터 정리
            updated_df = self.clean_invalid_data(updated_df)
            
            # 시간 순 정렬 및 중복 제거
            updated_df = updated_df.sort_values('open_time').drop_duplicates('open_time', keep='last')
            
            # 최대 캔들 수 제한
            if len(updated_df) > self.max_candles:
                updated_df = updated_df.tail(self.max_candles)
            
            # 데이터 품질 검증
            if not self.validate_candle_data_quality(updated_df):
                self.logger.warning("Merged data failed quality check")
            
            return updated_df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error merging candle: {e}")
            return historical_df
    
    def _validate_new_candle(self, candle: Dict) -> bool:
        """새 캔들 데이터 유효성 검사"""
        try:
            # 필수 필드 확인
            required_fields = ['t', 'T', 'o', 'h', 'l', 'c', 'v', 'q', 'n']
            for field in required_fields:
                if field not in candle:
                    self.logger.warning(f"Missing field in candle: {field}")
                    return False
            
            # 데이터 타입 및 값 검증
            o, h, l, c = float(candle['o']), float(candle['h']), float(candle['l']), float(candle['c'])
            v = float(candle['v'])
            
            # 가격 유효성
            if any(price <= 0 for price in [o, h, l, c]):
                self.logger.warning(f"Invalid prices in candle: O:{o}, H:{h}, L:{l}, C:{c}")
                return False
            
            # OHLC 논리적 유효성
            if h < max(o, l, c) or l > min(o, h, c):
                self.logger.warning(f"Invalid OHLC logic: O:{o}, H:{h}, L:{l}, C:{c}")
                return False
            
            # 거래량 유효성 (음수는 안됨, 0은 허용)
            if v < 0:
                self.logger.warning(f"Negative volume: {v}")
                return False
            
            # 시간 유효성
            if candle['t'] >= candle['T']:
                self.logger.warning(f"Invalid time range: {candle['t']} >= {candle['T']}")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Data type error in candle validation: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating new candle: {e}")
            return False
    
    def format_rest_candles(self, klines: List) -> pd.DataFrame:
        """
        Format REST API kline data to DataFrame - 강화된 검증
        
        Args:
            klines: Raw kline data from Binance
            
        Returns:
            pd.DataFrame: Formatted candle data
        """
        try:
            if not klines:
                self.logger.warning("No klines data provided")
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce', downcast='integer')
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce', downcast='integer')
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce', downcast='integer')
            
            # 필요한 컬럼만 선택
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'quote_volume', 'trades']]
            
            # 데이터 정리
            df = self.clean_invalid_data(df)
            
            # 데이터 품질 검증
            if not self.validate_candle_data_quality(df):
                self.logger.warning("REST candle data failed quality check")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error formatting REST candles: {e}")
            return pd.DataFrame()
    
    def format_ws_candle(self, kline_data: Dict) -> Dict:
        """
        Format WebSocket kline data - 강화된 검증
        
        Args:
            kline_data: Raw kline data from WebSocket
            
        Returns:
            Dict: Formatted current candle
        """
        try:
            # 데이터 구조 확인
            if 'k' not in kline_data:
                self.logger.error("Invalid WebSocket kline data structure")
                return {}
            
            k = kline_data['k']
            
            # 필수 필드 확인
            required_fields = ['t', 'T', 'o', 'h', 'l', 'c', 'v', 'q', 'n', 'x']
            for field in required_fields:
                if field not in k:
                    self.logger.warning(f"Missing field in WebSocket candle: {field}")
                    return {}
            
            candle = {
                't': k['t'],  # Open time
                'T': k['T'],  # Close time
                'o': float(k['o']),  # Open
                'h': float(k['h']),  # High
                'l': float(k['l']),  # Low
                'c': float(k['c']),  # Close
                'v': float(k['v']),  # Volume
                'q': float(k['q']),  # Quote volume
                'n': int(k['n']),    # Number of trades
                'x': k['x']  # Is closed
            }
            
            # 유효성 검사
            if not self._validate_new_candle(candle):
                self.logger.warning("WebSocket candle failed validation")
                return {}
            
            return candle
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Data type error in WebSocket candle formatting: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error formatting WebSocket candle: {e}")
            return {}
    
    def should_refresh_all(self, df: pd.DataFrame) -> bool:
        """
        Check if all candles should be refreshed - 강화된 판단 로직
        
        Args:
            df: Current candle DataFrame
            
        Returns:
            bool: True if refresh needed
        """
        if df is None or len(df) < self.validation_candles:
            self.logger.info("Refresh needed: insufficient data")
            return True
        
        # 1. 간격 유효성 검사
        if not self.validate_candle_intervals(df):
            self.logger.warning("Refresh needed: invalid intervals")
            return True
        
        # 2. 데이터 품질 검사
        if not self.validate_candle_data_quality(df):
            self.logger.warning("Refresh needed: poor data quality")
            return True
        
        # 3. 오래된 데이터 검사
        if self.detect_stale_data(df):
            self.logger.warning("Refresh needed: stale data")
            return True
        
        # 4. 과도한 0 볼륨 캔들 검사
        recent_candles = df.tail(20)
        zero_volume_ratio = (recent_candles['volume'] == 0).sum() / len(recent_candles)
        if zero_volume_ratio > 0.3:  # 30% 이상이 0 볼륨
            self.logger.warning(f"Refresh needed: too many zero volume candles ({zero_volume_ratio:.1%})")
            return True
        
        return False
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        데이터 품질 보고서 생성
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            Dict: Quality report
        """
        try:
            if df.empty:
                return {'status': 'empty', 'issues': ['No data']}
            
            report = {
                'status': 'good',
                'total_candles': len(df),
                'issues': [],
                'warnings': []
            }
            
            # 0 볼륨 캔들 확인
            zero_volume_count = (df['volume'] == 0).sum()
            if zero_volume_count > 0:
                report['warnings'].append(f"Zero volume candles: {zero_volume_count}")
            
            # 동일한 캔들 확인
            recent_df = df.tail(20)
            duplicated_prices = 0
            for i in range(1, len(recent_df)):
                prev = recent_df.iloc[i-1]
                curr = recent_df.iloc[i]
                if (prev['open'] == curr['open'] and prev['high'] == curr['high'] and
                    prev['low'] == curr['low'] and prev['close'] == curr['close']):
                    duplicated_prices += 1
            
            if duplicated_prices > 3:
                report['issues'].append(f"Too many identical price candles: {duplicated_prices}")
                report['status'] = 'poor'
            
            # 시간 간격 확인
            if not self.validate_candle_intervals(df):
                report['issues'].append("Invalid time intervals")
                report['status'] = 'poor'
            
            # 데이터 신선도 확인
            if self.detect_stale_data(df):
                report['issues'].append("Stale data detected")
                report['status'] = 'poor'
            
            # 전체 거래량 확인
            total_volume = df['volume'].sum()
            if total_volume < self.min_volume_threshold * len(df):
                report['warnings'].append(f"Low total volume: {total_volume}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {'status': 'error', 'issues': [str(e)]}
    
    def cleanup_old_data(self, df: pd.DataFrame, max_age_hours: int = 48) -> pd.DataFrame:
        """
        오래된 데이터 정리
        
        Args:
            df: DataFrame with candle data
            max_age_hours: Maximum age in hours
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            if df.empty:
                return df
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            # 오래된 데이터 제거
            df_cleaned = df[df['open_time'] >= cutoff_timestamp]
            
            removed_count = len(df) - len(df_cleaned)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} old candles (older than {max_age_hours}h)")
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning old data: {e}")
            return df