import numpy as np
import pandas as pd
import math
from scipy import stats
from src.utils.logger import get_logger
from typing import List  # 파일 상단에 추가

class Indicators:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def vwma(df, length, symbol=None):
        """
        Volume Weighted Moving Average 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (close, volume 컬럼 필요)
            length (int): 이동평균 기간
            symbol (str, optional): 로깅용 심볼명
        
        Returns:
            np.array: VWMA 값 배열
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        if 'close' not in df.columns or 'volume' not in df.columns:
            logger.error(f"{symbol_prefix}VWMA 계산을 위해 'close'와 'volume' 컬럼이 필요합니다.")
            return np.array([])
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}VWMA 계산 실패: 빈 데이터프레임")
                return np.array([])
            
            # NaN 값 확인 및 처리
            if df['close'].isna().any() or df['volume'].isna().any():
                logger.warning(f"{symbol_prefix}VWMA 계산 중 NaN 값 발견, 보간 처리")
                df = df.copy()
                df['close'] = df['close'].interpolate(method='linear')
                df['volume'] = df['volume'].interpolate(method='linear')
            
            # 가격과 거래량의 곱 계산
            df['close_volume'] = df['close'] * df['volume']
            
            # 이동 합계 계산 - 정확한 rolling window 적용
            volume_sum = df['volume'].rolling(window=length, min_periods=1).sum()
            close_volume_sum = df['close_volume'].rolling(window=length, min_periods=1).sum()
            
            # 0으로 나누기 방지 (거래량이 0인 구간 처리)
            volume_sum = volume_sum.replace(0, np.nan)
            
            # VWMA 계산
            vwma = close_volume_sum / volume_sum
            
            # NaN 값을 이전 값으로 채우기 (앞에서부터) - ffill() 사용
            vwma = vwma.ffill()
            
            # 첫 번째 값의 NaN을 현재 가격으로 채우기
            vwma = vwma.fillna(df['close'])
            
            return vwma.values
        
        except Exception as e:
            logger.error(f"{symbol_prefix}VWMA 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 오류 발생 시 원본 가격 반환
            return df['close'].values

    @staticmethod
    def cci(df, length=20, smoothing=20, symbol=None):
        """
        Commodity Channel Index (CCI) 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            length (int): CCI 계산 기간
            smoothing (int): 이동평균 스무딩 기간
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            np.array: CCI 값 배열
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}CCI 계산 실패: 빈 데이터프레임")
                return np.array([])
            
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}CCI 계산을 위해 'high', 'low', 'close' 컬럼이 필요합니다.")
                return np.array([])
            
            # 전형적 가격 (Typical Price) = (H + L + C) / 3
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            
            # 전형적 가격의 이동평균
            df['tp_ma'] = df['tp'].rolling(window=length).mean()
            
            # 전형적 가격의 이동평균과의 편차
            df['dev'] = df['tp'] - df['tp_ma']
            
            # 평균 편차 (Mean Deviation)
            df['mean_dev'] = df['tp'].rolling(window=length).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
            
            # CCI 계산: (TP - TP_MA) / (0.015 * Mean Deviation)
            # 0으로 나누기 방지
            df['mean_dev'] = df['mean_dev'].replace(0, np.nan)
            df['cci'] = df['dev'] / (0.015 * df['mean_dev'])
            
            # 스무딩 적용 (선택적)
            if smoothing > 0 and smoothing != length:
                df['cci'] = df['cci'].rolling(window=smoothing).mean()
            
            # NaN 값 채우기
            df['cci'] = df['cci'].fillna(0)
            
            return df['cci'].values
            
        except Exception as e:
            logger.error(f"{symbol_prefix}CCI 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 오류 발생 시 기본값 반환
            return np.zeros(len(df))

    @staticmethod
    def rsi(df, length=14, symbol=None):
        """
        Relative Strength Index (RSI) calculation
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'close' column
            length (int): Period length for RSI calculation (default: 14)
            symbol (str, optional): Symbol name for logging
            
        Returns:
            np.array: RSI values
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # Input validation
            if df.empty or len(df) < length + 1:
                logger.warning(f"{symbol_prefix}Not enough data for RSI calculation. Need: {length+1}, Current: {len(df)}")
                return np.zeros(len(df))
            
            # Get price data
            n = len(df)
            close = df['close'].values
            
            # Initialize arrays
            gains = np.zeros(n)
            losses = np.zeros(n)
            avg_gains = np.zeros(n)
            avg_losses = np.zeros(n)
            rs = np.zeros(n)
            rsi = np.zeros(n)
            
            # Calculate price changes and separate gains/losses
            for i in range(1, n):
                change = close[i] - close[i-1]
                if change > 0:
                    gains[i] = change
                    losses[i] = 0
                else:
                    gains[i] = 0
                    losses[i] = abs(change)
            
            # Apply Wilder's RSI calculation
            if n > length:
                # First averages: Simple average of first 'length' periods
                first_avg_gain = np.sum(gains[1:length+1]) / length
                first_avg_loss = np.sum(losses[1:length+1]) / length
                
                avg_gains[length] = first_avg_gain
                avg_losses[length] = first_avg_loss
                
                # Calculate first RS and RSI value
                if first_avg_loss == 0:
                    rsi[length] = 100.0
                else:
                    rs[length] = first_avg_gain / first_avg_loss
                    rsi[length] = 100.0 - (100.0 / (1.0 + rs[length]))
                
                # Calculate smoothed averages using Wilder's method for subsequent periods
                for i in range(length+1, n):
                    # Wilder smoothing formula:
                    # New average = ((Previous average × (length-1)) + Current value) / length
                    avg_gains[i] = ((avg_gains[i-1] * (length - 1)) + gains[i]) / length
                    avg_losses[i] = ((avg_losses[i-1] * (length - 1)) + losses[i]) / length
                    
                    # Calculate RS and RSI
                    if avg_losses[i] == 0:
                        rsi[i] = 100.0
                    else:
                        rs[i] = avg_gains[i] / avg_losses[i]
                        rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
            
            # Fill initial values with neutral RSI (50)
            for i in range(length):
                rsi[i] = 50.0
            
            return rsi
            
        except Exception as e:
            logger.error(f"{symbol_prefix}Error calculating RSI: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros on error
            return np.zeros(len(df))
    
    @staticmethod
    def supertrend(df, atr_length=10, multiplier=3.0, symbol=None):
        """
        Supertrend 지표 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            atr_length (int): ATR 계산 기간
            multiplier (float): ATR 승수
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            tuple: (Supertrend 값 배열, 추세 방향 배열(1=상승, -1=하락), 상단 밴드, 하단 밴드)
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}Supertrend 계산 실패: 빈 데이터프레임")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}Supertrend 계산을 위해 'high', 'low', 'close' 컬럼이 필요합니다.")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # ATR 계산
            n = len(df)
            
            # True Range 계산
            tr = np.zeros(n)
            tr[0] = df['high'].iloc[0] - df['low'].iloc[0]  # 첫 번째 값 설정
            
            for i in range(1, n):
                tr[i] = max(
                    df['high'].iloc[i] - df['low'].iloc[i],
                    abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                    abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                )
            
            # ATR 계산 (Wilder의 이동평균 사용)
            atr = np.zeros(n)
            atr[atr_length-1] = np.mean(tr[:atr_length])
            
            for i in range(atr_length, n):
                atr[i] = ((atr_length - 1) * atr[i-1] + tr[i]) / atr_length
            
            # 기본 밴드 계산
            hl2 = (df['high'] + df['low']) / 2
            basic_upper_band = hl2 + (multiplier * atr)
            basic_lower_band = hl2 - (multiplier * atr)
            
            # 최종 밴드 및 추세 방향 계산
            final_upper_band = np.zeros(n)
            final_lower_band = np.zeros(n)
            supertrend = np.zeros(n)
            trend_direction = np.zeros(n)  # 1: 상승 추세, -1: 하락 추세
            
            # 첫 번째 값 설정 (ATR이 계산되는 시점부터)
            if n <= atr_length:
                logger.warning(f"{symbol_prefix}Supertrend 계산을 위한 충분한 데이터가 없습니다.")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # 초기값 설정 (첫 번째 ATR 계산 시점에서 시작)
            final_upper_band[atr_length-1] = basic_upper_band.iloc[atr_length-1]
            final_lower_band[atr_length-1] = basic_lower_band.iloc[atr_length-1]
            trend_direction[atr_length-1] = -1  # 기본 방향은 하락
            supertrend[atr_length-1] = final_upper_band[atr_length-1]
            
            # 나머지 계산
            for i in range(atr_length, n):
                # 상단 밴드 계산
                if (basic_upper_band.iloc[i] < final_upper_band[i-1]) or (df['close'].iloc[i-1] > final_upper_band[i-1]):
                    final_upper_band[i] = basic_upper_band.iloc[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]
                
                # 하단 밴드 계산
                if (basic_lower_band.iloc[i] > final_lower_band[i-1]) or (df['close'].iloc[i-1] < final_lower_band[i-1]):
                    final_lower_band[i] = basic_lower_band.iloc[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]
                
                # 추세 방향 결정
                if supertrend[i-1] == final_upper_band[i-1]:
                    if df['close'].iloc[i] > final_upper_band[i]:
                        trend_direction[i] = 1  # 상승 추세
                        supertrend[i] = final_lower_band[i]
                    else:
                        trend_direction[i] = -1  # 하락 추세
                        supertrend[i] = final_upper_band[i]
                elif supertrend[i-1] == final_lower_band[i-1]:
                    if df['close'].iloc[i] < final_lower_band[i]:
                        trend_direction[i] = -1  # 하락 추세
                        supertrend[i] = final_upper_band[i]
                    else:
                        trend_direction[i] = 1  # 상승 추세
                        supertrend[i] = final_lower_band[i]
            
            return supertrend, trend_direction, final_upper_band, final_lower_band
            
        except Exception as e:
            logger.error(f"{symbol_prefix}Supertrend 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), np.array([]), np.array([])

    @staticmethod
    def atr(df, length=24, symbol=None):
        """
        Average True Range (ATR) 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            length (int): ATR 계산 기간
            symbol (str, optional): 로깅용 심볼명
                
        Returns:
            np.array: ATR 값 배열
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}ATR 계산 실패: 빈 데이터프레임")
                return np.array([])
            
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}ATR 계산을 위해 'high', 'low', 'close' 컬럼이 필요합니다.")
                return np.array([])
            
            n = len(df)
            
            # True Range 계산
            tr = np.zeros(n)
            tr[0] = df['high'].iloc[0] - df['low'].iloc[0]  # 첫 번째 값 설정
            
            for i in range(1, n):
                tr[i] = max(
                    df['high'].iloc[i] - df['low'].iloc[i],
                    abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                    abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                )
            
            # ATR 계산 (Wilder의 이동평균 사용)
            atr = np.zeros(n)
            atr[length-1] = np.mean(tr[:length])
            
            for i in range(length, n):
                atr[i] = ((length - 1) * atr[i-1] + tr[i]) / length
            
            # 첫 번째 값들 (계산되지 않은 부분) 채우기
            for i in range(length-1):
                if i == 0:
                    atr[i] = tr[i]
                else:
                    atr[i] = np.mean(tr[:i+1])
            
            return atr
            
        except Exception as e:
            logger.error(f"{symbol_prefix}ATR 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(df))    
        
    @staticmethod
    def psar(df, start=0.02, increment=0.02, maximum=0.2, symbol=None):
        """
        Parabolic SAR (Stop and Reverse) 지표 계산
        J. Welles Wilder의 원래 알고리즘에 기반함
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            start (float): 시작 AF (Acceleration Factor) 값, 기본값 0.02
            increment (float): AF 증가값, 기본값 0.02
            maximum (float): 최대 AF 값, 기본값 0.2
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            tuple: (PSAR 값 배열, 추세 방향 배열(1=상승, -1=하락))
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}PSAR 계산 실패: 빈 데이터프레임")
                return np.array([]), np.array([])
            
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}PSAR 계산을 위해 'high', 'low', 'close' 컬럼이 필요합니다.")
                return np.array([]), np.array([])
            
            n = len(df)
            if n < 2:
                logger.warning(f"{symbol_prefix}PSAR 계산을 위해 최소 2개의 캔들이 필요합니다.")
                return np.zeros(n), np.zeros(n)
            
            # 결과 배열 초기화
            psar = np.zeros(n)
            trend = np.zeros(n)  # 1: 상승 추세, -1: 하락 추세
            ep = np.zeros(n)  # Extreme Point
            af = np.zeros(n)  # Acceleration Factor
            
            # 두 번째 캔들에서 트렌드 결정 (첫 번째 캔들과 두 번째 캔들 종가 비교)
            trend[1] = 1 if df['close'].iloc[1] > df['close'].iloc[0] else -1
            
            # 초기 PSAR 설정
            if trend[1] == 1:
                # 상승 추세 시작
                psar[1] = min(df['low'].iloc[0], df['low'].iloc[1])  # 이전 최저가
                ep[1] = df['high'].iloc[1]  # 현재 최고가
            else:
                # 하락 추세 시작
                psar[1] = max(df['high'].iloc[0], df['high'].iloc[1])  # 이전 최고가
                ep[1] = df['low'].iloc[1]  # 현재 최저가
            
            af[1] = start  # 초기 AF 설정
            
            # 나머지 캔들 계산
            for i in range(2, n):
                # 이전 추세 유지
                trend[i] = trend[i-1]
                
                # PSAR 계산
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # PSAR 제한: 상승 추세에서는 이전 2개 캔들의 최저가보다 높게 설정 불가
                # 하락 추세에서는 이전 2개 캔들의 최고가보다 낮게 설정 불가
                if trend[i] == 1:
                    # 상승 추세
                    psar[i] = min(psar[i], min(df['low'].iloc[i-1], df['low'].iloc[i-2]))
                    
                    # 신규 최고가 갱신 시 EP와 AF 업데이트
                    if df['high'].iloc[i] > ep[i-1]:
                        ep[i] = df['high'].iloc[i]
                        af[i] = min(af[i-1] + increment, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # 추세 반전 체크
                    if df['low'].iloc[i] < psar[i]:
                        # 하락 추세로 전환
                        trend[i] = -1
                        psar[i] = ep[i-1]  # 이전 최고가로 PSAR 재설정
                        ep[i] = df['low'].iloc[i]  # 새 EP는 현재 최저가
                        af[i] = start  # AF 초기화
                else:
                    # 하락 추세
                    psar[i] = max(psar[i], max(df['high'].iloc[i-1], df['high'].iloc[i-2]))
                    
                    # 신규 최저가 갱신 시 EP와 AF 업데이트
                    if df['low'].iloc[i] < ep[i-1]:
                        ep[i] = df['low'].iloc[i]
                        af[i] = min(af[i-1] + increment, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # 추세 반전 체크
                    if df['high'].iloc[i] > psar[i]:
                        # 상승 추세로 전환
                        trend[i] = 1
                        psar[i] = ep[i-1]  # 이전 최저가로 PSAR 재설정
                        ep[i] = df['high'].iloc[i]  # 새 EP는 현재 최고가
                        af[i] = start  # AF 초기화
            
            # 첫 번째 값은 계산할 수 없으므로 두 번째 값으로 채움
            psar[0] = psar[1]
            trend[0] = trend[1]
            
            return psar, trend
            
        except Exception as e:
            logger.error(f"{symbol_prefix}PSAR 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(df)), np.zeros(len(df))
    
    @staticmethod
    def vortex(df, length=14, symbol=None):
        """
        Vortex Indicator (VI) 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            length (int): VI 계산 기간
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            tuple: (VI+, VI-) 값 배열의 튜플
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}VI 계산 실패: 빈 데이터프레임")
                return np.array([]), np.array([])
            
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}VI 계산을 위해 'high', 'low', 'close' 컬럼이 필요합니다.")
                return np.array([]), np.array([])
            
            n = len(df)
            if n < length + 1:
                logger.warning(f"{symbol_prefix}VI 계산을 위해 최소 {length+1}개의 캔들이 필요합니다.")
                return np.zeros(n), np.zeros(n)
            
            # True Range (TR) 계산
            tr = np.zeros(n)
            tr[0] = df['high'].iloc[0] - df['low'].iloc[0]  # 첫 번째 캔들은 간단하게 계산
            
            # VM+ 및 VM- 계산 (Uptrend/Downtrend Movement)
            vm_plus = np.zeros(n)
            vm_minus = np.zeros(n)
            
            for i in range(1, n):
                # True Range 계산
                tr[i] = max(
                    df['high'].iloc[i] - df['low'].iloc[i],
                    abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                    abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                )
                
                # VM+ = |현재 고가 - 이전 저가|
                vm_plus[i] = abs(df['high'].iloc[i] - df['low'].iloc[i-1])
                
                # VM- = |현재 저가 - 이전 고가|
                vm_minus[i] = abs(df['low'].iloc[i] - df['high'].iloc[i-1])
   
            # 결과 배열 초기화
            vi_plus = np.zeros(n)
            vi_minus = np.zeros(n)
            
            # 롤링 합계 계산 및 VI+/VI- 계산
            for i in range(length, n):
                # length 기간 동안의 TR, VM+, VM- 합계
                sum_tr = np.sum(tr[i-length+1:i+1])
                sum_vm_plus = np.sum(vm_plus[i-length+1:i+1])
                sum_vm_minus = np.sum(vm_minus[i-length+1:i+1])
                
                # VI+ = SUM(VM+) / SUM(TR)
                vi_plus[i] = sum_vm_plus / sum_tr if sum_tr > 0 else 0
                
                # VI- = SUM(VM-) / SUM(TR)
                vi_minus[i] = sum_vm_minus / sum_tr if sum_tr > 0 else 0
            
            return vi_plus, vi_minus
            
        except Exception as e:
            logger.error(f"{symbol_prefix}VI 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(df)), np.zeros(len(df))

    @staticmethod
    def mfi(df, length=14, symbol=None):
        """
        Money Flow Index (MFI) 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 (high, low, close, volume 컬럼 필요)
            length (int): MFI 계산 기간 (기본값: 14)
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            np.array: MFI 값 배열
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 입력 데이터 검증
            if df.empty:
                logger.error(f"{symbol_prefix}MFI 계산 실패: 빈 데이터프레임")
                return np.array([])
            
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"{symbol_prefix}MFI 계산을 위해 'high', 'low', 'close', 'volume' 컬럼이 필요합니다.")
                return np.array([])
            
            n = len(df)
            if n < length + 1:
                logger.warning(f"{symbol_prefix}MFI 계산을 위해 최소 {length+1}개의 캔들이 필요합니다.")
                return np.zeros(n)
            
            # 전형적 가격 (Typical Price) = (H + L + C) / 3
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            
            # 자금 흐름 (Money Flow) = TP * Volume
            df['money_flow'] = df['tp'] * df['volume']
            
            # 자금 흐름 방향 계산 (Positive/Negative Money Flow)
            df['price_change'] = df['tp'].diff()
            df['positive_flow'] = np.where(df['price_change'] > 0, df['money_flow'], 0)
            df['negative_flow'] = np.where(df['price_change'] < 0, df['money_flow'], 0)
            
            # 결과 배열 초기화
            mfi = np.zeros(n)
            
            # 롤링 합계 계산 및 MFI 계산
            for i in range(length, n):
                positive_sum = np.sum(df['positive_flow'].iloc[i-length+1:i+1])
                negative_sum = np.sum(df['negative_flow'].iloc[i-length+1:i+1])
                
                if negative_sum == 0:
                    # 음수 자금 흐름이 없으면 MFI = 100
                    mfi[i] = 100
                else:
                    money_ratio = positive_sum / negative_sum
                    mfi[i] = 100 - (100 / (1 + money_ratio))
            
            # 초기값 채우기 (50 - 중립)
            for i in range(length):
                mfi[i] = 50
            
            return mfi
            
        except Exception as e:
            logger.error(f"{symbol_prefix}MFI 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(df))

    @staticmethod
    def adx(df, length=14, smoothing=14):
        """
        Binance-compatible Average Directional Index (ADX) calculation
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'high', 'low', 'close' columns
            length (int): Period length for DI calculations
            smoothing (int): Smoothing period for ADX
            
        Returns:
            tuple: (ADX values, +DI values, -DI values)
        """
        logger = get_logger("indicators")
        
        try:
            # Input validation
            if df.empty or len(df) < length + smoothing:
                logger.warning(f"Not enough data for ADX calculation. Need: {length + smoothing}, Current: {len(df)}")
                return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
            
            # Working copy and array length
            n = len(df)
            
            # Initialize arrays
            tr = np.zeros(n)
            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            tr_smooth = np.zeros(n)
            plus_dm_smooth = np.zeros(n)
            minus_dm_smooth = np.zeros(n)
            plus_di = np.zeros(n)
            minus_di = np.zeros(n)
            dx = np.zeros(n)
            adx = np.zeros(n)
            
            # Get price data as numpy arrays for efficiency
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate TR, +DM, -DM for each bar
            for i in range(1, n):
                # True Range - Binance uses the original Wilder formula
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                
                # Directional Movement
                high_diff = high[i] - high[i-1]
                low_diff = low[i-1] - low[i]
                
                # +DM and -DM - Binance follows Wilder's exact method
                if high_diff > low_diff and high_diff > 0:
                    plus_dm[i] = high_diff
                else:
                    plus_dm[i] = 0
                    
                if low_diff > high_diff and low_diff > 0:
                    minus_dm[i] = low_diff
                else:
                    minus_dm[i] = 0
            
            # Initial smoothing - Binance's implementation with correct precision
            if n > length:
                # First value is the sum (not average) for the first 'length' periods
                tr_smooth[length] = np.sum(tr[1:length+1])
                plus_dm_smooth[length] = np.sum(plus_dm[1:length+1])
                minus_dm_smooth[length] = np.sum(minus_dm[1:length+1])
                
                # Calculate smoothed values for subsequent periods using Wilder's formula
                for i in range(length+1, n):
                    # Wilder's smoothing with correct rounding to match Binance
                    tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/length) + tr[i]
                    plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/length) + plus_dm[i]
                    minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/length) + minus_dm[i]
                
                # Calculate +DI and -DI from length onwards
                for i in range(length, n):
                    if tr_smooth[i] > 0:  # Avoid division by zero
                        plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
                        minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
                    else:
                        plus_di[i] = 0.0
                        minus_di[i] = 0.0
                    
                    # Calculate DX
                    di_sum = plus_di[i] + minus_di[i]
                    if di_sum > 0:  # Avoid division by zero
                        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
                    else:
                        dx[i] = 0.0
                
                # Calculate ADX - This part is where most implementations differ from Binance
                if n >= length + smoothing:
                    # First ADX value - Binance uses RMA (Wilder's moving average) for this
                    first_adx = 0.0
                    for i in range(length, length + smoothing):
                        first_adx += dx[i]
                    first_adx /= smoothing
                    adx[length+smoothing-1] = first_adx
                    
                    # Subsequent ADX values - Binance uses a modified Wilder's smoothing here
                    for i in range(length+smoothing, n):
                        # This exact formula matches Binance's ADX values at high smoothing periods
                        adx[i] = (adx[i-1] * (smoothing - 1) + dx[i]) / smoothing
            
            # Ensure we don't have NaN values (Binance doesn't return NaN)
            adx = np.nan_to_num(adx)
            plus_di = np.nan_to_num(plus_di)
            minus_di = np.nan_to_num(minus_di)
            
            return adx, plus_di, minus_di
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

    @staticmethod
    def hurst(df, long_window=64, rs_lag=16, smoothing_period=14, symbol=None):
        """
        실시간 거래를 위한 개선된 Hurst 지수 계산 (추세/횡보 구분용)
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임 ('close' 컬럼 필요)
            long_window (int): 분석 최대 구간 (기본값: 64)
            rs_lag (int): Modified R/S 보정용 최대 lag (기본값: 16)
            smoothing_period (int): EMA 스무딩 기간 (기본값: 14)
            symbol (str, optional): 로깅용 심볼명
            
        Returns:
            tuple: (raw Hurst 배열, EMA 스무딩 배열)
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""

        try:
            if df.empty or 'close' not in df.columns:
                logger.error(f"{symbol_prefix}Hurst 계산 실패: 빈 데이터프레임 또는 'close' 컬럼 없음")
                return np.array([]), np.array([])

            prices = df['close'].values
            n = len(prices)
            min_length = max(long_window * 3, 150)

            if n < min_length:
                logger.warning(f"{symbol_prefix}Hurst 계산을 위한 데이터 부족: 필요 {min_length}, 현재 {n}")
                return np.array([]), np.array([])

            hurst_raw = np.full(n, np.nan)

            for i in range(min_length, n):
                window_len = min(i, long_window * 4)
                segment = prices[i - window_len:i]
                returns = np.diff(np.log(segment))

                if len(returns) < min_length // 2:
                    continue

                returns = returns[np.isfinite(returns)]
                if len(returns) < 16:
                    continue

                min_seg = 8
                max_seg = min(len(returns) // 4, long_window)
                seg_count = 12

                if max_seg < min_seg * 2:
                    continue

                lengths = np.logspace(np.log10(min_seg), np.log10(max_seg), seg_count)
                segments = np.unique(np.floor(lengths).astype(int))
                if len(segments) < 3:
                    continue

                seg_lengths = []
                rs_vals = []

                mean_r = np.mean(returns)
                var_r = np.var(returns, ddof=0)

                adj_rs_lag = min(rs_lag, len(returns) // 4)
                autocov = []
                try:
                    autocov = [np.cov(returns[:-lag], returns[lag:])[0,1] if lag > 0 else var_r
                            for lag in range(0, adj_rs_lag + 1)]
                except:
                    continue

                valid_rs_count = 0
                for m in segments:
                    k = len(returns) // m
                    if k < 1:
                        continue
                    rs_seg = []
                    for j in range(k):
                        if j*m >= len(returns) or (j+1)*m > len(returns):
                            continue
                        seg = returns[j*m:(j+1)*m]
                        if len(seg) < 4:
                            continue
                        dev = seg - mean_r
                        cum = np.cumsum(dev)
                        R = cum.max() - cum.min()

                        try:
                            S2_q = var_r + 2*sum((1 - lag/(adj_rs_lag+1)) * autocov[lag]
                                                for lag in range(1, adj_rs_lag+1))
                            if S2_q <= 0:
                                continue
                            S_q = np.sqrt(S2_q)
                            rs_val = R / S_q
                            if 0 < rs_val < 100:
                                rs_seg.append(rs_val)
                                valid_rs_count += 1
                        except:
                            continue
                    if len(rs_seg) >= 2:
                        seg_lengths.append(m)
                        rs_vals.append(np.mean(rs_seg))

                if len(rs_vals) < 3:
                    continue

                try:
                    x = np.log10(seg_lengths)
                    y = np.log10(rs_vals)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    if r_value**2 < 0.5 or std_err > 0.25:
                        continue
                    H = slope
                    if 0 < H < 1:
                        hurst_raw[i] = H
                except:
                    continue

            # EMA smoothing
            hurst_smooth = np.full(n, np.nan)
            alpha = 2.0 / (smoothing_period + 1)
            valid_indices = np.where(~np.isnan(hurst_raw))[0]

            if valid_indices.size:
                first_valid = valid_indices[0]
                hurst_smooth[first_valid] = hurst_raw[first_valid]
                for i in valid_indices[1:]:
                    prev_indices = valid_indices[valid_indices < i]
                    if prev_indices.size:
                        prev_idx = prev_indices[-1]
                        if i - prev_idx > smoothing_period * 3:
                            hurst_smooth[i] = hurst_raw[i]
                        else:
                            hurst_smooth[i] = alpha * hurst_raw[i] + (1 - alpha) * hurst_smooth[prev_idx]
                    else:
                        hurst_smooth[i] = hurst_raw[i]

            if np.all(np.isnan(hurst_raw)):
                logger.error(f"{symbol_prefix}유효한 Hurst 지수를 계산하지 못했습니다.")
                return np.array([]), np.array([])

            valid_count = np.sum(~np.isnan(hurst_raw))
            logger.info(f"{symbol_prefix}Hurst 지수 계산 완료: 유효값 {valid_count}개 / 총 {n}개")

            return hurst_raw, hurst_smooth

        except Exception as e:
            logger.error(f"{symbol_prefix}Hurst 계산 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])

    @staticmethod
    def calculate_all(df,
                    cci_length=14, cci_smoothing=14, rsi_length=14,
                    supertrend_atr_length=10, supertrend_multiplier=3.0,
                    psar_start=0.02, psar_increment=0.02, psar_maximum=0.2,
                    vi_length=14, mfi_length=14,
                    hurst_window=24, hurst_rs_lag=24, hurst_smoothing=24,
                    vwma_short=6, vwma_mid=12, vwma_long=24,
                    atr_length=24,
                    adx_length=14, adx_smoothing=14,
                    symbol=None,
                    exclude_hurst=False):
        """
        모든 지표 계산 (KISS, DRY, YAGNI 원칙 적용)
        - 단일 Hurst window 사용
        - 단일 ADX 설정 사용 (short/long 분리 제거)
        """
        logger = get_logger("indicators")
        symbol_prefix = f"[{symbol}] " if symbol else ""
        
        try:
            # 기본 지표들 계산 (DRY 적용)
            result = {
                'cci': Indicators.cci(df, cci_length, cci_smoothing, symbol),
                'rsi': Indicators.rsi(df, rsi_length, symbol),
                'mfi': Indicators.mfi(df, mfi_length, symbol)
            }
            
            # Supertrend 계산
            supertrend, trend_direction, upper_band, lower_band = Indicators.supertrend(
                df, supertrend_atr_length, supertrend_multiplier, symbol
            )
            result.update({
                'supertrend': supertrend,
                'trend_direction': trend_direction,
                'supertrend_upper': upper_band,
                'supertrend_lower': lower_band
            })
            
            # PSAR 계산
            psar_values, psar_trend = Indicators.psar(df, psar_start, psar_increment, psar_maximum, symbol)
            result.update({
                'psar': psar_values,
                'psar_trend': psar_trend
            })
            
            # VI 계산
            vi_plus, vi_minus = Indicators.vortex(df, vi_length, symbol)
            result.update({
                'vi_plus': vi_plus,
                'vi_minus': vi_minus
            })
            
            # ADX 계산 (단일 설정)
            adx_values, plus_di, minus_di = Indicators.adx(df, adx_length, adx_smoothing)
            result.update({
                'adx': adx_values,
                'plus_di': plus_di,
                'minus_di': minus_di
            })
            
            # 기타 지표들
            result.update({
                'vwma_short': Indicators.vwma(df, vwma_short, symbol),
                'vwma_mid': Indicators.vwma(df, vwma_mid, symbol),
                'vwma_long': Indicators.vwma(df, vwma_long, symbol),
                'atr': Indicators.atr(df, atr_length, symbol)
            })
            
            # Hurst 지수 계산 (단일 window)
            if not exclude_hurst:
                logger.debug(f"{symbol_prefix}Hurst 지수 계산 중...")
                hurst_values, hurst_smoothed = Indicators.hurst(
                    df, hurst_window, hurst_rs_lag, hurst_smoothing, symbol
                )
                result.update({
                    'hurst': hurst_values,
                    'hurst_smoothed': hurst_smoothed
                })
            
            logger.debug(f"{symbol_prefix}모든 지표 계산 완료")
            return result
            
        except Exception as e:
            logger.error(f"{symbol_prefix}지표 계산 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 기본 빈 결과 반환 (KISS 적용)
            empty_array = np.array([])
            return {key: empty_array for key in [
                'cci', 'rsi', 'mfi', 'supertrend', 'trend_direction',
                'supertrend_upper', 'supertrend_lower', 'psar', 'psar_trend',
                'vi_plus', 'vi_minus', 'vwma_short', 'vwma_mid', 'vwma_long',
                'atr', 'adx', 'plus_di', 'minus_di'
            ]}