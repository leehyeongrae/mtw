"""
거래 신호 생성 모듈 (수정됨)
추세장/횡보장 판단 및 진입/청산 신호 생성
cs_rsi, cs_cci, cs_mfi 함수 구현 포함
"""
import numpy as np
from typing import Dict, Optional, Tuple
from src.utils.config import config
from src.utils.logger import get_logger

# 🔥 제거: detect_trending_market 함수 제거됨 (DRY 원칙)
# is_trending_market_by_slope 메소드로 통합됨

class SignalGenerator:
    """거래 신호 생성기"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = get_logger(f"signals_{symbol}")
        self.weight_rsi = config.weight_rsi
        self.weight_cci = config.weight_cci
        self.weight_mfi = config.weight_mfi
        
    def _normalize_indicator(self, value: float, min_val: float, max_val: float,
                           max_output: float, exponent: float = 1.0) -> float:
        """
        공통 지표 정규화 함수 (DRY 원칙 적용)
        
        Args:
            value: 정규화할 값
            min_val: 최소 기준값
            max_val: 최대 기준값
            max_output: 최대 출력값
            exponent: 비선형 변환 지수
            
        Returns:
            float: 정규화된 값
        """
        if value <= min_val:
            return 0.0
        
        # min_val에서 max_val까지 비선형 매핑
        normalized = min((value - min_val) / (max_val - min_val), 1.0)
        
        # 비선형 변환
        result = normalized ** exponent * max_output
        
        return min(result, max_output)

    def cs_rsi(self, rsi: float) -> float:
        """RSI 정규화: 50~90 → 0~2.0 (지수 1.5)"""
        return self._normalize_indicator(rsi, 50, 90, 2.0, 1.5)
    
    def cs_cci(self, cci: float) -> float:
        """CCI 정규화: 0~200 → 0~1.5 (지수 1.2)"""
        return self._normalize_indicator(cci, 0, 200, 1.5, 1.2)
    
    def cs_mfi(self, mfi: float) -> float:
        """MFI 정규화: 50~90 → 0~1.5 (지수 1.4)"""
        return self._normalize_indicator(mfi, 50, 90, 1.5, 1.4)
        
    def _apply_symmetric_normalization(self, value: float, threshold: float,
                                     normalize_func, mirror_transform=None) -> float:
        """
        대칭 정규화 공통 함수 (DRY 원칙 적용)
        
        Args:
            value: 정규화할 값
            threshold: 대칭 기준점
            normalize_func: 정규화 함수
            mirror_transform: 대칭 변환 함수 (기본: 2*threshold - value)
            
        Returns:
            float: 대칭 정규화된 값
        """
        if value < threshold:
            if mirror_transform:
                mirrored_value = mirror_transform(value)
            else:
                mirrored_value = 2 * threshold - value
            return -float(normalize_func(mirrored_value))
        return float(normalize_func(value))

    def normalize_rsi(self, rsi: float) -> float:
        """RSI 대칭 정규화: 50 기준 대칭"""
        return self._apply_symmetric_normalization(rsi, 50, self.cs_rsi)

    def normalize_cci(self, cci: float) -> float:
        """CCI 대칭 정규화: 0 기준 대칭"""
        return self._apply_symmetric_normalization(cci, 0, self.cs_cci, lambda x: -x)

    def normalize_mfi(self, mfi: float) -> float:
        """MFI 대칭 정규화: 50 기준 대칭"""
        return self._apply_symmetric_normalization(mfi, 50, self.cs_mfi)

    def calculate_score(self, rsi: float, cci: float, mfi: float, 
                       w_rsi: Optional[float] = None, w_cci: Optional[float] = None, 
                       w_mfi: Optional[float] = None) -> float:
        """
        RSI, CCI, MFI 값을 기반으로 과매수/과매도 점수 계산
        
        Args:
            rsi (float): RSI 값
            cci (float): CCI 값
            mfi (float): MFI 값
            w_rsi (float, optional): RSI 가중치 (기본값: config.weight_rsi)
            w_cci (float, optional): CCI 가중치 (기본값: config.weight_cci)
            w_mfi (float, optional): MFI 가중치 (기본값: config.weight_mfi)
                
        Returns:
            float: 통합 점수 (음수: 과매도, 양수: 과매수)
        """
        # 가중치 기본값 설정
        if w_rsi is None:
            w_rsi = self.weight_rsi
        if w_cci is None:
            w_cci = self.weight_cci
        if w_mfi is None:
            w_mfi = self.weight_mfi
            
        # 지표 정규화
        normalized_rsi = self.normalize_rsi(rsi)
        normalized_cci = self.normalize_cci(cci)
        normalized_mfi = self.normalize_mfi(mfi)
        
        # 가중 평균 계산
        score = (w_rsi * normalized_rsi) + (w_cci * normalized_cci) + (w_mfi * normalized_mfi)
        
        # 로깅
        self.logger.debug(
            f"점수 계산: RSI={rsi:.1f} (정규화: {normalized_rsi:.2f}), "
            f"CCI={cci:.1f} (정규화: {normalized_cci:.2f}), "
            f"MFI={mfi:.1f} (정규화: {normalized_mfi:.2f}), "
            f"최종 점수={score:.2f}"
        )
        
        return score
    
    def calculate_slope(self, values: np.ndarray, n_candles: int) -> float:
        """
        기울기 계산: (현재 값 - N캔들 전 값) / N - 안전 버전
        
        Args:
            values: 지표 값 배열
            n_candles: 기울기 계산에 사용할 캔들 수
            
        Returns:
            float: 기울기 값
        """
        try:
            if not isinstance(values, np.ndarray) or len(values) < n_candles + 1:
                return 0.0
            
            # NaN 값 확인
            if np.isnan(values[-1]) or np.isnan(values[-(n_candles + 1)]):
                return 0.0
            
            current_value = float(values[-1])
            past_value = float(values[-(n_candles + 1)])
            
            # 기울기 계산: (현재 값 - N캔들 전 값) / N
            slope = (current_value - past_value) / n_candles
            
            return slope
            
        except (IndexError, ValueError, TypeError) as e:
            self.logger.debug(f"기울기 계산 오류: {e}")
            return 0.0
    
    def _safe_calculate_slope(self, values: np.ndarray, n_candles: int) -> float:
        """
        안전한 기울기 계산 (NaN 값 처리 포함)
        
        Args:
            values: 지표 값 배열 (이미 NaN 필터링됨)
            n_candles: 기울기 계산에 사용할 캔들 수
            
        Returns:
            float or None: 기울기 값 또는 None (계산 실패 시)
        """
        try:
            if len(values) < n_candles + 1:
                return None
            
            current_value = float(values[-1])
            past_value = float(values[-(n_candles + 1)])
            
            # 기울기 계산
            slope = (current_value - past_value) / n_candles
            
            # 무한값 확인
            if not np.isfinite(slope):
                return None
            
            return slope
            
        except Exception as e:
            self.logger.debug(f"안전한 기울기 계산 오류: {e}")
            return None
    
    def is_trending_market_by_slope(self, adx_values: np.ndarray, hurst_values: np.ndarray) -> int:
        """
        기울기 기반 추세장 판별 - 개선된 안전 버전
        
        Args:
            adx_values: ADX 값 배열
            hurst_values: Hurst 지수 배열
            
        Returns:
            int: 1 (추세장), 0 (횡보장)
        """
        try:
            n_candles = getattr(config, 'trend_detection_candles', 5)
            
            # 입력 검증
            if not isinstance(adx_values, np.ndarray) or not isinstance(hurst_values, np.ndarray):
                self.logger.debug("추세장 판별: 입력값이 numpy 배열이 아님")
                return 0
            
            # 배열 길이 검증
            if len(adx_values) < n_candles + 1 or len(hurst_values) < n_candles + 1:
                self.logger.debug(f"추세장 판별: 데이터 부족 (ADX: {len(adx_values)}, Hurst: {len(hurst_values)}, 필요: {n_candles + 1})")
                return 0
            
            # NaN 값 필터링
            adx_clean = adx_values[~np.isnan(adx_values)]
            hurst_clean = hurst_values[~np.isnan(hurst_values)]
            
            if len(adx_clean) < n_candles + 1 or len(hurst_clean) < n_candles + 1:
                self.logger.debug(f"추세장 판별: NaN 제거 후 데이터 부족 (ADX: {len(adx_clean)}, Hurst: {len(hurst_clean)})")
                return 0
            
            # 기울기 계산 (안전한 버전)
            adx_slope = self._safe_calculate_slope(adx_clean, n_candles)
            hurst_slope = self._safe_calculate_slope(hurst_clean, n_candles)
            
            if adx_slope is None or hurst_slope is None:
                self.logger.debug("추세장 판별: 기울기 계산 실패")
                return 0
            
            # 두 기울기가 모두 양수이면 추세장
            is_trending = (adx_slope > 0 and hurst_slope > 0)
            
            self.logger.debug(
                f"추세장 판별 (기울기 기반): ADX 기울기={adx_slope:.6f}, "
                f"Hurst 기울기={hurst_slope:.6f}, "
                f"N캔들={n_candles}, 추세장={1 if is_trending else 0}"
            )
            
            return 1 if is_trending else 0
            
        except Exception as e:
            self.logger.debug(f"추세장 판별 중 오류: {e}")
            return 0
    
# 🔥 제거: is_trending_market 메소드 제거됨 (YAGNI 원칙)
# 사용되지 않는 레거시 메소드이므로 완전히 제거
    
    def get_trend_signal(self, psar_trend: int, supertrend_trend: int,
                        vi_plus: float, vi_minus: float, 
                        current_position: Optional[str] = None) -> Optional[str]:
        """
        추세장 진입/청산 신호 생성
        
        Args:
            psar_trend: PSAR 추세 (1: 상승, -1: 하락)
            supertrend_trend: Supertrend 추세 (1: 상승, -1: 하락)
            vi_plus: VI+
            vi_minus: VI-
            current_position: 현재 포지션 ('long', 'short', None)
            
        Returns:
            Optional[str]: 신호 ('long_entry', 'short_entry', 'long_exit', 'short_exit', None)
        """
        # 진입 신호
        if current_position is None:
            if psar_trend == 1 and supertrend_trend == 1:
                self.logger.info("추세장 롱 진입 신호: PSAR 상승 + Supertrend 상승")
                return 'long_entry'
            elif psar_trend == -1 and supertrend_trend == -1:
                self.logger.info("추세장 숏 진입 신호: PSAR 하락 + Supertrend 하락")
                return 'short_entry'
        
        # 청산 신호
        elif current_position == 'long':
            # 롱 청산 조건 1: PSAR와 Supertrend 추세가 반대이고 VI- > VI+
            if psar_trend != supertrend_trend and vi_minus > vi_plus:
                self.logger.info("추세장 롱 청산 신호: 추세 반대 + VI- > VI+")
                return 'long_exit'
            # 롱 청산 조건 2: 포지션 방향이 롱이고 PSAR, Supertrend 모두 하락
            elif psar_trend == -1 and supertrend_trend == -1:
                self.logger.info("추세장 롱 청산 신호: PSAR 하락 + Supertrend 하락")
                return 'long_exit'
                
        elif current_position == 'short':
            # 숏 청산 조건 1: PSAR와 Supertrend 추세가 반대이고 VI+ > VI-
            if psar_trend != supertrend_trend and vi_plus > vi_minus:
                self.logger.info("추세장 숏 청산 신호: 추세 반대 + VI+ > VI-")
                return 'short_exit'
            # 숏 청산 조건 2: 포지션 방향이 숏이고 PSAR, Supertrend 모두 상승
            elif psar_trend == 1 and supertrend_trend == 1:
                self.logger.info("추세장 숏 청산 신호: PSAR 상승 + Supertrend 상승")
                return 'short_exit'
        
        return None
    
    def get_ranging_signal(self, oscillator_score: float, 
                          current_position: Optional[str] = None) -> Optional[str]:
        """
        횡보장 진입/청산 신호 생성
        
        Args:
            oscillator_score: 오실레이터 점수
            current_position: 현재 포지션 ('long', 'short', None)
            
        Returns:
            Optional[str]: 신호 ('long_entry', 'short_entry', 'long_exit', 'short_exit', None)
        """
        # 진입 신호
        if current_position is None:
            if oscillator_score <= config.oscillator_long_threshold:
                self.logger.info(f"횡보장 롱 진입 신호: 점수={oscillator_score:.2f}")
                return 'long_entry'
            elif oscillator_score >= config.oscillator_short_threshold:
                self.logger.info(f"횡보장 숏 진입 신호: 점수={oscillator_score:.2f}")
                return 'short_entry'
        
        # 청산 신호
        elif current_position == 'long':
            if oscillator_score >= config.oscillator_short_threshold:
                self.logger.info(f"횡보장 롱 청산 신호: 점수={oscillator_score:.2f}")
                return 'long_exit'
                
        elif current_position == 'short':
            if oscillator_score <= config.oscillator_long_threshold:
                self.logger.info(f"횡보장 숏 청산 신호: 점수={oscillator_score:.2f}")
                return 'short_exit'
        
        return None
    
    def generate_signal(self, indicators: Dict, current_position: Optional[str] = None) -> Optional[Dict]:
        """
        종합 거래 신호 생성 - 개선된 안전 버전
        
        Args:
            indicators: 계산된 지표들 딕셔너리
            current_position: 현재 포지션
            
        Returns:
            Optional[Dict]: 신호 상세 정보 딕셔너리 또는 None
        """
        try:
            # 입력 검증
            if not indicators or not isinstance(indicators, dict):
                self.logger.debug("신호 생성: indicators가 비어있거나 딕셔너리가 아님")
                return None
            
            # 필수 지표 존재 확인
            required_indicators = ['adx', 'hurst_smoothed', 'psar_trend', 'trend_direction',
                                 'vi_plus', 'vi_minus', 'rsi', 'cci', 'mfi']
            
            for indicator in required_indicators:
                if indicator not in indicators or len(indicators[indicator]) == 0:
                    self.logger.debug(f"신호 생성: 필수 지표 '{indicator}' 누락 또는 비어있음")
                    return None
            
            # 안전하게 최신 지표값 추출
            def safe_get_last_value(arr, default_value, convert_type=float):
                try:
                    if len(arr) == 0:
                        return convert_type(default_value)
                    val = arr[-1]
                    if np.isnan(val) or not np.isfinite(val):
                        return convert_type(default_value)
                    return convert_type(val)
                except (IndexError, ValueError, TypeError):
                    return convert_type(default_value)
            
            # 지표값 안전 추출
            adx = safe_get_last_value(indicators['adx'], 0.0)
            hurst = safe_get_last_value(indicators['hurst_smoothed'], 0.5)
            
            psar_trend = safe_get_last_value(indicators['psar_trend'], 0, int)
            supertrend_trend = safe_get_last_value(indicators['trend_direction'], 0, int)
            
            vi_plus = safe_get_last_value(indicators['vi_plus'], 1.0)
            vi_minus = safe_get_last_value(indicators['vi_minus'], 1.0)
            
            rsi = safe_get_last_value(indicators['rsi'], 50.0)
            cci = safe_get_last_value(indicators['cci'], 0.0)
            mfi = safe_get_last_value(indicators['mfi'], 50.0)
            
            # 현재 지표값들을 딕셔너리로 저장
            current_indicators = {
                'adx': round(adx, 2),
                'hurst': round(hurst, 3),
                'psar_trend': psar_trend,
                'supertrend_trend': supertrend_trend,
                'vi_plus': round(vi_plus, 3),
                'vi_minus': round(vi_minus, 3),
                'rsi': round(rsi, 1),
                'cci': round(cci, 1),
                'mfi': round(mfi, 1)
            }
            
            # 추세장/횡보장 판단 (안전한 기울기 기반)
            trending_signal = self.is_trending_market_by_slope(
                indicators['adx'],
                indicators['hurst_smoothed']
            )
            is_trending = (trending_signal == 1)
            market_type = "trending" if is_trending else "ranging"
            
            # 현재 지표값에 추세장 판별 결과 추가
            current_indicators['trending_signal'] = trending_signal
            
            if is_trending:
                # 추세장 신호 생성
                signal_result = self.get_trend_signal(psar_trend, supertrend_trend,
                                                    vi_plus, vi_minus, current_position)
                
                if signal_result:
                    # 추세장 신호 이유 생성
                    reason = self._generate_trend_reason(signal_result, psar_trend, supertrend_trend,
                                                    vi_plus, vi_minus, current_position)
                    
                    return {
                        "action": signal_result,
                        "market_type": market_type,
                        "indicators": current_indicators,
                        "reason": reason
                    }
            else:
                # 횡보장 신호 생성
                try:
                    oscillator_score = self.calculate_score(rsi, cci, mfi)
                    current_indicators['oscillator_score'] = round(oscillator_score, 2)
                    
                    signal_result = self.get_ranging_signal(oscillator_score, current_position)
                    
                    if signal_result:
                        # 횡보장 신호 이유 생성
                        reason = self._generate_ranging_reason(signal_result, oscillator_score,
                                                            rsi, cci, mfi, current_position)
                        
                        return {
                            "action": signal_result,
                            "market_type": market_type,
                            "indicators": current_indicators,
                            "reason": reason
                        }
                except Exception as score_error:
                    self.logger.debug(f"오실레이터 점수 계산 오류: {score_error}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"신호 생성 오류: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _generate_trend_reason(self, signal: str, psar_trend: int, supertrend_trend: int,
                            vi_plus: float, vi_minus: float, current_position: Optional[str]) -> str:
        """추세장 신호 이유 생성"""
        
        if 'entry' in signal:
            if 'long' in signal:
                return f"추세장 롱 진입: PSAR {'상승' if psar_trend == 1 else '하락'} + Supertrend {'상승' if supertrend_trend == 1 else '하락'}"
            else:  # short entry
                return f"추세장 숏 진입: PSAR {'상승' if psar_trend == 1 else '하락'} + Supertrend {'상승' if supertrend_trend == 1 else '하락'}"
        
        elif 'exit' in signal:
            if current_position == 'long':
                if psar_trend != supertrend_trend:
                    return f"추세장 롱 청산: 추세 불일치 (PSAR: {psar_trend}, Supertrend: {supertrend_trend}) + VI-({vi_minus:.3f}) > VI+({vi_plus:.3f})"
                else:
                    return f"추세장 롱 청산: 하락 추세 전환 (PSAR: {psar_trend}, Supertrend: {supertrend_trend})"
            
            elif current_position == 'short':
                if psar_trend != supertrend_trend:
                    return f"추세장 숏 청산: 추세 불일치 (PSAR: {psar_trend}, Supertrend: {supertrend_trend}) + VI+({vi_plus:.3f}) > VI-({vi_minus:.3f})"
                else:
                    return f"추세장 숏 청산: 상승 추세 전환 (PSAR: {psar_trend}, Supertrend: {supertrend_trend})"
        
        return "추세장 신호"

    def _generate_ranging_reason(self, signal: str, oscillator_score: float, 
                            rsi: float, cci: float, mfi: float, 
                            current_position: Optional[str]) -> str:
        """횡보장 신호 이유 생성"""
        
        # 오실레이터 상태 분석
        rsi_state = "과매도" if rsi < 30 else "과매수" if rsi > 70 else "중립"
        cci_state = "과매도" if cci < -100 else "과매수" if cci > 100 else "중립"
        mfi_state = "과매도" if mfi < 20 else "과매수" if mfi > 80 else "중립"
        
        indicator_details = f"RSI: {rsi:.1f}({rsi_state}), CCI: {cci:.1f}({cci_state}), MFI: {mfi:.1f}({mfi_state})"
        
        if 'entry' in signal:
            if 'long' in signal:
                return f"횡보장 롱 진입: 과매도 신호 (점수: {oscillator_score:.2f}) - {indicator_details}"
            else:  # short entry
                return f"횡보장 숏 진입: 과매수 신호 (점수: {oscillator_score:.2f}) - {indicator_details}"
        
        elif 'exit' in signal:
            if current_position == 'long':
                return f"횡보장 롱 청산: 과매수 전환 (점수: {oscillator_score:.2f}) - {indicator_details}"
            elif current_position == 'short':
                return f"횡보장 숏 청산: 과매도 전환 (점수: {oscillator_score:.2f}) - {indicator_details}"
        
        return f"횡보장 신호 (점수: {oscillator_score:.2f})"        