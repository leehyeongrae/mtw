"""
Multiprocessing indicator calculation - 개선된 버전
"""
import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.indicators.indicators import Indicators
from src.indicators.signals import SignalGenerator
from typing import Dict, Optional, List 

class IndicatorProcessor:
    """Process indicators for a single symbol - 개선된 버전"""
    
    def __init__(self, symbol: str, data_manager: DataManager):
        self.symbol = symbol
        self.data_manager = data_manager
        self.logger = get_logger(f"indicator_{symbol}")
        self.indicators = Indicators()
        self.signal_generator = SignalGenerator(symbol)
        
        # Hurst 캐싱 개선
        self.last_closed_candle_time = None
        self.cached_hurst = None
        self.hurst_calculation_interval = 5  # 5개 캔들마다 강제 재계산
        self.candles_since_hurst = 0
        
        # 최소 데이터 요구량 완화
        self.min_data_for_hurst = 100  # 150에서 100으로 완화
        self.min_data_for_indicators = 50
        
    def _should_calculate_hurst(self, current_candle: Optional[Dict], df_length: int) -> bool:
        """Hurst 계산 여부 결정 - 개선된 로직"""
        try:
            # 최소 데이터 확인
            if df_length < self.min_data_for_hurst:
                self.logger.debug(f"Insufficient data for Hurst: {df_length}/{self.min_data_for_hurst}")
                return False
            
            # 현재 캔들이 완료된 경우에만 계산
            if current_candle and current_candle.get('x', False):
                candle_time = current_candle['T']
                
                # 새로운 완료된 캔들인지 확인
                if self.last_closed_candle_time != candle_time:
                    self.last_closed_candle_time = candle_time
                    self.candles_since_hurst += 1
                    
                    # 주기적 강제 재계산 또는 첫 계산
                    if (self.cached_hurst is None or 
                        self.candles_since_hurst >= self.hurst_calculation_interval):
                        self.candles_since_hurst = 0
                        return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error in Hurst calculation decision: {e}")
            return False
    
    def _get_fallback_hurst_values(self, df_length: int) -> Dict:
        """Hurst 계산 실패 시 대체값 생성"""
        # 기본값 0.5 (랜덤워크)로 채운 배열
        fallback_hurst = np.full(df_length, 0.5)
        fallback_smoothed = np.full(df_length, 0.5)
        
        return {
            'hurst': fallback_hurst,
            'hurst_smoothed': fallback_smoothed
        }
    
    def calculate_indicators(self, df: pd.DataFrame, current_candle: Optional[Dict] = None) -> Dict:
        """
        Calculate all indicators - Hurst 계산 최적화
        
        Args:
            df: Historical candles DataFrame
            current_candle: Current candle dict
            
        Returns:
            Dict of calculated indicators
        """
        try:
            # 현재 캔들 병합
            if current_candle and not current_candle.get('x', False):
                current_row = pd.DataFrame([{
                    'open_time': current_candle['t'],
                    'open': current_candle['o'],
                    'high': current_candle['h'],
                    'low': current_candle['l'],
                    'close': current_candle['c'],
                    'volume': current_candle['v'],
                    'close_time': current_candle['T'],
                    'quote_volume': current_candle['q'],
                    'trades': current_candle['n']
                }])
                df = pd.concat([df, current_row], ignore_index=True)
            
            df_length = len(df)
            
            # 최소 데이터 확인
            if df_length < self.min_data_for_indicators:
                self.logger.warning(f"Insufficient data for indicators: {df_length}/{self.min_data_for_indicators}")
                return {}
            
            # Hurst 계산 여부 결정
            calculate_hurst = self._should_calculate_hurst(current_candle, df_length)
            
            self.logger.debug(f"Calculating indicators - DataFrame length: {df_length}, Calculate Hurst: {calculate_hurst}")
            
            # 기본 지표들 계산
            result = self.indicators.calculate_all(
                df,
                cci_length=config.cci_length,
                cci_smoothing=config.cci_smoothing,
                rsi_length=config.rsi_length,
                supertrend_atr_length=config.supertrend_atr_length,
                supertrend_multiplier=config.supertrend_multiplier,
                psar_start=config.psar_start,
                psar_increment=config.psar_increment,
                psar_maximum=config.psar_maximum,
                vi_length=config.vi_length,
                mfi_length=config.mfi_length,
                hurst_window=config.hurst_window,
                hurst_rs_lag=config.hurst_rs_lag,
                hurst_smoothing=config.hurst_smoothing,
                vwma_short=config.vwma_short,
                vwma_mid=config.vwma_mid,
                vwma_long=config.vwma_long,
                atr_length=config.atr_length,
                adx_length=config.adx_length,
                adx_smoothing=config.adx_smoothing,
                symbol=self.symbol,
                exclude_hurst=not calculate_hurst  # Hurst 계산 여부 전달
            )
            
            # Hurst 처리 개선
            if calculate_hurst:
                # 새로 계산된 Hurst 값 확인
                if 'hurst' in result and 'hurst_smoothed' in result:
                    hurst_values = result['hurst']
                    hurst_smoothed = result['hurst_smoothed']
                    
                    # 유효한 값이 있는지 확인
                    if (len(hurst_values) > 0 and len(hurst_smoothed) > 0 and
                        not np.all(np.isnan(hurst_values)) and not np.all(np.isnan(hurst_smoothed))):
                        
                        # 캐시 업데이트
                        self.cached_hurst = {
                            'hurst': hurst_values.copy(),
                            'hurst_smoothed': hurst_smoothed.copy()
                        }
                        self.logger.info(f"Hurst values updated - Latest: {hurst_smoothed[-1]:.3f}")
                    else:
                        self.logger.warning("Calculated Hurst values are invalid, using fallback")
                        fallback = self._get_fallback_hurst_values(df_length)
                        result.update(fallback)
                else:
                    self.logger.warning("Hurst calculation returned no values, using fallback")
                    fallback = self._get_fallback_hurst_values(df_length)
                    result.update(fallback)
            else:
                # 캐시된 값 사용 또는 대체값
                if self.cached_hurst is not None:
                    # 캐시된 값을 현재 데이터 길이에 맞게 조정
                    cached_hurst = self.cached_hurst['hurst']
                    cached_smoothed = self.cached_hurst['hurst_smoothed']
                    
                    if len(cached_hurst) > 0 and len(cached_smoothed) > 0:
                        # 마지막 값으로 확장
                        last_hurst = cached_hurst[-1] if not np.isnan(cached_hurst[-1]) else 0.5
                        last_smoothed = cached_smoothed[-1] if not np.isnan(cached_smoothed[-1]) else 0.5
                        
                        result['hurst'] = np.full(df_length, last_hurst)
                        result['hurst_smoothed'] = np.full(df_length, last_smoothed)
                        
                        self.logger.debug(f"Using cached Hurst values: {last_smoothed:.3f}")
                    else:
                        fallback = self._get_fallback_hurst_values(df_length)
                        result.update(fallback)
                else:
                    # 첫 실행 시 대체값 사용
                    self.logger.debug("No cached Hurst values, using fallback")
                    fallback = self._get_fallback_hurst_values(df_length)
                    result.update(fallback)
            
            # 추세장 판별 추가
            if len(result.get('adx', [])) > 0 and len(result.get('hurst_smoothed', [])) > 0:
                result['is_trending'] = self.signal_generator.is_trending_market_by_slope(
                    result['adx'],
                    result['hurst_smoothed']
                )
            else:
                result['is_trending'] = 0
            
            # 결과 검증
            self._validate_indicators(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _validate_indicators(self, indicators: Dict):
        """지표 결과 검증"""
        try:
            # Hurst 값 검증
            if 'hurst_smoothed' in indicators:
                hurst_values = indicators['hurst_smoothed']
                if len(hurst_values) > 0:
                    latest_hurst = hurst_values[-1]
                    if np.isnan(latest_hurst) or latest_hurst == 0.0:
                        self.logger.warning(f"Invalid Hurst value detected: {latest_hurst}")
                    else:
                        self.logger.debug(f"Hurst validation passed: {latest_hurst:.3f}")
            
            # 기본 지표들 검증
            required_indicators = ['rsi', 'cci', 'mfi', 'adx']
            for indicator in required_indicators:
                if indicator in indicators:
                    values = indicators[indicator]
                    if len(values) > 0:
                        latest = values[-1]
                        if np.isnan(latest):
                            self.logger.warning(f"NaN value in {indicator}")
                        
        except Exception as e:
            self.logger.debug(f"Indicator validation error: {e}")
    
    def generate_signal(self, indicators: Dict) -> Optional[Dict]:
        """Generate trading signal from indicators"""
        try:
            # 현재 포지션 확인
            position = self.data_manager.get_position(self.symbol)
            current_position = None
            
            if position:
                current_position = position.get('side')
            
            # 신호 생성
            signal = self.signal_generator.generate_signal(indicators, current_position)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def process(self):
        """Main processing loop for this symbol"""
        self.logger.info(f"Starting indicator processor for {self.symbol}")
        
        while self.data_manager.is_running():
            try:
                # 최신 캔들 데이터 가져오기
                historical_df, current_candle = self.data_manager.get_candles(self.symbol)
                
                if historical_df is not None and len(historical_df) >= self.min_data_for_indicators:
                    # 지표 계산
                    indicators = self.calculate_indicators(historical_df, current_candle)
                    
                    if indicators:
                        # 데이터 매니저에 업데이트
                        self.data_manager.update_indicators(self.symbol, indicators)
                        
                        # 신호 생성
                        signal = self.generate_signal(indicators)
                        
                        if signal:
                            self.data_manager.update_signal(self.symbol, signal)
                            self.logger.info(f"Signal generated for {self.symbol}: {signal['action']}")
                else:
                    if historical_df is None:
                        self.logger.debug(f"No historical data for {self.symbol}")
                    else:
                        self.logger.debug(f"Insufficient data for {self.symbol}: {len(historical_df)}/{self.min_data_for_indicators}")
                
                # CPU 부하 방지
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(2)
        
        self.logger.info(f"Indicator processor stopped for {self.symbol}")


def run_indicator_processor(symbol: str):
    """
    Worker function for multiprocessing - pickle 문제 해결
    
    Args:
        symbol: Trading symbol
    """
    # Create a DataManager instance for this process (각 프로세스에서 독립 생성)
    from src.core.data_manager import DataManager
    data_manager = DataManager()  # 각 프로세스에서 독립적으로 생성
    
    processor = IndicatorProcessor(symbol, data_manager)
    processor.process()


class IndicatorProcessorManager:
    """Manages multiple indicator processors - 개선된 버전"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("indicator_manager")
        self.data_manager = data_manager
        self.processes = {}
        self.max_processes = min(config.symbol_count, mp.cpu_count())
        
    def start_processor(self, symbol: str):
        """Start indicator processor for a symbol"""
        if symbol not in self.processes:
            try:
                process = mp.Process(
                    target=run_indicator_processor,
                    args=(symbol,),  # shared_memory_manager 매개변수 제거
                    name=f"indicator_{symbol}"
                )
                process.start()
                self.processes[symbol] = process
                self.logger.info(f"Started processor for {symbol} (PID: {process.pid})")
            except Exception as e:
                self.logger.error(f"Failed to start processor for {symbol}: {e}")
    
    def stop_processor(self, symbol: str):
        """Stop indicator processor for a symbol"""
        if symbol in self.processes:
            try:
                process = self.processes[symbol]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    
                    if process.is_alive():
                        self.logger.warning(f"Force killing processor for {symbol}")
                        process.kill()
                        process.join()
                
                del self.processes[symbol]
                self.logger.info(f"Stopped processor for {symbol}")
            except Exception as e:
                self.logger.error(f"Error stopping processor for {symbol}: {e}")
    
    def update_symbols(self, symbols: List[str]):
        """Update active symbol processors"""
        current_symbols = set(self.processes.keys())
        new_symbols = set(symbols)
        
        # 제거된 심볼 프로세서 중지
        for symbol in current_symbols - new_symbols:
            self.stop_processor(symbol)
        
        # 새 심볼 프로세서 시작
        for symbol in new_symbols - current_symbols:
            self.start_processor(symbol)
        
        self.logger.info(f"Updated processors: {len(self.processes)} active")
    
    def stop_all(self):
        """Stop all processors"""
        self.logger.info("Stopping all indicator processors...")
        
        for symbol in list(self.processes.keys()):
            self.stop_processor(symbol)
        
        self.logger.info("All processors stopped")
    
    def get_status(self) -> Dict:
        """Get status of all processors"""
        status = {}
        for symbol, process in self.processes.items():
            status[symbol] = {
                'pid': process.pid,
                'alive': process.is_alive(),
                'exitcode': process.exitcode
            }
        return status