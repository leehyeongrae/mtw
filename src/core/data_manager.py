"""
Central data management with shared memory support - mmap 기반으로 개선
"""
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.shared_memory import SharedMemoryManager

class DataManager:
    """Central data manager for multiprocessing coordination - mmap 기반으로 개선"""
    
    def __init__(self):
        self.logger = get_logger("data_manager")
        
        # Shared Memory Manager 초기화
        self.shared_memory = SharedMemoryManager()
        
        # 상태 및 제어 (기본 multiprocessing 요소들은 유지)
        self.running = mp.Value('b', True)
        
        # 스레드 안전성을 위한 락
        self.locks = {
            'candle': mp.Lock(),
            'indicator': mp.Lock(),
            'signal': mp.Lock(),
            'position': mp.Lock(),
            'symbol': mp.Lock(),
            'health': mp.Lock()
        }
        
        # 성능 모니터링 (로컬 변수로 변경)
        self.stats = {
            'candle_updates': 0,
            'indicator_updates': 0,
            'signal_updates': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # 로컬 캐시 (성능 향상을 위해)
        self._symbol_list = []
        self._last_update = {}
        self._health_status = {}
        
        self.logger.info("DataManager initialized with shared memory support")
    
    def _increment_stat(self, stat_name: str):
        """통계 증가"""
        try:
            current = self.stats.get(stat_name, 0)
            self.stats[stat_name] = current + 1
        except:
            pass  # 통계는 실패해도 시스템에 영향 없음
    
    def _handle_error(self, operation: str, error: Exception):
        """에러 처리 및 로깅"""
        self.logger.error(f"Error in {operation}: {error}")
        self._increment_stat('errors')
    
    def update_candles(self, symbol: str, historical_df: Optional[pd.DataFrame] = None,
                      current_candle: Optional[Dict] = None) -> bool:
        """
        Update candle data for a symbol using shared memory
        
        Args:
            symbol: Trading symbol
            historical_df: Historical candles DataFrame
            current_candle: Current candle dict
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['candle']:
                # Get candles handler for this symbol
                candles_handler = self.shared_memory.get_candles_handler(symbol)
                
                # Write candle data to shared memory
                success = candles_handler.write_candles(historical_df, current_candle)
                
                if success:
                    self._last_update[symbol] = time.time()
                    self._increment_stat('candle_updates')
                    self._update_health_status(symbol, 'candle_update', True)
                
                return success
                
        except Exception as e:
            self._handle_error(f"update_candles({symbol})", e)
            self._update_health_status(symbol, 'candle_update', False, str(e))
            return False
    
    def get_candles(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Get candle data for a symbol from shared memory
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (historical_df, current_candle)
        """
        try:
            with self.locks['candle']:
                # Get candles handler for this symbol
                candles_handler = self.shared_memory.get_candles_handler(symbol)
                
                # Read candle data from shared memory
                historical_df, current_candle = candles_handler.read_candles()
                
                return historical_df, current_candle
                
        except Exception as e:
            self._handle_error(f"get_candles({symbol})", e)
            return None, None
    
    def update_indicators(self, symbol: str, indicators: Dict) -> bool:
        """
        Update indicators for a symbol using shared memory
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of calculated indicators
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['indicator']:
                # numpy 배열을 리스트로 안전하게 변환
                safe_indicators = {}
                for key, value in indicators.items():
                    try:
                        if isinstance(value, np.ndarray):
                            # NaN과 무한값 처리
                            clean_value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                            safe_indicators[key] = clean_value.tolist()
                        elif isinstance(value, (list, tuple)):
                            # 리스트/튜플의 각 요소도 확인
                            safe_list = []
                            for item in value:
                                if isinstance(item, (int, float)) and np.isfinite(item):
                                    safe_list.append(float(item))
                                else:
                                    safe_list.append(0.0)
                            safe_indicators[key] = safe_list
                        elif isinstance(value, (int, float)):
                            if np.isfinite(value):
                                safe_indicators[key] = float(value)
                            else:
                                safe_indicators[key] = 0.0
                        else:
                            safe_indicators[key] = value
                    except Exception as e:
                        self.logger.warning(f"Error processing indicator {key} for {symbol}: {e}")
                        safe_indicators[key] = 0.0
                
                # Get indicators handler and write to shared memory
                indicators_handler = self.shared_memory.get_indicators_handler(symbol)
                success = indicators_handler.write_indicators(safe_indicators)
                
                if success:
                    self._increment_stat('indicator_updates')
                    self._update_health_status(symbol, 'indicator_update', True)
                
                return success
                
        except Exception as e:
            self._handle_error(f"update_indicators({symbol})", e)
            self._update_health_status(symbol, 'indicator_update', False, str(e))
            return False
    
    def get_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get indicators for a symbol from shared memory
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators or None
        """
        try:
            with self.locks['indicator']:
                # Get indicators handler and read from shared memory
                indicators_handler = self.shared_memory.get_indicators_handler(symbol)
                indicators = indicators_handler.read_indicators()
                
                return indicators
                
        except Exception as e:
            self._handle_error(f"get_indicators({symbol})", e)
            return None
    
    def update_signal(self, symbol: str, signal: Dict) -> bool:
        """
        Update trading signal for a symbol using shared memory
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['signal']:
                # Get signals handler and write to shared memory
                signals_handler = self.shared_memory.get_signals_handler(symbol)
                success = signals_handler.write_signal(signal)
                
                if success:
                    self._increment_stat('signal_updates')
                    self._update_health_status(symbol, 'signal_update', True)
                
                return success
                
        except Exception as e:
            self._handle_error(f"update_signal({symbol})", e)
            self._update_health_status(symbol, 'signal_update', False, str(e))
            return False
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get current signal for a symbol from shared memory"""
        try:
            with self.locks['signal']:
                # Get signals handler and read from shared memory
                signals_handler = self.shared_memory.get_signals_handler(symbol)
                signal = signals_handler.read_signal()
                
                return signal
                
        except Exception as e:
            self._handle_error(f"get_signal({symbol})", e)
            return None
    
    def update_position(self, symbol: str, position_info: Optional[Dict]) -> bool:
        """
        Update position information for a symbol using shared memory
        
        Args:
            symbol: Trading symbol
            position_info: Position information or None to clear
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['position']:
                # Read current positions from shared memory
                current_positions = self.shared_memory.metadata.read_positions()
                
                if position_info is None:
                    # Remove position
                    if symbol in current_positions:
                        del current_positions[symbol]
                else:
                    # Update position
                    safe_position = dict(position_info)
                    safe_position['last_update'] = time.time()
                    current_positions[symbol] = safe_position
                
                # Write back to shared memory
                success = self.shared_memory.metadata.write_positions(current_positions)
                
                if success:
                    self._update_health_status(symbol, 'position_update', True)
                
                return success
                
        except Exception as e:
            self._handle_error(f"update_position({symbol})", e)
            self._update_health_status(symbol, 'position_update', False, str(e))
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position information for a symbol from shared memory"""
        try:
            with self.locks['position']:
                positions = self.shared_memory.metadata.read_positions()
                return positions.get(symbol)
        except Exception as e:
            self._handle_error(f"get_position({symbol})", e)
            return None
    
    def get_all_positions(self) -> Dict:
        """Get all open positions from shared memory"""
        try:
            with self.locks['position']:
                return self.shared_memory.metadata.read_positions()
        except Exception as e:
            self._handle_error("get_all_positions", e)
            return {}
    
    def update_symbol_list(self, symbols: List[str]) -> bool:
        """
        Update the active symbol list using shared memory
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['symbol']:
                # 유효성 검사
                if not symbols or not isinstance(symbols, list):
                    self.logger.error("Invalid symbol list provided")
                    return False
                
                # 중복 제거 및 정리
                clean_symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s.strip()]))
                
                if not clean_symbols:
                    self.logger.error("No valid symbols in list")
                    return False
                
                # Write to shared memory
                success = self.shared_memory.metadata.write_symbol_list(clean_symbols)
                
                if success:
                    # Update local cache
                    self._symbol_list = clean_symbols
                    self.logger.info(f"Symbol list updated: {clean_symbols}")
                    
                    # 새 심볼들의 건강 상태 초기화
                    for symbol in clean_symbols:
                        if symbol not in self._health_status:
                            self._init_health_status(symbol)
                
                return success
                
        except Exception as e:
            self._handle_error("update_symbol_list", e)
            return False
    
    def get_symbol_list(self) -> List[str]:
        """Get the active symbol list from shared memory"""
        try:
            with self.locks['symbol']:
                return self.shared_memory.metadata.read_symbol_list()
        except Exception as e:
            self._handle_error("get_symbol_list", e)
            return []
    
    def _init_health_status(self, symbol: str):
        """심볼 건강 상태 초기화"""
        try:
            with self.locks['health']:
                self._health_status[symbol] = {
                    'last_candle_update': 0,
                    'last_indicator_update': 0,
                    'last_signal_update': 0,
                    'last_position_update': 0,
                    'errors': [],
                    'status': 'initializing'
                }
        except:
            pass
    
    def _update_health_status(self, symbol: str, operation: str, success: bool, error_msg: str = None):
        """건강 상태 업데이트"""
        try:
            with self.locks['health']:
                if symbol not in self._health_status:
                    self._init_health_status(symbol)
                
                health = self._health_status[symbol]
                current_time = time.time()
                
                if success:
                    health[f'last_{operation}'] = current_time
                    health['status'] = 'healthy'
                else:
                    if error_msg:
                        health['errors'].append({
                            'operation': operation,
                            'error': error_msg,
                            'timestamp': current_time
                        })
                        # 최근 10개 에러만 유지
                        health['errors'] = health['errors'][-10:]
                    health['status'] = 'error'
        except:
            pass
    
    def is_running(self) -> bool:
        """Check if system is running"""
        try:
            return self.running.value
        except:
            return False
    
    def stop(self) -> None:
        """Stop the data manager"""
        try:
            self.running.value = False
            self.logger.info("DataManager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping DataManager: {e}")
    
    def get_status(self, symbol: str) -> Dict:
        """
        Get complete status for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with complete status information
        """
        try:
            historical, current = self.get_candles(symbol)
            indicators = self.get_indicators(symbol)
            signal = self.get_signal(symbol)
            position = self.get_position(symbol)
            
            # 기본 상태 정보
            status = {
                'symbol': symbol,
                'timestamp': time.time(),
                'has_historical': historical is not None,
                'has_current': current is not None,
                'has_indicators': indicators is not None,
                'has_signal': signal is not None,
                'has_position': position is not None,
                'last_update': self.last_update.get(symbol, 0)
            }
            
            # 데이터 세부 정보
            if historical is not None:
                status['historical_length'] = len(historical)
                status['oldest_candle'] = historical.iloc[0]['open_time'] if len(historical) > 0 else 0
                status['newest_candle'] = historical.iloc[-1]['open_time'] if len(historical) > 0 else 0
            
            if current:
                status['current_price'] = current.get('c', 0)
                status['current_volume'] = current.get('v', 0)
                status['candle_closed'] = current.get('x', False)
            
            if indicators:
                status['market_type'] = 'trending' if indicators.get('is_trending') else 'ranging'
                # 주요 지표값들
                if 'hurst_smoothed' in indicators and len(indicators['hurst_smoothed']) > 0:
                    status['hurst'] = indicators['hurst_smoothed'][-1]
                if 'adx' in indicators and len(indicators['adx']) > 0:
                    status['adx'] = indicators['adx'][-1]
                if 'rsi' in indicators and len(indicators['rsi']) > 0:
                    status['rsi'] = indicators['rsi'][-1]
            
            if signal:
                status['signal_action'] = signal.get('action')
                status['signal_reason'] = signal.get('reason')
                status['signal_timestamp'] = signal.get('timestamp', 0)
            
            if position:
                status['position_side'] = position.get('side')
                status['position_size'] = position.get('size')
                status['position_pnl'] = position.get('pnl_percent', 0)
            
            # 건강 상태 정보 추가
            health = self.health_status.get(symbol, {})
            status['health'] = {
                'status': health.get('status', 'unknown'),
                'last_candle_update': health.get('last_candle_update', 0),
                'last_indicator_update': health.get('last_indicator_update', 0),
                'error_count': len(health.get('errors', []))
            }
            
            return status
            
        except Exception as e:
            self._handle_error(f"get_status({symbol})", e)
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_system_stats(self) -> Dict:
        """
        Get overall system statistics
        
        Returns:
            Dict: System statistics
        """
        try:
            stats = dict(self.stats)
            
            # 추가 통계 정보
            stats.update({
                'active_symbols': len(self.symbol_list),
                'total_positions': len(self.positions),
                'queue_sizes': {
                    'ws_queue': self.ws_queue.qsize(),
                    'rest_queue': self.rest_queue.qsize(),
                    'signal_queue': self.signal_queue.qsize()
                },
                'uptime': time.time() - stats.get('start_time', time.time()),
                'is_running': self.is_running()
            })
            
            # 심볼별 건강 상태 요약
            healthy_symbols = 0
            error_symbols = 0
            
            for symbol in self.symbol_list:
                health = self.health_status.get(symbol, {})
                status = health.get('status', 'unknown')
                if status == 'healthy':
                    healthy_symbols += 1
                elif status == 'error':
                    error_symbols += 1
            
            stats['health_summary'] = {
                'healthy': healthy_symbols,
                'error': error_symbols,
                'unknown': len(self.symbol_list) - healthy_symbols - error_symbols
            }
            
            return stats
            
        except Exception as e:
            self._handle_error("get_system_stats", e)
            return {'error': str(e)}
    
    def get_health_report(self) -> Dict:
        """
        Get detailed health report for all symbols
        
        Returns:
            Dict: Health report
        """
        try:
            with self.locks['health']:
                report = {
                    'timestamp': time.time(),
                    'symbols': {},
                    'summary': {
                        'total': len(self.symbol_list),
                        'healthy': 0,
                        'warning': 0,
                        'error': 0
                    }
                }
                
                current_time = time.time()
                
                for symbol in self.symbol_list:
                    health = self.health_status.get(symbol, {})
                    
                    symbol_health = {
                        'status': health.get('status', 'unknown'),
                        'last_updates': {
                            'candle': current_time - health.get('last_candle_update', 0),
                            'indicator': current_time - health.get('last_indicator_update', 0),
                            'signal': current_time - health.get('last_signal_update', 0)
                        },
                        'recent_errors': len([e for e in health.get('errors', []) 
                                           if current_time - e.get('timestamp', 0) < 3600])  # 1시간 이내
                    }
                    
                    # 건강 상태 판정
                    if symbol_health['status'] == 'healthy':
                        # 최근 업데이트 확인
                        if (symbol_health['last_updates']['candle'] > 1800 or  # 30분
                            symbol_health['last_updates']['indicator'] > 1800):
                            symbol_health['status'] = 'warning'
                            symbol_health['warning'] = 'Stale data'
                    
                    report['symbols'][symbol] = symbol_health
                    
                    # 요약 업데이트
                    if symbol_health['status'] == 'healthy':
                        report['summary']['healthy'] += 1
                    elif symbol_health['status'] == 'warning':
                        report['summary']['warning'] += 1
                    else:
                        report['summary']['error'] += 1
                
                return report
                
        except Exception as e:
            self._handle_error("get_health_report", e)
            return {'error': str(e)}
    
    def cleanup_stale_data(self, max_age_seconds: int = 3600):
        """
        Clean up stale data and reset error counts
        
        Args:
            max_age_seconds: Maximum age for data in seconds
        """
        try:
            current_time = time.time()
            cleanup_count = 0
            
            # 오래된 캔들 데이터 정리
            with self.locks['candle']:
                for symbol in list(self.candle_data.keys()):
                    data = self.candle_data[symbol]
                    last_update = data.get('last_update', 0)
                    
                    if current_time - last_update > max_age_seconds:
                        del self.candle_data[symbol]
                        cleanup_count += 1
                        self.logger.info(f"Cleaned stale candle data for {symbol}")
            
            # 오래된 지표 데이터 정리
            with self.locks['indicator']:
                for symbol in list(self.indicators.keys()):
                    if symbol not in self.symbol_list:
                        del self.indicators[symbol]
                        cleanup_count += 1
            
            # 오래된 에러 기록 정리
            with self.locks['health']:
                for symbol in self.health_status:
                    health = self.health_status[symbol]
                    if 'errors' in health:
                        # 1시간 이내 에러만 유지
                        recent_errors = [e for e in health['errors'] 
                                       if current_time - e.get('timestamp', 0) < 3600]
                        health['errors'] = recent_errors
            
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} stale data entries")
                
        except Exception as e:
            self._handle_error("cleanup_stale_data", e)
    
    def force_refresh_symbol(self, symbol: str) -> bool:
        """
        Force refresh all data for a specific symbol
        
        Args:
            symbol: Trading symbol to refresh
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Force refreshing data for {symbol}")
            
            # 캔들 데이터 초기화
            with self.locks['candle']:
                if symbol in self.candle_data:
                    del self.candle_data[symbol]
            
            # 지표 데이터 초기화
            with self.locks['indicator']:
                if symbol in self.indicators:
                    del self.indicators[symbol]
            
            # 신호 데이터 초기화
            with self.locks['signal']:
                if symbol in self.signals:
                    del self.signals[symbol]
            
            # 건강 상태 초기화
            self._init_health_status(symbol)
            
            self.logger.info(f"Successfully refreshed data for {symbol}")
            return True
            
        except Exception as e:
            self._handle_error(f"force_refresh_symbol({symbol})", e)
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.logger.info("Starting DataManager cleanup...")
            
            # 시스템 중지
            self.stop()
            
            # Shared memory cleanup
            self.shared_memory.cleanup_all()
            
            self.logger.info("DataManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def emergency_reset(self):
        """
        Emergency reset of all data structures
        """
        try:
            self.logger.warning("Performing emergency reset of DataManager")
            
            # 모든 락을 획득하여 안전하게 리셋
            with self.locks['candle'], self.locks['indicator'], self.locks['signal'], \
                 self.locks['position'], self.locks['health']:
                
                # Shared memory cleanup and reset
                self.shared_memory.cleanup_all()
                self.shared_memory = SharedMemoryManager()
                
                # 로컬 데이터 구조 초기화
                self._symbol_list.clear()
                self._last_update.clear()
                self._health_status.clear()
                
                # 통계 리셋
                self.stats = {
                    'candle_updates': 0,
                    'indicator_updates': 0,
                    'signal_updates': 0,
                    'errors': 0,
                    'reset_time': time.time()
                }
            
            self.logger.warning("Emergency reset completed")
            
        except Exception as e:
            self.logger.error(f"Error during emergency reset: {e}")