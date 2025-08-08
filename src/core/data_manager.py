"""
Central data management with multiprocessing support - 안정성 강화
"""
import multiprocessing as mp
from multiprocessing import Manager, Queue, Process
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from src.utils.logger import get_logger
from src.utils.config import config

class DataManager:
    """Central data manager for multiprocessing coordination - 안정성 강화"""
    
    def __init__(self):
        self.logger = get_logger("data_manager")
        self.manager = Manager()
        
        # 공유 데이터 구조
        self.candle_data = self.manager.dict()  # {symbol: {'historical': df, 'current': dict}}
        self.indicators = self.manager.dict()   # {symbol: dict of indicators}
        self.signals = self.manager.dict()      # {symbol: signal_dict}
        self.positions = self.manager.dict()    # {symbol: position_info}
        self.symbol_list = self.manager.list()  # Active symbols
        
        # 상태 및 제어
        self.running = mp.Value('b', True)
        self.last_update = self.manager.dict()  # {symbol: timestamp}
        self.health_status = self.manager.dict()  # {symbol: health_info}
        
        # 스레드 안전성을 위한 락
        self.locks = {
            'candle': mp.Lock(),
            'indicator': mp.Lock(),
            'signal': mp.Lock(),
            'position': mp.Lock(),
            'symbol': mp.Lock(),
            'health': mp.Lock()
        }
        
        # 통신 큐
        self.ws_queue = Queue(maxsize=1000)  # WebSocket 데이터
        self.rest_queue = Queue(maxsize=500)  # REST API 데이터
        self.signal_queue = Queue(maxsize=200)  # 거래 신호
        
        # 성능 모니터링
        self.stats = self.manager.dict({
            'candle_updates': 0,
            'indicator_updates': 0,
            'signal_updates': 0,
            'errors': 0
        })
        
        self.logger.info("DataManager initialized with enhanced stability")
    
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
        Update candle data for a symbol - 안전성 강화
        
        Args:
            symbol: Trading symbol
            historical_df: Historical candles DataFrame
            current_candle: Current candle dict
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['candle']:
                # 심볼 데이터 초기화
                if symbol not in self.candle_data:
                    self.candle_data[symbol] = self.manager.dict({
                        'historical': None,
                        'current': None,
                        'last_update': time.time()
                    })
                
                data = self.candle_data[symbol]
                updated = False
                
                # Historical 데이터 업데이트
                if historical_df is not None:
                    try:
                        # DataFrame을 dict로 안전하게 변환
                        df_dict = historical_df.to_dict('records')
                        data['historical'] = df_dict
                        data['historical_meta'] = {
                            'length': len(historical_df),
                            'timestamp': time.time(),
                            'columns': list(historical_df.columns)
                        }
                        updated = True
                    except Exception as e:
                        self.logger.error(f"Error converting DataFrame for {symbol}: {e}")
                        return False
                
                # Current 캔들 업데이트
                if current_candle is not None:
                    try:
                        data['current'] = dict(current_candle)  # 안전한 복사
                        data['current_timestamp'] = time.time()
                        updated = True
                    except Exception as e:
                        self.logger.error(f"Error updating current candle for {symbol}: {e}")
                        return False
                
                if updated:
                    data['last_update'] = time.time()
                    self.candle_data[symbol] = data
                    self.last_update[symbol] = time.time()
                    self._increment_stat('candle_updates')
                    
                    # 건강 상태 업데이트
                    self._update_health_status(symbol, 'candle_update', True)
                
                return updated
                
        except Exception as e:
            self._handle_error(f"update_candles({symbol})", e)
            self._update_health_status(symbol, 'candle_update', False, str(e))
            return False
    
    def get_candles(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Get candle data for a symbol - 안전성 강화
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (historical_df, current_candle)
        """
        try:
            with self.locks['candle']:
                if symbol not in self.candle_data:
                    return None, None
                
                data = self.candle_data[symbol]
                
                # Historical 데이터 복원
                historical = None
                if data.get('historical') is not None:
                    try:
                        df_records = data['historical']
                        if df_records:
                            historical = pd.DataFrame(df_records)
                            # 필요시 데이터 타입 복원
                            if 'historical_meta' in data:
                                expected_cols = data['historical_meta'].get('columns', [])
                                if set(expected_cols) <= set(historical.columns):
                                    historical = historical[expected_cols]
                    except Exception as e:
                        self.logger.error(f"Error restoring DataFrame for {symbol}: {e}")
                        historical = None
                
                # Current 캔들 복원
                current = None
                if data.get('current') is not None:
                    try:
                        current = dict(data['current'])
                    except Exception as e:
                        self.logger.error(f"Error restoring current candle for {symbol}: {e}")
                        current = None
                
                return historical, current
                
        except Exception as e:
            self._handle_error(f"get_candles({symbol})", e)
            return None, None
    
    def update_indicators(self, symbol: str, indicators: Dict) -> bool:
        """
        Update indicators for a symbol - 타입 안전성 강화
        
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
                
                self.indicators[symbol] = safe_indicators
                self._increment_stat('indicator_updates')
                self._update_health_status(symbol, 'indicator_update', True)
                
                return True
                
        except Exception as e:
            self._handle_error(f"update_indicators({symbol})", e)
            self._update_health_status(symbol, 'indicator_update', False, str(e))
            return False
    
    def get_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get indicators for a symbol - 안전한 복사본 반환
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators or None
        """
        try:
            with self.locks['indicator']:
                indicators = self.indicators.get(symbol)
                if indicators:
                    # 안전한 복사본 반환
                    return dict(indicators)
                return None
        except Exception as e:
            self._handle_error(f"get_indicators({symbol})", e)
            return None
    
    def update_signal(self, symbol: str, signal: Dict) -> bool:
        """
        Update trading signal for a symbol - 큐 오버플로우 방지
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['signal']:
                # 신호에 타임스탬프 추가
                safe_signal = dict(signal)
                safe_signal['timestamp'] = time.time()
                safe_signal['symbol'] = symbol
                
                self.signals[symbol] = safe_signal
                
                # 큐에 추가 (논블로킹)
                try:
                    self.signal_queue.put_nowait((symbol, safe_signal))
                except:
                    # 큐가 가득 찬 경우 가장 오래된 것 제거 후 추가
                    try:
                        self.signal_queue.get_nowait()
                        self.signal_queue.put_nowait((symbol, safe_signal))
                    except:
                        self.logger.warning(f"Signal queue full, dropping signal for {symbol}")
                
                self._increment_stat('signal_updates')
                self._update_health_status(symbol, 'signal_update', True)
                
                return True
                
        except Exception as e:
            self._handle_error(f"update_signal({symbol})", e)
            self._update_health_status(symbol, 'signal_update', False, str(e))
            return False
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get current signal for a symbol"""
        try:
            with self.locks['signal']:
                signal = self.signals.get(symbol)
                if signal:
                    return dict(signal)
                return None
        except Exception as e:
            self._handle_error(f"get_signal({symbol})", e)
            return None
    
    def update_position(self, symbol: str, position_info: Optional[Dict]) -> bool:
        """
        Update position information for a symbol
        
        Args:
            symbol: Trading symbol
            position_info: Position information or None to clear
            
        Returns:
            bool: Success status
        """
        try:
            with self.locks['position']:
                if position_info is None:
                    if symbol in self.positions:
                        del self.positions[symbol]
                else:
                    safe_position = dict(position_info)
                    safe_position['last_update'] = time.time()
                    self.positions[symbol] = safe_position
                
                self._update_health_status(symbol, 'position_update', True)
                return True
                
        except Exception as e:
            self._handle_error(f"update_position({symbol})", e)
            self._update_health_status(symbol, 'position_update', False, str(e))
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position information for a symbol"""
        try:
            with self.locks['position']:
                position = self.positions.get(symbol)
                if position:
                    return dict(position)
                return None
        except Exception as e:
            self._handle_error(f"get_position({symbol})", e)
            return None
    
    def get_all_positions(self) -> Dict:
        """Get all open positions"""
        try:
            with self.locks['position']:
                return {symbol: dict(pos) for symbol, pos in self.positions.items()}
        except Exception as e:
            self._handle_error("get_all_positions", e)
            return {}
    
    def update_symbol_list(self, symbols: List[str]) -> bool:
        """
        Update the active symbol list
        
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
                
                self.symbol_list[:] = clean_symbols
                self.logger.info(f"Symbol list updated: {clean_symbols}")
                
                # 새 심볼들의 건강 상태 초기화
                for symbol in clean_symbols:
                    if symbol not in self.health_status:
                        self._init_health_status(symbol)
                
                return True
                
        except Exception as e:
            self._handle_error("update_symbol_list", e)
            return False
    
    def get_symbol_list(self) -> List[str]:
        """Get the active symbol list"""
        try:
            with self.locks['symbol']:
                return list(self.symbol_list)
        except Exception as e:
            self._handle_error("get_symbol_list", e)
            return []
    
    def _init_health_status(self, symbol: str):
        """심볼 건강 상태 초기화"""
        try:
            with self.locks['health']:
                self.health_status[symbol] = {
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
                if symbol not in self.health_status:
                    self._init_health_status(symbol)
                
                health = self.health_status[symbol]
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
    
    def add_ws_data(self, data: Dict) -> None:
        """Add WebSocket data to queue"""
        try:
            self.ws_queue.put_nowait(data)
        except:
            # 큐가 가득 찬 경우 오래된 데이터 제거
            try:
                self.ws_queue.get_nowait()
                self.ws_queue.put_nowait(data)
            except:
                pass
    
    def add_rest_data(self, data: Dict) -> None:
        """Add REST API data to queue"""
        try:
            self.rest_queue.put_nowait(data)
        except:
            try:
                self.rest_queue.get_nowait()
                self.rest_queue.put_nowait(data)
            except:
                pass
    
    def get_ws_data(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get WebSocket data from queue"""
        try:
            return self.ws_queue.get(timeout=timeout)
        except:
            return None
    
    def get_rest_data(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get REST API data from queue"""
        try:
            return self.rest_queue.get(timeout=timeout)
        except:
            return None
    
    def get_signal_from_queue(self, timeout: float = 0.1) -> Optional[Tuple[str, Dict]]:
        """Get signal from queue"""
        try:
            return self.signal_queue.get(timeout=timeout)
        except:
            return None
    
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
            
            # 큐 정리
            self._drain_queue(self.ws_queue, "WebSocket")
            self._drain_queue(self.rest_queue, "REST")
            self._drain_queue(self.signal_queue, "Signal")
            
            # 매니저 종료
            try:
                self.manager.shutdown()
            except Exception as e:
                self.logger.warning(f"Manager shutdown error: {e}")
            
            self.logger.info("DataManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _drain_queue(self, queue: Queue, name: str):
        """큐 비우기"""
        try:
            drained = 0
            while True:
                try:
                    queue.get_nowait()
                    drained += 1
                except:
                    break
            
            if drained > 0:
                self.logger.info(f"Drained {drained} items from {name} queue")
                
            queue.close()
            
        except Exception as e:
            self.logger.warning(f"Error draining {name} queue: {e}")
    
    def emergency_reset(self):
        """
        Emergency reset of all data structures
        """
        try:
            self.logger.warning("Performing emergency reset of DataManager")
            
            # 모든 락을 획득하여 안전하게 리셋
            with self.locks['candle'], self.locks['indicator'], self.locks['signal'], \
                 self.locks['position'], self.locks['health']:
                
                # 모든 데이터 구조 초기화
                self.candle_data.clear()
                self.indicators.clear()
                self.signals.clear()
                self.positions.clear()
                self.health_status.clear()
                self.last_update.clear()
                
                # 통계 리셋
                self.stats.clear()
                self.stats.update({
                    'candle_updates': 0,
                    'indicator_updates': 0,
                    'signal_updates': 0,
                    'errors': 0,
                    'reset_time': time.time()
                })
                
                # 심볼별 건강 상태 재초기화
                for symbol in self.symbol_list:
                    self._init_health_status(symbol)
            
            self.logger.warning("Emergency reset completed")
            
        except Exception as e:
            self.logger.error(f"Error during emergency reset: {e}")
            # 최후의 수단: 새 매니저 생성
            try:
                self.manager = Manager()
                self.logger.error("Created new manager instance after reset failure")
            except Exception as e2:
                self.logger.critical(f"Failed to create new manager: {e2}")