"""
Main entry point for the trading system - 안정성 강화 버전
"""
import asyncio
import signal
import sys
import time
from typing import List
from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.status_display import StatusDisplay
from src.core.data_manager import DataManager
from src.core.symbol_manager import SymbolManager
from src.exchange.binance_rest import BinanceREST
from src.exchange.binance_ws import BinanceWebSocket
from src.strategy.indicator_processor import IndicatorProcessorManager
from src.strategy.trading_engine import TradingEngine
from src.telegram.bot import TelegramBot

class TradingSystem:
    """Main trading system coordinator - 강화된 안정성"""
    
    def __init__(self):
        self.logger = get_logger("main")
        self.logger.info("Initializing enhanced trading system...")
        
        # 초기화 상태 추적
        self.initialization_success = False
        self.components_started = False
        self.shutdown_in_progress = False
        
        # 핵심 컴포넌트
        self.data_manager = None
        self.symbol_manager = None
        
        # 거래소 연결
        self.rest_client = None
        self.ws_client = None
        
        # 거래 컴포넌트
        self.indicator_manager = None
        self.trading_engine = None
        
        # 모니터링
        self.telegram_bot = None
        self.status_display = None
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Signal handlers configured")
        except Exception as e:
            self.logger.error(f"Failed to setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if not self.shutdown_in_progress:
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_in_progress = True
            
            # 비동기 컨텍스트에서 실행되도록 처리
            try:
                asyncio.create_task(self._async_shutdown())
            except RuntimeError:
                # 이벤트 루프가 없는 경우 동기 종료
                self.shutdown()
                sys.exit(0)
    
    async def _async_shutdown(self):
        """비동기 종료 처리"""
        await asyncio.sleep(0.1)  # 현재 작업 완료 대기
        self.shutdown()
        sys.exit(0)
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing core components...")
            
            # 1. 데이터 매니저 (가장 먼저)
            self.data_manager = DataManager()
            self.logger.info("✓ DataManager initialized")
            
            # 2. 심볼 매니저 (중요: 거래량 기준 심볼 선택)
            self.symbol_manager = SymbolManager()
            self.logger.info("✓ SymbolManager initialized")
            
            # 3. 거래소 클라이언트
            self.rest_client = BinanceREST(self.data_manager)
            self.logger.info("✓ REST client initialized")
            
            self.ws_client = BinanceWebSocket(self.data_manager)
            self.logger.info("✓ WebSocket client initialized")
            
            # 4. 거래 컴포넌트
            self.indicator_manager = IndicatorProcessorManager(self.data_manager)
            self.logger.info("✓ IndicatorProcessorManager initialized")
            
            self.trading_engine = TradingEngine(self.data_manager)
            self.logger.info("✓ TradingEngine initialized")
            
            # 5. 모니터링 컴포넌트
            self.telegram_bot = TelegramBot(self.data_manager)
            self.logger.info("✓ TelegramBot initialized")
            
            self.status_display = StatusDisplay(self.data_manager)
            self.logger.info("✓ StatusDisplay initialized")
            
            self.initialization_success = True
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            if not self.initialization_success:
                self.logger.error("Components not properly initialized")
                return False
            
            self.logger.info("Starting system initialization...")
            
            # 1. 심볼 리스트 가져오기 (중요!)
            self.logger.info("Fetching top symbols by volume...")
            max_attempts = 3
            symbols = None
            
            for attempt in range(max_attempts):
                try:
                    symbols = self.symbol_manager.update_symbol_list()
                    if symbols and len(symbols) >= 3:  # 최소 3개 심볼
                        break
                    else:
                        self.logger.warning(f"Insufficient symbols returned: {symbols}")
                        if attempt < max_attempts - 1:
                            self.logger.info(f"Retrying... (attempt {attempt + 2}/{max_attempts})")
                            await asyncio.sleep(5)
                except Exception as e:
                    self.logger.error(f"Symbol fetch attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(10)
                    else:
                        raise
            
            if not symbols or len(symbols) < 3:
                self.logger.error("Failed to get sufficient trading symbols")
                return False
            
            self.logger.info(f"Successfully selected {len(symbols)} symbols: {symbols}")
            
            # 2. 데이터 매니저에 심볼 업데이트
            if not self.data_manager.update_symbol_list(symbols):
                self.logger.error("Failed to update symbol list in data manager")
                return False
            
            # 3. 지표 프로세서 시작
            self.logger.info("Starting indicator processors...")
            self.indicator_manager.update_symbols(symbols)
            
            # 4. 초기 데이터 로딩 (REST API)
            self.logger.info("Loading initial market data...")
            await self._load_initial_data(symbols)
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    async def _load_initial_data(self, symbols: List[str]):
        """초기 시장 데이터 로딩"""
        try:
            self.logger.info("Loading initial candle data for all symbols...")
            
            # REST 클라이언트로 초기 데이터 로딩
            await self.rest_client.update_all_symbols()
            
            # 데이터 로딩 확인
            loaded_symbols = 0
            for symbol in symbols:
                historical, _ = self.data_manager.get_candles(symbol)
                if historical is not None and len(historical) > 0:
                    loaded_symbols += 1
                    self.logger.info(f"✓ Loaded {len(historical)} candles for {symbol}")
                else:
                    self.logger.warning(f"✗ No data loaded for {symbol}")
            
            self.logger.info(f"Initial data loaded for {loaded_symbols}/{len(symbols)} symbols")
            
            if loaded_symbols == 0:
                raise Exception("No initial data loaded for any symbol")
            
        except Exception as e:
            self.logger.error(f"Failed to load initial data: {e}")
            raise
    
    async def update_symbols_periodically(self):
        """Update symbol list periodically (daily)"""
        update_interval = 86400  # 24 hours
        
        while self.data_manager.is_running() and not self.shutdown_in_progress:
            try:
                await asyncio.sleep(update_interval)
                
                if self.shutdown_in_progress:
                    break
                
                self.logger.info("Performing daily symbol update...")
                
                # 새 심볼 리스트 가져오기
                new_symbols = self.symbol_manager.update_symbol_list()
                
                if new_symbols and len(new_symbols) >= 3:
                    current_symbols = self.data_manager.get_symbol_list()
                    
                    if set(new_symbols) != set(current_symbols):
                        self.logger.info(f"Symbol list changed: {current_symbols} -> {new_symbols}")
                        
                        # 데이터 매니저 업데이트
                        self.data_manager.update_symbol_list(new_symbols)
                        
                        # 지표 프로세서 업데이트
                        self.indicator_manager.update_symbols(new_symbols)
                        
                        # WebSocket 연결 업데이트
                        self.ws_client.update_symbols(new_symbols)
                        
                        self.logger.info("Symbol update completed successfully")
                    else:
                        self.logger.info("Symbol list unchanged")
                else:
                    self.logger.warning("Failed to update symbol list, keeping current symbols")
                    
            except Exception as e:
                self.logger.error(f"Error in periodic symbol update: {e}")
    
    async def system_health_monitor(self):
        """시스템 건강 상태 모니터링"""
        check_interval = 300  # 5분마다 확인
        
        while self.data_manager.is_running() and not self.shutdown_in_progress:
            try:
                await asyncio.sleep(check_interval)
                
                if self.shutdown_in_progress:
                    break
                
                # 건강 상태 보고서 생성
                health_report = self.data_manager.get_health_report()
                
                # 문제가 있는 심볼 확인
                problem_symbols = []
                for symbol, health in health_report.get('symbols', {}).items():
                    if health['status'] in ['error', 'warning']:
                        problem_symbols.append(symbol)
                
                if problem_symbols:
                    self.logger.warning(f"Health issues detected for symbols: {problem_symbols}")
                    
                    # 심각한 문제가 있는 경우 데이터 정리
                    if len(problem_symbols) > len(health_report['symbols']) * 0.5:
                        self.logger.warning("More than 50% of symbols have issues, performing cleanup")
                        self.data_manager.cleanup_stale_data()
                
                # 시스템 통계 로깅
                stats = self.data_manager.get_system_stats()
                self.logger.info(
                    f"System stats - Updates: {stats.get('candle_updates', 0)}, "
                    f"Errors: {stats.get('errors', 0)}, "
                    f"Queue sizes: WS={stats.get('queue_sizes', {}).get('ws_queue', 0)}, "
                    f"REST={stats.get('queue_sizes', {}).get('rest_queue', 0)}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def run(self):
        """Run the trading system"""
        if not self.initialization_success:
            self.logger.error("Cannot start system - initialization failed")
            return
        
        # 시스템 초기화
        if not await self.initialize():
            self.logger.error("Failed to initialize system")
            return
        
        self.logger.info("Starting all system components...")
        
        # 모든 컴포넌트의 태스크 생성
        tasks = []
        
        try:
            # 핵심 컴포넌트
            tasks.extend([
                asyncio.create_task(self.rest_client.run(), name="REST_CLIENT"),
                asyncio.create_task(self.ws_client.run(), name="WEBSOCKET_CLIENT"),
                asyncio.create_task(self.trading_engine.run(), name="TRADING_ENGINE"),
            ])
            
            # 모니터링 컴포넌트
            tasks.extend([
                asyncio.create_task(self.telegram_bot.run(), name="TELEGRAM_BOT"),
                asyncio.create_task(self.status_display.run(), name="STATUS_DISPLAY"),
            ])
            
            # 시스템 관리 태스크
            tasks.extend([
                asyncio.create_task(self.update_symbols_periodically(), name="SYMBOL_UPDATER"),
                asyncio.create_task(self.system_health_monitor(), name="HEALTH_MONITOR"),
            ])
            
            self.components_started = True
            self.logger.info(f"Trading system is running with {len(tasks)} components")
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"System error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            if not self.shutdown_in_progress:
                self.logger.info("System stopped unexpectedly, initiating shutdown")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the trading system"""
        if self.shutdown_in_progress:
            return
        
        self.shutdown_in_progress = True
        self.logger.info("Shutting down trading system...")
        
        try:
            # 1. 데이터 매니저 중지 (모든 컴포넌트에 중지 신호)
            if self.data_manager:
                self.data_manager.stop()
                self.logger.info("✓ DataManager stopped")
            
            # 2. 지표 프로세서 중지
            if self.indicator_manager:
                self.indicator_manager.stop_all()
                self.logger.info("✓ Indicator processors stopped")
            
            # 3. WebSocket 연결 중지
            if self.ws_client:
                self.ws_client.stop()
                self.logger.info("✓ WebSocket client stopped")
            
            # 4. 거래 엔진 중지
            if self.trading_engine:
                self.trading_engine.stop()
                self.logger.info("✓ Trading engine stopped")
            
            # 5. 텔레그램 봇 중지
            if self.telegram_bot:
                self.telegram_bot.stop()
                self.logger.info("✓ Telegram bot stopped")
            
            # 6. 데이터 매니저 정리
            if self.data_manager:
                self.data_manager.cleanup()
                self.logger.info("✓ DataManager cleanup completed")
            
            self.logger.info("Trading system shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.logger.info("Shutdown process finished")


async def main():
    """Main entry point with enhanced error handling"""
    logger = get_logger("main")
    system = None
    
    try:
        logger.info("=" * 60)
        logger.info("BINANCE TRADING BOT - ENHANCED VERSION")
        logger.info(f"Timeframe: 15m")
        logger.info(f"Max Symbols: {config.symbol_count}")
        logger.info(f"Leverage: {config.leverage}x")
        logger.info(f"Take Profit: {config.take_profit_percent}%")
        logger.info(f"Position Size: {config.position_size_percent}%")
        logger.info("=" * 60)
        
        # 환경 검증
        if not config.binance_api_key or not config.binance_api_secret:
            raise ValueError("Binance API credentials not configured")
        
        if config.symbol_count < 1 or config.symbol_count > 20:
            raise ValueError(f"Invalid symbol count: {config.symbol_count} (must be 1-20)")
        
        logger.info("Environment validation passed")
        
        # 거래 시스템 생성 및 실행
        system = TradingSystem()
        
        # 시스템 시작 전 상태 확인
        if not system.initialization_success:
            raise RuntimeError("Trading system initialization failed")
        
        logger.info("Starting enhanced trading system...")
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 시스템이 생성된 경우 안전하게 종료
        if system:
            try:
                system.shutdown()
            except Exception as shutdown_error:
                logger.error(f"Error during emergency shutdown: {shutdown_error}")
    
    finally:
        logger.info("Program terminated")
        
        # 최종 정리
        try:
            # 남은 프로세스 정리
            import multiprocessing as mp
            for process in mp.active_children():
                if process.is_alive():
                    logger.warning(f"Terminating orphaned process: {process.name}")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
        except Exception as e:
            logger.error(f"Error cleaning up processes: {e}")


if __name__ == "__main__":
    # 멀티프로세싱 설정 (Windows 호환성)
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # 비동기 메인 함수 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program failed to start: {e}")
        import traceback
        traceback.print_exc()