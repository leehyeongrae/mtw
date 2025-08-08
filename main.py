"""
Main entry point for the trading system
"""
import asyncio
import signal
import sys
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
    """Main trading system coordinator"""
    
    def __init__(self):
        self.logger = get_logger("main")
        self.logger.info("Initializing trading system...")
        
        # Core components
        self.data_manager = DataManager()
        self.symbol_manager = SymbolManager()
        
        # Exchange connections
        self.rest_client = BinanceREST(self.data_manager)
        self.ws_client = BinanceWebSocket(self.data_manager)
        
        # Trading components
        self.indicator_manager = IndicatorProcessorManager(self.data_manager)
        self.trading_engine = TradingEngine(self.data_manager)
        
        # Monitoring
        self.telegram_bot = TelegramBot(self.data_manager)
        self.status_display = StatusDisplay(self.data_manager)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    async def initialize(self):
        """Initialize the trading system"""
        try:
            # Get initial symbol list
            self.logger.info("Fetching top symbols by volume...")
            symbols = self.symbol_manager.update_symbol_list()
            
            if not symbols:
                self.logger.error("No symbols to trade")
                return False
            
            self.logger.info(f"Trading symbols: {symbols}")
            
            # Update data manager with symbols
            self.data_manager.update_symbol_list(symbols)
            
            # Start indicator processors for each symbol
            self.indicator_manager.update_symbols(symbols)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def update_symbols_periodically(self):
        """Update symbol list periodically (daily)"""
        while self.data_manager.is_running():
            try:
                # Wait 24 hours
                await asyncio.sleep(86400)
                
                # Update symbol list
                self.logger.info("Updating symbol list...")
                symbols = self.symbol_manager.update_symbol_list()
                
                if symbols:
                    self.data_manager.update_symbol_list(symbols)
                    self.indicator_manager.update_symbols(symbols)
                    
            except Exception as e:
                self.logger.error(f"Error updating symbols: {e}")
    
    async def run(self):
        """Run the trading system"""
        # Initialize
        if not await self.initialize():
            self.logger.error("Failed to initialize system")
            return
        
        self.logger.info("Starting all components...")
        
        # Create tasks for all components
        tasks = [
            asyncio.create_task(self.rest_client.run()),
            asyncio.create_task(self.ws_client.run()),
            asyncio.create_task(self.trading_engine.run()),
            asyncio.create_task(self.telegram_bot.run()),
            asyncio.create_task(self.status_display.run()),
            asyncio.create_task(self.update_symbols_periodically())
        ]
        
        self.logger.info("Trading system is running")
        
        try:
            # Wait for all tasks
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the trading system"""
        self.logger.info("Shutting down trading system...")
        
        # Stop data manager (signals all components)
        self.data_manager.stop()
        
        # Stop indicator processors
        self.indicator_manager.stop_all()
        
        # Stop WebSocket
        self.ws_client.stop()
        
        # Stop trading engine
        self.trading_engine.stop()
        
        # Stop Telegram bot
        self.telegram_bot.stop()
        
        # Cleanup data manager
        self.data_manager.cleanup()
        
        self.logger.info("Trading system shutdown complete")


async def main():
    """Main entry point"""
    logger = get_logger("main")
    
    try:
        logger.info("="*50)
        logger.info("BINANCE TRADING BOT")
        logger.info(f"Timeframe: 15m")
        logger.info(f"Max Symbols: {config.symbol_count}")
        logger.info(f"Leverage: {config.leverage}x")
        logger.info("="*50)
        
        # Create and run trading system
        system = TradingSystem()
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Program terminated")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())