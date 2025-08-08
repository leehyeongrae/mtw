"""
Telegram bot for monitoring and control
"""
import asyncio
from typing import Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager
from src.exchange.order_executor import OrderExecutor

class TelegramBot:
    """Telegram bot for trading system control"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("telegram_bot")
        self.data_manager = data_manager
        self.order_executor = OrderExecutor()
        self.bot = None
        self.application = None
        
        if config.telegram_bot_token and config.telegram_chat_id:
            self._init_bot()
    
    def _init_bot(self):
        """Initialize Telegram bot"""
        try:
            self.application = Application.builder().token(config.telegram_bot_token).build()
            self.bot = self.application.bot
            
            # Add command handlers
            self.application.add_handler(CommandHandler("stop", self.cmd_stop))
            self.application.add_handler(CommandHandler("balance", self.cmd_balance))
            self.application.add_handler(CommandHandler("status", self.cmd_status))
            self.application.add_handler(CommandHandler("help", self.cmd_help))
            
            self.logger.info("Telegram bot initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        try:
            await update.message.reply_text("⚠️ Stopping trading system...")
            self.data_manager.stop()
            await update.message.reply_text("✅ Trading system stopped")
        except Exception as e:
            self.logger.error(f"Error in stop command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            balance = self.order_executor.get_account_balance()
            positions = self.order_executor.get_all_positions()
            
            total_pnl = sum(p.get('pnl', 0) for p in positions.values())
            
            message = f"💰 **Account Balance**\n"
            message += f"├ Available: ${balance:.2f}\n"
            message += f"├ Positions: {len(positions)}\n"
            message += f"└ Unrealized PnL: ${total_pnl:.2f}"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in balance command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            symbols = self.data_manager.get_symbol_list()
            
            if not symbols:
                await update.message.reply_text("No symbols being tracked")
                return
            
            message = "📊 **System Status**\n\n"
            
            for symbol in symbols[:10]:  # Limit to 10 symbols for readability
                status = self.data_manager.get_status(symbol)
                
                # Get current price
                _, current = self.data_manager.get_candles(symbol)
                price = current.get('c', 0) if current else 0
                
                # Get indicators
                indicators = self.data_manager.get_indicators(symbol)
                
                message += f"**{symbol}**\n"
                message += f"├ Price: ${price:.4f}\n"
                
                if indicators:
                    # Market type
                    is_trending = indicators.get('is_trending', 0)
                    market_type = "📈 Trending" if is_trending else "📊 Ranging"
                    message += f"├ Market: {market_type}\n"
                    
                    # Key indicators
                    if len(indicators.get('adx', [])) > 0:
                        adx = indicators['adx'][-1]
                        message += f"├ ADX: {adx:.1f}\n"
                    
                    if len(indicators.get('rsi', [])) > 0:
                        rsi = indicators['rsi'][-1]
                        message += f"├ RSI: {rsi:.1f}\n"
                    
                    if len(indicators.get('hurst_smoothed', [])) > 0:
                        hurst = indicators['hurst_smoothed'][-1]
                        message += f"├ Hurst: {hurst:.3f}\n"
                
                # Position info
                position = self.data_manager.get_position(symbol)
                if position:
                    side = position.get('side', 'unknown')
                    pnl = position.get('pnl_percent', 0)
                    emoji = "🟢" if pnl > 0 else "🔴"
                    message += f"└ Position: {side.upper()} {emoji} {pnl:.2f}%\n"
                else:
                    message += f"└ Position: None\n"
                
                message += "\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🤖 **Trading Bot Commands**

/stop - Stop the trading system
/balance - Show account balance
/status - Show symbol status
/help - Show this help message

📊 The bot monitors and trades the top symbols by volume.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def send_notification(self, message: str):
        """
        Send notification to Telegram
        
        Args:
            message: Message to send
        """
        if self.bot and config.telegram_chat_id:
            try:
                await self.bot.send_message(
                    chat_id=config.telegram_chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                self.logger.error(f"Failed to send Telegram notification: {e}")
    
    async def run(self):
        """Run the Telegram bot"""
        if self.application:
            self.logger.info("Starting Telegram bot")
            
            # Send startup notification
            await self.send_notification("🚀 Trading bot started successfully!")
            
            # Start polling
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            # Keep running
            while self.data_manager.is_running():
                await asyncio.sleep(1)
            
            # Cleanup
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            
            self.logger.info("Telegram bot stopped")
        else:
            self.logger.warning("Telegram bot not configured")
    
    def stop(self):
        """Stop the Telegram bot"""
        if self.application:
            asyncio.create_task(self.send_notification("⚠️ Trading bot shutting down..."))