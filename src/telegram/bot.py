"""
telegram/bot.py 수정 - 폴링 모드 및 명령어 작동 문제 해결
"""
import asyncio
from typing import Optional, Dict, Any
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from src.utils.logger import get_logger
from src.utils.config import config

class TelegramBot:
    """텔레그램 봇 클래스"""
    
    def __init__(self, trading_system=None):
        self.logger = get_logger("telegram_bot")
        self.trading_system = trading_system
        self.bot: Optional[Bot] = None
        self.app: Optional[Application] = None
        self.polling_task = None
        
    async def initialize(self) -> None:
        """봇 초기화 - 폴링 모드로 변경"""
        if not config.telegram_bot_token:
            self.logger.warning("텔레그램 토큰 없음 - 봇 비활성화")
            return
        
        try:
            # 애플리케이션 생성
            self.app = Application.builder().token(config.telegram_bot_token).build()
            self.bot = self.app.bot
            
            # 핸들러 등록
            self.app.add_handler(CommandHandler("start", self.cmd_start))
            self.app.add_handler(CommandHandler("stop", self.cmd_stop))
            self.app.add_handler(CommandHandler("balance", self.cmd_balance))
            self.app.add_handler(CommandHandler("status", self.cmd_status))
            self.app.add_handler(CommandHandler("help", self.cmd_help))
            
            # 봇 초기화
            await self.app.initialize()
            await self.app.start()
            
            # 폴링 시작 (백그라운드 태스크로)
            self.polling_task = asyncio.create_task(self.app.updater.start_polling(drop_pending_updates=True))
            
            self.logger.info("텔레그램 봇 초기화 완료 (폴링 모드)")
            
            # 초기화 성공 메시지 전송
            await self.send_message("🤖 트레이딩 봇이 시작되었습니다!\n/help 명령어로 사용법을 확인하세요.")
            
        except Exception as e:
            self.logger.error(f"텔레그램 봇 초기화 실패: {e}")
    
    async def send_message(self, text: str, parse_mode: str = 'HTML') -> None:
        """
        메시지 전송
        
        Args:
            text: 메시지 텍스트
            parse_mode: 파싱 모드
        """
        if not self.bot or not config.telegram_chat_id:
            return
        
        try:
            await self.bot.send_message(
                chat_id=config.telegram_chat_id,
                text=text,
                parse_mode=parse_mode
            )
        except Exception as e:
            self.logger.error(f"메시지 전송 실패: {e}")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """시작 명령어"""
        await update.message.reply_text(
            "🤖 트레이딩 봇이 실행 중입니다.\n"
            "/help - 명령어 목록 보기"
        )
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """중지 명령어"""
        # 관리자 권한 체크 (chat_id 확인)
        if str(update.effective_chat.id) != str(config.telegram_chat_id):
            await update.message.reply_text("❌ 권한이 없습니다.")
            return
            
        if self.trading_system:
            await update.message.reply_text("⏹ 트레이딩 봇을 중지합니다...")
            # 비동기로 시스템 중지
            asyncio.create_task(self.trading_system.stop())
        else:
            await update.message.reply_text("❌ 트레이딩 시스템이 연결되지 않았습니다.")
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """잔고 확인 명령어"""
        if not self.trading_system:
            await update.message.reply_text("❌ 트레이딩 시스템이 연결되지 않았습니다.")
            return
        
        try:
            balance_info = await self.trading_system.get_account_info()
            
            if balance_info:
                text = "💰 <b>계정 정보</b>\n\n"
                text += f"잔고: ${balance_info.get('balance', 0):.2f}\n"
                text += f"미실현 PnL: ${balance_info.get('unrealized_pnl', 0):.2f}\n"
                text += f"마진 사용률: {balance_info.get('margin_ratio', 0):.2f}%\n"
                text += f"포지션 수: {balance_info.get('position_count', 0)}"
                
                await update.message.reply_text(text, parse_mode='HTML')
            else:
                await update.message.reply_text("❌ 계정 정보를 가져올 수 없습니다.")
                
        except Exception as e:
            self.logger.error(f"잔고 조회 오류: {e}")
            await update.message.reply_text("❌ 오류가 발생했습니다.")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """상태 확인 명령어 - 완성 버전"""
        if not self.trading_system:
            await update.message.reply_text("❌ 트레이딩 시스템이 연결되지 않았습니다.")
            return
        
        try:
            # 심볼 인자 확인
            args = context.args
            
            if not args:
                # 전체 상태 (간략히)
                status = await self.trading_system.get_all_status()
                
                if not status:
                    await update.message.reply_text("📊 활성 심볼이 없습니다.")
                    return
                
                text = "📊 <b>전체 상태</b>\n\n"
                for symbol, info in status.items():
                    if info:
                        text += f"<b>{symbol}</b>\n"
                        text += f"├ 가격: {info.get('price', 0):.4f}\n"
                        text += f"├ 시장: {info.get('market_type', 'N/A')}\n"
                        text += f"├ 포지션: {info.get('position', {}).get('side', 'None') if info.get('position') else 'None'}\n"
                        text += f"└ 추세: {info.get('trend', 'N/A')}\n\n"
                
            else:
                # 특정 심볼 상태 (상세)
                symbol = args[0].upper()
                info = await self.trading_system.get_symbol_status(symbol)
                
                if info:
                    text = f"📊 <b>{symbol} 상세 정보</b>\n\n"
                    
                    # 가격 정보
                    text += "<b>가격 정보</b>\n"
                    text += f"├ 현재가: {info.get('price', 0):.4f}\n"
                    if 'change_24h' in info:
                        text += f"├ 24h 변동: {info.get('change_24h', 0):.2f}%\n"
                    text += f"└ 거래량: {info.get('volume', 0):,.0f}\n\n"
                    
                    # 시장 상태
                    text += "<b>시장 분석</b>\n"
                    text += f"├ 시장 유형: {info.get('market_type', 'N/A')}\n"
                    text += f"├ ADX: {info.get('adx', 0):.2f}\n"
                    text += f"├ Hurst: {info.get('hurst', 0):.3f}\n"
                    text += f"└ 추세: {info.get('trend', 'N/A')}\n\n"
                    
                    # 지표
                    text += "<b>기술 지표</b>\n"
                    text += f"├ RSI: {info.get('rsi', 0):.1f}\n"
                    text += f"├ CCI: {info.get('cci', 0):.1f}\n"
                    text += f"├ MFI: {info.get('mfi', 0):.1f}\n"
                    text += f"├ VI+: {info.get('vi_plus', 0):.3f}\n"
                    text += f"├ VI-: {info.get('vi_minus', 0):.3f}\n"
                    if 'oscillator_score' in info:
                        text += f"└ Score: {info.get('oscillator_score', 0):.2f}\n\n"
                    else:
                        text += "\n"
                    
                    # 포지션
                    if info.get('position'):
                        text += "<b>포지션 정보</b>\n"
                        pos = info['position']
                        text += f"├ 방향: {'롱' if pos.get('side') == 'long' else '숏'}\n"
                        text += f"├ 진입가: {pos.get('entry_price', 0):.4f}\n"
                        text += f"├ PnL: ${pos.get('pnl', 0):.2f}\n"
                        if pos.get('stop_loss'):
                            text += f"├ SL: {pos.get('stop_loss', 0):.4f}\n"
                        if pos.get('take_profit'):
                            text += f"└ TP: {pos.get('take_profit', 0):.4f}\n"
                    else:
                        text += "<b>포지션</b>: 없음\n"
                else:
                    text = f"❌ {symbol} 정보를 찾을 수 없습니다."
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(f"상태 조회 오류: {e}")
            await update.message.reply_text("❌ 오류가 발생했습니다.")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """도움말 명령어"""
        help_text = """
🤖 <b>트레이딩 봇 명령어</b>

/stop - 봇 중지
/balance - 계정 잔고 확인
/status - 전체 심볼 상태
/status [SYMBOL] - 특정 심볼 상세 정보
/help - 이 도움말 보기

<b>예시:</b>
/status BTCUSDT
        """
        await update.message.reply_text(help_text, parse_mode='HTML')
    
    async def notify_trade(self, symbol: str, action: str, details: Dict) -> None:
        """
        거래 알림 전송
        
        Args:
            symbol: 심볼명
            action: 거래 액션
            details: 거래 상세 정보
        """
        emoji = {
            'long_entry': '🟢',
            'short_entry': '🔴',
            'long_exit': '⚪',
            'short_exit': '⚪',
            'stop_loss': '🛑',
            'take_profit': '💰'
        }.get(action, '📊')
        
        text = f"{emoji} <b>{symbol} - {action.upper()}</b>\n\n"
        
        if 'entry' in action:
            text += f"진입가: {details.get('price', 0):.4f}\n"
            text += f"수량: {details.get('quantity', 0):.4f}\n"
            text += f"시장: {details.get('market_type', 'N/A')}\n"
            text += f"이유: {details.get('reason', 'N/A')}"
        elif 'exit' in action or action in ['stop_loss', 'take_profit']:
            text += f"청산가: {details.get('price', 0):.4f}\n"
            text += f"PnL: ${details.get('pnl', 0):.2f}\n"
            text += f"수익률: {details.get('pnl_percent', 0):.2f}%\n"
            text += f"이유: {details.get('reason', action)}"
        
        await self.send_message(text)
    
    async def notify_error(self, error_msg: str) -> None:
        """
        에러 알림 전송
        
        Args:
            error_msg: 에러 메시지
        """
        text = f"⚠️ <b>시스템 오류</b>\n\n{error_msg}"
        await self.send_message(text)
    
    async def stop(self) -> None:
        """봇 종료 - 수정"""
        if self.app:
            # 폴링 중지
            if self.polling_task:
                await self.app.updater.stop()
                self.polling_task.cancel()
            
            await self.app.stop()
            await self.app.shutdown()
            self.logger.info("텔레그램 봇 종료")