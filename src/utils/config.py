"""
설정 관리 모듈 (KISS 원칙 적용)
.env 파일에서 설정을 로드하고 중앙 관리
"""
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class Config:
    """중앙 설정 관리 클래스"""
    
    def __init__(self):
        # Binance API
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.binance_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        
        # Telegram API
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Trading Settings
        self.leverage = int(os.getenv('LEVERAGE', '5'))
        self.symbol_count = int(os.getenv('SYMBOL_COUNT', '5'))
        self.take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', '5.0'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # Indicator Parameters (DRY 원칙 적용)
        self.adx_length = int(os.getenv('ADX_LENGTH', '24'))
        self.adx_smoothing = int(os.getenv('ADX_SMOOTHING', '24'))
        self.vi_length = int(os.getenv('VI_LENGTH', '48'))
        self.mfi_length = int(os.getenv('MFI_LENGTH', '24'))
        self.rsi_length = int(os.getenv('RSI_LENGTH', '24'))
        self.rsi_smoothing = int(os.getenv('RSI_SMOOTHING', '24'))
        self.cci_length = int(os.getenv('CCI_LENGTH', '24'))
        self.cci_smoothing = int(os.getenv('CCI_SMOOTHING', '24'))
        self.psar_start = float(os.getenv('PSAR_START', '0.005'))
        self.psar_increment = float(os.getenv('PSAR_INCREMENT', '0.005'))
        self.psar_maximum = float(os.getenv('PSAR_MAXIMUM', '0.05'))
        self.supertrend_length = int(os.getenv('SUPERTREND_LENGTH', '24'))
        self.supertrend_multiplier = float(os.getenv('SUPERTREND_MULTIPLIER', '5.0'))
        self.hurst_window = int(os.getenv('HURST_WINDOW', '24'))
        self.hurst_rs_lag = int(os.getenv('HURST_RS_LAG', '24'))
        self.hurst_smoothing = int(os.getenv('HURST_SMOOTHING', '24'))
        self.atr_length = int(os.getenv('ATR_LENGTH', '24'))
        
        # Oscillator Normalization
        self.weight_rsi = float(os.getenv('WEIGHT_RSI', '0.4'))
        self.weight_cci = float(os.getenv('WEIGHT_CCI', '0.3'))
        self.weight_mfi = float(os.getenv('WEIGHT_MFI', '0.3'))
        self.oscillator_long_threshold = float(os.getenv('OSCILLATOR_LONG_THRESHOLD', '-1.0'))
        self.oscillator_short_threshold = float(os.getenv('OSCILLATOR_SHORT_THRESHOLD', '1.0'))
        
        # Market Detection
        self.trend_detection_candles = int(os.getenv('TREND_DETECTION_CANDLES', '8'))
        
        # Stop Loss / Trailing
        self.stop_trailing_multiplier = float(os.getenv('STOP_TRAILING_MULTIPLIER', '2.0'))
        self.stop_trailing_min_percent = float(os.getenv('STOP_TRAILING_MIN_PERCENT', '1.0'))
        self.stop_trailing_max_percent = float(os.getenv('STOP_TRAILING_MAX_PERCENT', '10.0'))
        
        # Cooldown Settings
        self.cooldown_seconds = int(os.getenv('COOLDOWN_SECONDS', '300'))
        self.cooldown_multiplier = int(os.getenv('COOLDOWN_MULTIPLIER', '12'))
        
        # Data Management
        self.candle_limit = int(os.getenv('CANDLE_LIMIT', '500'))
        self.candle_check_count = int(os.getenv('CANDLE_CHECK_COUNT', '50'))
        self.rest_api_delay = int(os.getenv('REST_API_DELAY', '10'))
        self.rest_api_interval = int(os.getenv('REST_API_INTERVAL', '3'))
        
        # System
        self.timeframe = os.getenv('TIMEFRAME', '15m')
        self.base_currency = os.getenv('BASE_CURRENCY', 'USDT')
        self.min_symbol_age_days = int(os.getenv('MIN_SYMBOL_AGE_DAYS', '7'))
        
    def get_binance_base_url(self) -> str:
        """바이낸스 API 베이스 URL 반환"""
        if self.binance_testnet:
            return 'https://testnet.binancefuture.com'
        return 'https://fapi.binance.com'
    
    def get_binance_ws_url(self) -> str:
        """바이낸스 웹소켓 URL 반환"""
        if self.binance_testnet:
            return 'wss://stream.binancefuture.com'
        return 'wss://fstream.binance.com'
    
    def validate(self) -> bool:
        """설정 유효성 검증"""
        errors = []
        
        if not self.binance_api_key or not self.binance_api_secret:
            errors.append("Binance API credentials missing")
        
        if self.weight_rsi + self.weight_cci + self.weight_mfi != 1.0:
            errors.append(f"Oscillator weights must sum to 1.0, current: {self.weight_rsi + self.weight_cci + self.weight_mfi}")
        
        if self.symbol_count > 20:
            errors.append("Symbol count cannot exceed 20")
        
        if errors:
            for error in errors:
                print(f"Config Error: {error}")
            return False
        
        return True

# 싱글톤 인스턴스
config = Config()