"""
Configuration management module
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Central configuration class"""
    
    # Binance API
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    binance_testnet: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # Telegram
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Trading
    leverage: int = int(os.getenv('LEVERAGE', 10))
    symbol_count: int = int(os.getenv('SYMBOL_COUNT', 5))
    position_size_percent: float = float(os.getenv('POSITION_SIZE_PERCENT', 2.0))
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_to_file: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    
    # Indicator Parameters
    adx_length: int = int(os.getenv('ADX_LENGTH', 24))
    adx_smoothing: int = int(os.getenv('ADX_SMOOTHING', 24))
    vi_length: int = int(os.getenv('VI_LENGTH', 48))
    mfi_length: int = int(os.getenv('MFI_LENGTH', 24))
    rsi_length: int = int(os.getenv('RSI_LENGTH', 24))
    rsi_smoothing: int = int(os.getenv('RSI_SMOOTHING', 24))
    cci_length: int = int(os.getenv('CCI_LENGTH', 24))
    cci_smoothing: int = int(os.getenv('CCI_SMOOTHING', 24))
    psar_start: float = float(os.getenv('PSAR_START', 0.005))
    psar_increment: float = float(os.getenv('PSAR_INCREMENT', 0.005))
    psar_maximum: float = float(os.getenv('PSAR_MAXIMUM', 0.05))
    supertrend_atr_length: int = int(os.getenv('SUPERTREND_ATR_LENGTH', 10))
    supertrend_multiplier: float = float(os.getenv('SUPERTREND_MULTIPLIER', 3.0))
    hurst_window: int = int(os.getenv('HURST_WINDOW', 24))
    hurst_rs_lag: int = int(os.getenv('HURST_RS_LAG', 24))
    hurst_smoothing: int = int(os.getenv('HURST_SMOOTHING', 24))
    atr_length: int = int(os.getenv('ATR_LENGTH', 24))
    vwma_short: int = int(os.getenv('VWMA_SHORT', 6))
    vwma_mid: int = int(os.getenv('VWMA_MID', 12))
    vwma_long: int = int(os.getenv('VWMA_LONG', 24))
    
    # Oscillator Normalization
    weight_rsi: float = float(os.getenv('WEIGHT_RSI', 0.4))
    weight_cci: float = float(os.getenv('WEIGHT_CCI', 0.3))
    weight_mfi: float = float(os.getenv('WEIGHT_MFI', 0.3))
    oscillator_long_threshold: float = float(os.getenv('OSCILLATOR_LONG_THRESHOLD', -1.0))
    oscillator_short_threshold: float = float(os.getenv('OSCILLATOR_SHORT_THRESHOLD', 1.0))
    
    # Trend Detection
    trend_detection_candles: int = int(os.getenv('TREND_DETECTION_CANDLES', 8))
    
    # Risk Management
    take_profit_percent: float = float(os.getenv('TAKE_PROFIT_PERCENT', 5.0))
    stop_trailing_atr_multiplier: float = float(os.getenv('STOP_TRAILING_ATR_MULTIPLIER', 2.0))
    stop_trailing_min_percent: float = float(os.getenv('STOP_TRAILING_MIN_PERCENT', 1.0))
    stop_trailing_max_percent: float = float(os.getenv('STOP_TRAILING_MAX_PERCENT', 10.0))
    
    # Cooldown
    position_cooldown_seconds: int = int(os.getenv('POSITION_COOLDOWN_SECONDS', 300))
    cooldown_multiplier: float = float(os.getenv('COOLDOWN_MULTIPLIER', 1.5))
    
    # System
    max_workers: int = int(os.getenv('MAX_WORKERS', 20))
    rest_api_interval: int = int(os.getenv('REST_API_INTERVAL', 3))
    websocket_reconnect_delay: int = int(os.getenv('WEBSOCKET_RECONNECT_DELAY', 5))
    candle_confirmation_delay: int = int(os.getenv('CANDLE_CONFIRMATION_DELAY', 10))
    status_display_interval: int = int(os.getenv('STATUS_DISPLAY_INTERVAL', 30))
    
    @property
    def binance_base_url(self) -> str:
        """Get Binance base URL based on testnet setting"""
        if self.binance_testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"
    
    @property
    def binance_ws_url(self) -> str:
        """Get Binance WebSocket URL based on testnet setting"""
        if self.binance_testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

# Global config instance
config = Config()