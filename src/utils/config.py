"""
Configuration management module - 개선된 버전
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Central configuration class with enhanced validation"""
    
    # Binance API
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    binance_testnet: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # Telegram
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Trading Parameters
    leverage: int = max(1, min(int(os.getenv('LEVERAGE', 10)), 125))  # 1-125 범위
    symbol_count: int = max(1, min(int(os.getenv('SYMBOL_COUNT', 5)), 20))  # 1-20 범위
    position_size_percent: float = max(0.1, min(float(os.getenv('POSITION_SIZE_PERCENT', 2.0)), 10.0))  # 0.1-10% 범위
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_to_file: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    
    # Indicator Parameters (검증된 기본값)
    adx_length: int = max(5, min(int(os.getenv('ADX_LENGTH', 24)), 100))
    adx_smoothing: int = max(5, min(int(os.getenv('ADX_SMOOTHING', 24)), 100))
    vi_length: int = max(5, min(int(os.getenv('VI_LENGTH', 48)), 200))
    mfi_length: int = max(5, min(int(os.getenv('MFI_LENGTH', 24)), 100))
    rsi_length: int = max(5, min(int(os.getenv('RSI_LENGTH', 24)), 100))
    rsi_smoothing: int = max(1, min(int(os.getenv('RSI_SMOOTHING', 24)), 50))
    cci_length: int = max(5, min(int(os.getenv('CCI_LENGTH', 24)), 100))
    cci_smoothing: int = max(1, min(int(os.getenv('CCI_SMOOTHING', 24)), 50))
    
    # PSAR Parameters
    psar_start: float = max(0.001, min(float(os.getenv('PSAR_START', 0.005)), 0.1))
    psar_increment: float = max(0.001, min(float(os.getenv('PSAR_INCREMENT', 0.005)), 0.1))
    psar_maximum: float = max(0.01, min(float(os.getenv('PSAR_MAXIMUM', 0.05)), 0.5))
    
    # SuperTrend Parameters
    supertrend_atr_length: int = max(3, min(int(os.getenv('SUPERTREND_ATR_LENGTH', 10)), 50))
    supertrend_multiplier: float = max(1.0, min(float(os.getenv('SUPERTREND_MULTIPLIER', 3.0)), 10.0))
    
    # Hurst Parameters (완화된 기본값)
    hurst_window: int = max(10, min(int(os.getenv('HURST_WINDOW', 24)), 100))
    hurst_rs_lag: int = max(5, min(int(os.getenv('HURST_RS_LAG', 24)), 50))
    hurst_smoothing: int = max(3, min(int(os.getenv('HURST_SMOOTHING', 24)), 50))
    
    # ATR Parameters
    atr_length: int = max(5, min(int(os.getenv('ATR_LENGTH', 24)), 100))
    
    # VWMA Parameters
    vwma_short: int = max(3, min(int(os.getenv('VWMA_SHORT', 6)), 20))
    vwma_mid: int = max(5, min(int(os.getenv('VWMA_MID', 12)), 50))
    vwma_long: int = max(10, min(int(os.getenv('VWMA_LONG', 24)), 100))
    
    # Oscillator Normalization (검증된 가중치)
    weight_rsi: float = max(0.1, min(float(os.getenv('WEIGHT_RSI', 0.4)), 1.0))
    weight_cci: float = max(0.1, min(float(os.getenv('WEIGHT_CCI', 0.3)), 1.0))
    weight_mfi: float = max(0.1, min(float(os.getenv('WEIGHT_MFI', 0.3)), 1.0))
    
    # 가중치 정규화 확인
    def __post_init__(self):
        # 가중치 합이 1이 되도록 정규화
        total_weight = self.weight_rsi + self.weight_cci + self.weight_mfi
        if total_weight != 1.0:
            self.weight_rsi = self.weight_rsi / total_weight
            self.weight_cci = self.weight_cci / total_weight
            self.weight_mfi = self.weight_mfi / total_weight
    
    oscillator_long_threshold: float = max(-3.0, min(float(os.getenv('OSCILLATOR_LONG_THRESHOLD', -1.0)), -0.1))
    oscillator_short_threshold: float = max(0.1, min(float(os.getenv('OSCILLATOR_SHORT_THRESHOLD', 1.0)), 3.0))
    
    # Trend Detection
    trend_detection_candles: int = max(3, min(int(os.getenv('TREND_DETECTION_CANDLES', 8)), 20))
    
    # Risk Management (검증된 범위)
    take_profit_percent: float = max(0.5, min(float(os.getenv('TAKE_PROFIT_PERCENT', 5.0)), 20.0))
    stop_trailing_atr_multiplier: float = max(0.5, min(float(os.getenv('STOP_TRAILING_ATR_MULTIPLIER', 2.0)), 10.0))
    stop_trailing_min_percent: float = max(0.1, min(float(os.getenv('STOP_TRAILING_MIN_PERCENT', 1.0)), 5.0))
    stop_trailing_max_percent: float = max(2.0, min(float(os.getenv('STOP_TRAILING_MAX_PERCENT', 10.0)), 25.0))
    
    # Cooldown (초 단위)
    position_cooldown_seconds: int = max(60, min(int(os.getenv('POSITION_COOLDOWN_SECONDS', 300)), 3600))
    cooldown_multiplier: float = max(1.0, min(float(os.getenv('COOLDOWN_MULTIPLIER', 1.5)), 5.0))
    
    # System Parameters (안전한 기본값)
    max_workers: int = max(1, min(int(os.getenv('MAX_WORKERS', 20)), 50))
    rest_api_interval: int = max(1, min(int(os.getenv('REST_API_INTERVAL', 3)), 10))  # 1-10초
    websocket_reconnect_delay: int = max(1, min(int(os.getenv('WEBSOCKET_RECONNECT_DELAY', 5)), 30))
    candle_confirmation_delay: int = max(5, min(int(os.getenv('CANDLE_CONFIRMATION_DELAY', 10)), 60))
    status_display_interval: int = max(10, min(int(os.getenv('STATUS_DISPLAY_INTERVAL', 30)), 300))
    
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
    
    def validate_configuration(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # API 키 검증
        if not self.binance_api_key:
            errors.append("BINANCE_API_KEY is required")
        
        if not self.binance_api_secret:
            errors.append("BINANCE_API_SECRET is required")
        
        # 논리적 검증
        if self.vwma_short >= self.vwma_mid:
            errors.append("VWMA_SHORT must be less than VWMA_MID")
        
        if self.vwma_mid >= self.vwma_long:
            errors.append("VWMA_MID must be less than VWMA_LONG")
        
        if self.stop_trailing_min_percent >= self.stop_trailing_max_percent:
            errors.append("STOP_TRAILING_MIN_PERCENT must be less than STOP_TRAILING_MAX_PERCENT")
        
        if self.oscillator_long_threshold >= self.oscillator_short_threshold:
            errors.append("OSCILLATOR_LONG_THRESHOLD must be less than OSCILLATOR_SHORT_THRESHOLD")
        
        # PSAR 검증
        if self.psar_start >= self.psar_maximum:
            errors.append("PSAR_START must be less than PSAR_MAXIMUM")
        
        if self.psar_increment >= self.psar_maximum:
            errors.append("PSAR_INCREMENT must be less than PSAR_MAXIMUM")
        
        # 로그 레벨 검증
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of: {valid_log_levels}")
        
        if errors:
            from src.utils.logger import get_logger
            logger = get_logger("config")
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_summary(self) -> dict:
        """
        Get configuration summary for logging
        
        Returns:
            dict: Configuration summary
        """
        return {
            'trading': {
                'leverage': self.leverage,
                'symbol_count': self.symbol_count,
                'position_size_percent': self.position_size_percent,
                'take_profit_percent': self.take_profit_percent,
            },
            'risk_management': {
                'stop_trailing_atr_multiplier': self.stop_trailing_atr_multiplier,
                'stop_trailing_min_percent': self.stop_trailing_min_percent,
                'stop_trailing_max_percent': self.stop_trailing_max_percent,
                'cooldown_seconds': self.position_cooldown_seconds,
            },
            'indicators': {
                'trend_detection_candles': self.trend_detection_candles,
                'hurst_window': self.hurst_window,
                'adx_length': self.adx_length,
                'rsi_length': self.rsi_length,
            },
            'system': {
                'testnet': self.binance_testnet,
                'rest_api_interval': self.rest_api_interval,
                'websocket_reconnect_delay': self.websocket_reconnect_delay,
                'log_level': self.log_level,
            }
        }

# 전역 설정 인스턴스
config = Config()

# 시작 시 설정 검증
if __name__ == "__main__":
    import json
    
    print("=== Configuration Validation ===")
    if config.validate_configuration():
        print("✓ Configuration is valid")
        print("\n=== Configuration Summary ===")
        summary = config.get_summary()
        print(json.dumps(summary, indent=2))
    else:
        print("✗ Configuration has errors")
        exit(1)