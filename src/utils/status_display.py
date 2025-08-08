"""
Status display module for console output
"""
import asyncio
from datetime import datetime
import pandas as pd
from typing import Dict, List
from src.utils.logger import get_logger
from src.utils.config import config
from src.core.data_manager import DataManager

class StatusDisplay:
    """Displays system status periodically"""
    
    def __init__(self, data_manager: DataManager):
        self.logger = get_logger("status_display")
        self.data_manager = data_manager
        self.display_interval = config.status_display_interval
        
    def format_candle_row(self, candle: Dict, is_current: bool = False) -> str:
        """Format a single candle for display"""
        if is_current:
            timestamp = datetime.fromtimestamp(candle.get('T', 0) / 1000).strftime('%H:%M')
            status = "CURRENT"
        else:
            timestamp = datetime.fromtimestamp(candle.get('open_time', 0) / 1000).strftime('%H:%M')
            status = "CLOSED"
        
        o = candle.get('o', candle.get('open', 0))
        h = candle.get('h', candle.get('high', 0))
        l = candle.get('l', candle.get('low', 0))
        c = candle.get('c', candle.get('close', 0))
        v = candle.get('v', candle.get('volume', 0))
        
        return f"  [{status:7}] {timestamp} | O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} C:{c:8.2f} V:{v:10.0f}"
    
    def format_indicators(self, indicators: Dict) -> List[str]:
        """Format indicators for display"""
        lines = []
        
        if not indicators:
            return ["  No indicators available"]
        
        # Market type
        is_trending = indicators.get('is_trending', 0)
        market_type = "TRENDING" if is_trending else "RANGING"
        lines.append(f"  Market Type: {market_type}")
        
        # Extract latest values
        def get_latest(arr):
            return arr[-1] if len(arr) > 0 else 0
        
        # Trend indicators
        adx = get_latest(indicators.get('adx', []))
        hurst = get_latest(indicators.get('hurst_smoothed', []))
        lines.append(f"  Trend: ADX={adx:.1f}, Hurst={hurst:.3f}")
        
        # Oscillators
        rsi = get_latest(indicators.get('rsi', []))
        cci = get_latest(indicators.get('cci', []))
        mfi = get_latest(indicators.get('mfi', []))
        lines.append(f"  Oscillators: RSI={rsi:.1f}, CCI={cci:.1f}, MFI={mfi:.1f}")
        
        # Directional indicators
        psar_trend = get_latest(indicators.get('psar_trend', []))
        supertrend_dir = get_latest(indicators.get('trend_direction', []))
        vi_plus = get_latest(indicators.get('vi_plus', []))
        vi_minus = get_latest(indicators.get('vi_minus', []))
        
        psar_str = "UP" if psar_trend == 1 else "DOWN"
        super_str = "UP" if supertrend_dir == 1 else "DOWN"
        lines.append(f"  Direction: PSAR={psar_str}, SuperTrend={super_str}, VI+={vi_plus:.3f}, VI-={vi_minus:.3f}")
        
        # ATR
        atr = get_latest(indicators.get('atr', []))
        lines.append(f"  Volatility: ATR={atr:.4f}")
        
        return lines
    
    def display_status(self):
        """Display current status of all symbols"""
        symbols = self.data_manager.get_symbol_list()
        
        if not symbols:
            print("\n" + "="*80)
            print("No symbols being tracked")
            print("="*80)
            return
        
        print("\n" + "="*80)
        print(f"TRADING SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        for symbol in symbols:
            print(f"\n{symbol}")
            print("-" * 40)
            
            # Get candle data
            historical_df, current_candle = self.data_manager.get_candles(symbol)
            
            if historical_df is not None and len(historical_df) >= 4:
                # Display last 4 closed candles
                print("  Recent Candles (15m):")
                for i in range(-4, 0):
                    candle = historical_df.iloc[i].to_dict()
                    print(self.format_candle_row(candle, False))
                
                # Display current candle
                if current_candle:
                    print(self.format_candle_row(current_candle, True))
            else:
                print("  No candle data available")
            
            # Display indicators
            print("\n  Indicators:")
            indicators = self.data_manager.get_indicators(symbol)
            for line in self.format_indicators(indicators):
                print(line)
            
            # Display position
            position = self.data_manager.get_position(symbol)
            if position:
                side = position.get('side', 'unknown')
                entry = position.get('entry_price', 0)
                current = position.get('current_price', 0)
                pnl = position.get('pnl_percent', 0)
                tp = position.get('take_profit', 0)
                sl = position.get('stop_loss', 0)
                
                print(f"\n  Position: {side.upper()}")
                print(f"    Entry: ${entry:.4f}, Current: ${current:.4f}")
                print(f"    PnL: {pnl:+.2f}%")
                print(f"    TP: ${tp:.4f}, SL: ${sl:.4f}" if sl else f"    TP: ${tp:.4f}, SL: Not set")
            else:
                print("\n  Position: None")
            
            # Display last signal
            signal = self.data_manager.get_signal(symbol)
            if signal:
                action = signal.get('action', 'None')
                reason = signal.get('reason', 'No reason')
                print(f"\n  Last Signal: {action}")
                print(f"    Reason: {reason}")
        
        print("\n" + "="*80)
    
    async def run(self):
        """Run the status display loop"""
        self.logger.info("Starting status display")
        
        while self.data_manager.is_running():
            try:
                self.display_status()
                await asyncio.sleep(self.display_interval)
                
            except Exception as e:
                self.logger.error(f"Error in status display: {e}")
                await asyncio.sleep(self.display_interval)
        
        self.logger.info("Status display stopped")