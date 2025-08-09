"""
리스크 관리 모듈 (KISS 원칙 적용)
Stop Trailing 및 Take Profit 관리
"""
import asyncio
from typing import Dict, Optional
from src.utils.logger import get_logger
from src.utils.config import config

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self, position_manager):
        self.logger = get_logger("risk_manager")
        self.position_manager = position_manager
        self.stop_losses: Dict[str, float] = {}  # 심볼별 스탑로스 가격
        self.take_profits: Dict[str, float] = {}  # 심볼별 테이크프로핏 가격
        
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        스탑로스 가격 계산 (ATR 기반)
        
        Args:
            entry_price: 진입 가격
            side: 포지션 방향
            atr: ATR 값
            
        Returns:
            float: 스탑로스 가격
        """
        # ATR 기반 스탑로스 거리
        stop_distance = atr * config.stop_trailing_multiplier
        
        # 최소/최대 제한 적용
        min_distance = entry_price * (config.stop_trailing_min_percent / 100)
        max_distance = entry_price * (config.stop_trailing_max_percent / 100)
        
        stop_distance = max(min_distance, min(stop_distance, max_distance))
        
        # 포지션 방향에 따른 스탑로스 가격
        if side == 'long':
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        return stop_price
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        테이크프로핏 가격 계산 (고정 %)
        
        Args:
            entry_price: 진입 가격
            side: 포지션 방향
            
        Returns:
            float: 테이크프로핏 가격
        """
        tp_distance = entry_price * (config.take_profit_percent / 100)
        
        if side == 'long':
            tp_price = entry_price + tp_distance
        else:
            tp_price = entry_price - tp_distance
        
        return tp_price
    
    def update_trailing_stop(self, symbol: str, current_price: float, 
                            position_side: str, atr: float) -> Optional[float]:
        """
        트레일링 스탑 업데이트
        
        Args:
            symbol: 심볼명
            current_price: 현재 가격
            position_side: 포지션 방향
            atr: 현재 ATR
            
        Returns:
            Optional[float]: 업데이트된 스탑로스 가격
        """
        current_stop = self.stop_losses.get(symbol)
        
        # 새 스탑로스 계산
        new_stop = self.calculate_stop_loss(current_price, position_side, atr)
        
        # 트레일링 스탑 업데이트 (유리한 방향으로만)
        if position_side == 'long':
            # 롱 포지션: 스탑로스를 위로만 이동
            if current_stop is None or new_stop > current_stop:
                self.stop_losses[symbol] = new_stop
                self.logger.debug(f"{symbol}: 트레일링 스탑 업데이트 (롱): {new_stop:.4f}")
                return new_stop
        else:
            # 숏 포지션: 스탑로스를 아래로만 이동
            if current_stop is None or new_stop < current_stop:
                self.stop_losses[symbol] = new_stop
                self.logger.debug(f"{symbol}: 트레일링 스탑 업데이트 (숏): {new_stop:.4f}")
                return new_stop
        
        return current_stop
    
    async def check_risk_limits(self, symbol: str, current_price: float) -> Optional[str]:
        """
        리스크 한도 체크
        
        Args:
            symbol: 심볼명
            current_price: 현재 가격
            
        Returns:
            Optional[str]: 'stop_loss', 'take_profit', None
        """
        position = self.position_manager.get_position(symbol)
        if not position:
            return None
        
        position_side = position['side']
        
        # 스탑로스 체크
        stop_price = self.stop_losses.get(symbol)
        if stop_price:
            if (position_side == 'long' and current_price <= stop_price) or \
               (position_side == 'short' and current_price >= stop_price):
                self.logger.warning(f"{symbol}: 스탑로스 도달 (가격: {current_price:.4f}, 스탑: {stop_price:.4f})")
                return 'stop_loss'
        
        # 테이크프로핏 체크
        tp_price = self.take_profits.get(symbol)
        if tp_price:
            if (position_side == 'long' and current_price >= tp_price) or \
               (position_side == 'short' and current_price <= tp_price):
                self.logger.info(f"{symbol}: 테이크프로핏 도달 (가격: {current_price:.4f}, TP: {tp_price:.4f})")
                return 'take_profit'
        
        return None
    
    def set_initial_limits(self, symbol: str, entry_price: float, 
                          position_side: str, atr: float) -> None:
        """
        초기 리스크 한도 설정
        
        Args:
            symbol: 심볼명
            entry_price: 진입 가격
            position_side: 포지션 방향
            atr: ATR 값
        """
        # 초기 스탑로스 설정
        self.stop_losses[symbol] = self.calculate_stop_loss(entry_price, position_side, atr)
        
        # 테이크프로핏 설정
        self.take_profits[symbol] = self.calculate_take_profit(entry_price, position_side)
        
        self.logger.info(
            f"{symbol}: 리스크 한도 설정 - "
            f"진입: {entry_price:.4f}, "
            f"SL: {self.stop_losses[symbol]:.4f}, "
            f"TP: {self.take_profits[symbol]:.4f}"
        )
    
    def clear_limits(self, symbol: str) -> None:
        """
        리스크 한도 제거
        
        Args:
            symbol: 심볼명
        """
        self.stop_losses.pop(symbol, None)
        self.take_profits.pop(symbol, None)
        self.logger.debug(f"{symbol}: 리스크 한도 제거")
    
    def get_risk_info(self, symbol: str) -> Dict:
        """
        리스크 정보 반환
        
        Args:
            symbol: 심볼명
            
        Returns:
            Dict: 리스크 정보
        """
        return {
            'stop_loss': self.stop_losses.get(symbol),
            'take_profit': self.take_profits.get(symbol)
        }