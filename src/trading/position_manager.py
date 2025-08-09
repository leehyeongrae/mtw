"""
포지션 관리 모듈 (YAGNI 원칙 적용)
"""
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from src.core import symbol_manager
from src.utils.logger import get_logger
from src.utils.config import config

class PositionManager:
    """포지션 관리 클래스"""
    
    def __init__(self, rest_manager):
        self.logger = get_logger("position_manager")
        self.rest_manager = rest_manager
        self.symbol_manager = symbol_manager  # 추가
        self.positions: Dict[str, Dict] = {}  # 심볼별 포지션
        self.cooldowns: Dict[str, datetime] = {}  # 심볼별 쿨다운
        self.lock = asyncio.Lock()
        
    async def update_positions(self) -> Dict[str, Dict]:
        """
        현재 포지션 업데이트
        
        Returns:
            Dict[str, Dict]: 포지션 정보
        """
        try:
            positions_data = await self.rest_manager.get_positions()
            if not positions_data:
                return self.positions
            
            async with self.lock:
                # 기존 포지션 클리어
                self.positions.clear()
                
                for pos in positions_data:
                    symbol = pos.get('symbol')
                    position_amt = float(pos.get('positionAmt', 0))
                    
                    if position_amt != 0:
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'position_amt': position_amt,
                            'side': 'long' if position_amt > 0 else 'short',
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'mark_price': float(pos.get('markPrice', 0)),
                            'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                            'margin': float(pos.get('isolatedWallet', 0) or pos.get('positionInitialMargin', 0)),
                            'leverage': int(pos.get('leverage', 1))
                        }
                
                self.logger.debug(f"포지션 업데이트: {len(self.positions)}개")
                return self.positions
                
        except Exception as e:
            self.logger.error(f"포지션 업데이트 오류: {e}")
            return self.positions
    
    def has_position(self, symbol: str) -> bool:
        """
        포지션 보유 여부 확인
        
        Args:
            symbol: 심볼명
            
        Returns:
            bool: 포지션 보유 여부
        """
        return symbol in self.positions
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        포지션 정보 반환
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[Dict]: 포지션 정보
        """
        return self.positions.get(symbol)
    
    def get_position_side(self, symbol: str) -> Optional[str]:
        """
        포지션 방향 반환
        
        Args:
            symbol: 심볼명
            
        Returns:
            Optional[str]: 'long', 'short', None
        """
        pos = self.positions.get(symbol)
        return pos['side'] if pos else None
    
    async def is_in_cooldown(self, symbol: str) -> bool:
        """
        쿨다운 상태 확인
        
        Args:
            symbol: 심볼명
            
        Returns:
            bool: 쿨다운 중 여부
        """
        async with self.lock:
            if symbol not in self.cooldowns:
                return False
            
            cooldown_end = self.cooldowns[symbol]
            now = datetime.now()
            
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds()
                self.logger.debug(f"{symbol}: 쿨다운 중 (남은 시간: {remaining:.1f}초)")
                return True
            
            # 쿨다운 종료
            del self.cooldowns[symbol]
            return False
    
    async def set_cooldown(self, symbol: str, multiplier: int = 1) -> None:
        """
        쿨다운 설정
        
        Args:
            symbol: 심볼명
            multiplier: 쿨다운 배수
        """
        async with self.lock:
            cooldown_seconds = config.cooldown_seconds * multiplier
            self.cooldowns[symbol] = datetime.now() + timedelta(seconds=cooldown_seconds)
            self.logger.info(f"{symbol}: 쿨다운 설정 ({cooldown_seconds}초)")
    
    async def open_position(self, symbol: str, side: str, quantity: float) -> bool:
        """
        포지션 오픈 - 수정 버전 (마진 모드 설정 추가)
        
        Args:
            symbol: 심볼명
            side: 'long' or 'short'
            quantity: 수량
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 포지션 중복 체크
            if self.has_position(symbol):
                self.logger.warning(f"{symbol}: 이미 포지션 존재")
                return False
            
            # 쿨다운 체크
            if await self.is_in_cooldown(symbol):
                return False
            
            # 마진 모드 설정 (ISOLATED)
            margin_mode_result = await self.rest_manager.set_margin_mode(symbol, 'ISOLATED')
            if margin_mode_result is None:
                # 이미 설정되어 있을 수 있음 (400 에러이지만 정상)
                self.logger.debug(f"{symbol}: 마진 모드 이미 ISOLATED로 설정됨")
            
            # 레버리지 설정
            await self.rest_manager.set_leverage(symbol, config.leverage)
            
            # 주문 실행
            order_side = 'BUY' if side == 'long' else 'SELL'
            result = await self.rest_manager.place_order(symbol, order_side, quantity)
            
            if result:
                self.logger.info(f"{symbol}: {side} 포지션 오픈 성공 (수량: {quantity})")
                await self.set_cooldown(symbol)  # 기본 쿨다운 설정
                return True
            else:
                self.logger.error(f"{symbol}: 포지션 오픈 실패")
            
            return False
            
        except Exception as e:
            self.logger.error(f"{symbol}: 포지션 오픈 실패 - {e}")
            return False
    
    async def close_position(self, symbol: str, reason: str = "signal") -> bool:
        """
        포지션 종료
        
        Args:
            symbol: 심볼명
            reason: 종료 사유
            
        Returns:
            bool: 성공 여부
        """
        try:
            position = self.get_position(symbol)
            if not position:
                self.logger.warning(f"{symbol}: 포지션 없음")
                return False
            
            # 반대 주문으로 포지션 종료
            order_side = 'SELL' if position['side'] == 'long' else 'BUY'
            quantity = abs(position['position_amt'])
            
            result = await self.rest_manager.place_order(symbol, order_side, quantity)
            
            if result:
                self.logger.info(f"{symbol}: 포지션 종료 성공 (사유: {reason})")
                
                # 스탑로스인 경우 쿨다운 배수 적용
                if reason == "stop_loss":
                    await self.set_cooldown(symbol, config.cooldown_multiplier)
                else:
                    await self.set_cooldown(symbol)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"{symbol}: 포지션 종료 실패 - {e}")
            return False
    
    def calculate_position_size(self, symbol: str, account_balance: float, current_price: float) -> float:
        """
        포지션 크기 계산 - 수정 버전 (설정 가능한 비율)
        
        Args:
            symbol: 심볼명
            account_balance: 계정 잔고
            current_price: 현재 가격
            
        Returns:
            float: 포지션 크기 (계약 수량)
        """
        try:
            # 심볼 정보 가져오기
            from src.core.symbol_manager import SymbolManager
            symbol_info = self.symbol_manager.get_symbol_info(symbol) if hasattr(self, 'symbol_manager') else None
            
            # 계정의 설정된 비율 사용 (config에서 가져옴)
            position_ratio = config.position_size_ratio
            position_value_usdt = account_balance * position_ratio
            
            # 레버리지 적용된 포지션 가치
            leveraged_value = position_value_usdt * config.leverage
            
            # 계약 수량 계산 (가치 / 현재가격)
            quantity = leveraged_value / current_price
            
            # 심볼 정밀도 적용
            if symbol_info:
                # 최소 수량 확인
                min_qty = symbol_info.get('minQty', 0.001)
                quantity_precision = symbol_info.get('quantityPrecision', 3)
                
                # 정밀도 적용
                quantity = round(quantity, quantity_precision)
                
                # 최소 수량 체크
                if quantity < min_qty:
                    self.logger.warning(f"{symbol}: 계산된 수량({quantity})이 최소 수량({min_qty})보다 작음")
                    quantity = min_qty
            
            self.logger.info(
                f"{symbol}: 포지션 크기 계산 - "
                f"잔고: ${account_balance:.2f}, "
                f"비율: {position_ratio*100:.1f}%, "
                f"레버리지: {config.leverage}x, "
                f"현재가: ${current_price:.4f}, "
                f"수량: {quantity}"
            )
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"{symbol}: 포지션 크기 계산 실패 - {e}")
            # 에러 시 최소 수량 반환
            return 0.001