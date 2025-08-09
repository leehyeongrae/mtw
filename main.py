"""
메인 트레이딩 시스템 (DRY, KISS, YAGNI 원칙 통합)
"""
import asyncio
import signal
import sys
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

# 모듈 임포트
from src.utils.config import config
from src.utils.logger import get_logger
from src.core.candle_manager import CandleManager
from src.core.websocket_manager import WebSocketManager
from src.core.rest_manager import RestManager
from src.core.symbol_manager import SymbolManager
from src.trading.indicators import Indicators
from src.trading.signals import SignalGenerator
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.telegram.bot import TelegramBot

class TradingSystem:
    """메인 트레이딩 시스템"""
    
    def __init__(self):
        self.logger = get_logger("trading_system")
        self.running = False
        
        # 매니저 초기화 (순서 중요)
        self.candle_manager = CandleManager()
        self.rest_manager = RestManager()
        self.symbol_manager = SymbolManager(self.rest_manager)
        self.position_manager = PositionManager(self.rest_manager, self.symbol_manager)  # 수정: symbol_manager 전달
        self.risk_manager = RiskManager(self.position_manager)
        self.websocket_manager = WebSocketManager(self.candle_manager)
        self.telegram_bot = TelegramBot(self)
        
        # 심볼별 신호 생성기 및 지표 캐시
        self.signal_generators: Dict[str, SignalGenerator] = {}
        self.indicators_cache: Dict[str, Dict] = {}
        self.hurst_cache: Dict[str, np.ndarray] = {}
        
    async def initialize(self) -> bool:
        """시스템 초기화 - 콜백 등록 수정"""
        try:
            self.logger.info("트레이딩 시스템 초기화 시작...")
            
            # 설정 검증
            if not config.validate():
                return False
            
            # REST API 초기화
            await self.rest_manager.initialize()
            
            # 심볼 리스트 초기화
            symbols = await self.symbol_manager.update_top_symbols()
            if not symbols:
                self.logger.error("심볼 리스트 생성 실패")
                return False
            
            self.logger.info(f"거래 심볼: {symbols}")
            
            # 심볼별 초기화
            for symbol in symbols:
                # 신호 생성기 생성
                self.signal_generators[symbol] = SignalGenerator(symbol)
                
                # 캔들 데이터 초기화
                candles = await self.rest_manager.get_klines(symbol, config.candle_limit)
                if candles:
                    await self.candle_manager.initialize_candles(symbol, candles)
                    
                    # 초기 Hurst 지수 계산
                    await self._calculate_hurst(symbol)
            
            # 포지션 업데이트
            await self.position_manager.update_positions()
            
            # 웹소켓 콜백 등록 (수정)
            #self.websocket_manager.register_callback('candle_update', self._on_candle_update)
            self.websocket_manager.register_callback('candle_closed', self._on_candle_closed)
            self.websocket_manager.register_callback('realtime_update', self._on_realtime_update)  # 추가
            
            # 텔레그램 봇 초기화
            await self.telegram_bot.initialize()
            
            # 텔레그램 시작 알림 추가
            await self.telegram_bot.send_message("🚀 트레이딩 봇이 시작되었습니다!")
            
            # 주기적 상태 출력 태스크 시작
            self.status_print_task = asyncio.create_task(self._periodic_status_print())
            
            self.logger.info("시스템 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"초기화 실패: {e}")
            return False

    async def _on_realtime_update(self, data: Dict) -> None:
        """
        실시간 업데이트 처리 (새로운 메서드)
        웹소켓 데이터 수신시 즉시 신호 및 리스크 체크
        """
        symbol = data['symbol']
        current_price = data['current_price']
        
        try:
            # 쿨다운 체크
            if await self.position_manager.is_in_cooldown(symbol):
                return
            
            # 캔들 데이터 가져오기
            df = await self.candle_manager.get_candles_for_analysis(symbol)
            if df is None or len(df) < 100:
                return
            
            # 지표 계산 (Hurst는 캐시 사용)
            indicators = await self._calculate_indicators_realtime(symbol, df)
            
            # 현재 포지션 확인
            current_position = self.position_manager.get_position_side(symbol)
            
            # 포지션이 있는 경우 - TP/SL 체크
            if current_position:
                # 리스크 체크 (TP/SL)
                risk_action = await self.risk_manager.check_risk_limits(symbol, current_price)
                if risk_action:
                    await self._handle_risk_action(symbol, risk_action)
                    return
                
                # 트레일링 스탑 업데이트
                atr = indicators['atr'][-1] if len(indicators['atr']) > 0 else 0
                if atr > 0:
                    self.risk_manager.update_trailing_stop(symbol, current_price, current_position, atr)
            
            # 신호 생성 및 실행
            signal_gen = self.signal_generators[symbol]
            signal = signal_gen.generate_signal(indicators, current_position)
            
            if signal:
                await self._execute_signal(symbol, signal, indicators)
                
        except Exception as e:
            self.logger.error(f"{symbol}: 실시간 업데이트 처리 실패 - {e}")

    async def _calculate_indicators_realtime(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        실시간 지표 계산 (새로운 메서드)
        Hurst는 캐시 사용, 나머지는 실시간 계산
        """
        try:
            # 지표 계산 (Hurst 제외)
            indicators = Indicators.calculate_all(
                df,
                cci_length=config.cci_length,
                cci_smoothing=config.cci_smoothing,
                rsi_length=config.rsi_length,
                supertrend_atr_length=config.supertrend_length,
                supertrend_multiplier=config.supertrend_multiplier,
                psar_start=config.psar_start,
                psar_increment=config.psar_increment,
                psar_maximum=config.psar_maximum,
                vi_length=config.vi_length,
                mfi_length=config.mfi_length,
                atr_length=config.atr_length,
                adx_length=config.adx_length,
                adx_smoothing=config.adx_smoothing,
                symbol=symbol,
                exclude_hurst=True
            )
            
            # 캐시된 Hurst 추가
            if symbol in self.hurst_cache:
                indicators['hurst_smoothed'] = self.hurst_cache[symbol]
            else:
                # Hurst가 없으면 기본값
                indicators['hurst_smoothed'] = np.array([0.5] * len(df))
            
            # 지표 캐싱
            self.indicators_cache[symbol] = indicators
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"{symbol}: 실시간 지표 계산 실패 - {e}")
            return {}

    async def _periodic_status_print(self) -> None:
        """
        60초마다 모든 심볼 상태 출력 (새로운 메서드)
        """
        while self.running:
            try:
                await asyncio.sleep(60)
                
                for symbol in self.symbol_manager.active_symbols:
                    await self._print_enhanced_status(symbol)
                    
            except Exception as e:
                self.logger.error(f"주기적 상태 출력 오류: {e}")

    async def _print_enhanced_status(self, symbol: str) -> None:
        """
        향상된 상태 출력 - 시간 형식 개선 버전
        """
        try:
            # 최근 캔들 가져오기
            recent_candles = self.candle_manager.get_latest_candles(symbol, 5)
            if recent_candles is None:
                return
            
            # 현재 진행 캔들
            current = self.candle_manager.current_candles.get(symbol, {})
            
            # 지표 가져오기
            indicators = self.indicators_cache.get(symbol, {})
            
            if not indicators:
                return
            
            # 추세장 판별
            signal_gen = self.signal_generators[symbol]
            is_trending = signal_gen.is_trending_market_by_slope(
                indicators.get('adx', np.array([])),
                indicators.get('hurst_smoothed', np.array([]))
            )
            
            # ADX와 Hurst 기울기 계산
            adx_slope = signal_gen.calculate_slope(
                indicators.get('adx', np.array([])), 
                config.trend_detection_candles
            )
            hurst_slope = signal_gen.calculate_slope(
                indicators.get('hurst_smoothed', np.array([])), 
                config.trend_detection_candles
            )
            
            # 오실레이터 점수 계산
            oscillator_score = signal_gen.calculate_score(
                indicators.get('rsi', [50])[-1],
                indicators.get('cci', [0])[-1],
                indicators.get('mfi', [50])[-1]
            )
            
            # 출력
            print(f"\n{'='*70}")
            print(f"[{symbol}] 상태 - {'추세장' if is_trending else '횡보장'}")
            print(f"{'='*70}")
            
            # 과거 4개 캔들 (REST)
            print("📊 과거 4개 캔들 (REST API):")
            for i in range(-4, 0):
                candle = recent_candles.iloc[i]
                # 시간을 읽기 쉬운 형식으로 변환
                time_str = candle['open_time'].strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f"  #{i} {time_str}: O:{candle['open']:.2f} H:{candle['high']:.2f} "
                    f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:.0f}")
            
            # 현재 캔들 (WebSocket)
            if current:
                # Unix timestamp를 datetime으로 변환
                import pandas as pd
                current_time = pd.to_datetime(current.get('open_time', 0), unit='ms')
                time_str = current_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                
                print(f"\n🔴 현재 진행 캔들 (WebSocket) - {time_str}:")
                print(f"  O:{current.get('open', 0):.2f} H:{current.get('high', 0):.2f} "
                    f"L:{current.get('low', 0):.2f} C:{current.get('close', 0):.2f} "
                    f"V:{current.get('volume', 0):.0f}")
            
            # 계산된 지표들 (이하 동일)
            print(f"\n📈 지표값:")
            print(f"  ADX: {indicators.get('adx', [0])[-1]:.2f} (기울기: {adx_slope:.4f})")
            print(f"  Hurst: {indicators.get('hurst_smoothed', [0])[-1]:.3f} (기울기: {hurst_slope:.4f})")
            print(f"  RSI: {indicators.get('rsi', [0])[-1]:.1f}")
            print(f"  CCI: {indicators.get('cci', [0])[-1]:.1f}")
            print(f"  MFI: {indicators.get('mfi', [0])[-1]:.1f}")
            print(f"  VI+: {indicators.get('vi_plus', [0])[-1]:.3f}")
            print(f"  VI-: {indicators.get('vi_minus', [0])[-1]:.3f}")
            print(f"  ATR: {indicators.get('atr', [0])[-1]:.4f}")
            
            # PSAR와 Supertrend
            psar_trend = indicators.get('psar_trend', [0])[-1]
            supertrend_trend = indicators.get('trend_direction', [0])[-1]
            print(f"  PSAR: {'상승↑' if psar_trend == 1 else '하락↓'}")
            print(f"  Supertrend: {'상승↑' if supertrend_trend == 1 else '하락↓'}")
            
            # 오실레이터 정규화 점수
            print(f"\n💯 오실레이터 정규화 점수: {oscillator_score:.2f}")
            if oscillator_score <= config.oscillator_long_threshold:
                print(f"    → 과매도 신호 (롱 진입 가능)")
            elif oscillator_score >= config.oscillator_short_threshold:
                print(f"    → 과매수 신호 (숏 진입 가능)")
            else:
                print(f"    → 중립")
            
            # 추세장 판단 결과
            print(f"\n🎯 추세장 판단:")
            print(f"  ADX 기울기 > 0: {'✅' if adx_slope > 0 else '❌'}")
            print(f"  Hurst 기울기 > 0: {'✅' if hurst_slope > 0 else '❌'}")
            print(f"  → 결과: {'추세장' if is_trending else '횡보장'}")
            
            # 포지션 정보
            position = self.position_manager.get_position(symbol)
            if position:
                print(f"\n💼 포지션:")
                print(f"  방향: {'롱' if position['side'] == 'long' else '숏'}")
                print(f"  진입가: {position['entry_price']:.4f}")
                print(f"  현재가: {position['mark_price']:.4f}")
                print(f"  미실현 PnL: ${position['unrealized_pnl']:.2f}")
                
                # 리스크 정보
                risk_info = self.risk_manager.get_risk_info(symbol)
                if risk_info['stop_loss']:
                    print(f"  SL: {risk_info['stop_loss']:.4f}")
                if risk_info['take_profit']:
                    print(f"  TP: {risk_info['take_profit']:.4f}")
            
            print(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"상태 출력 오류: {e}")

    async def _calculate_hurst(self, symbol: str) -> None:
        """Hurst 지수 계산 및 캐싱"""
        try:
            df = await self.candle_manager.get_candles_for_analysis(symbol)
            if df is None or len(df) < 150:
                return
            
            # Hurst 계산 (무거운 작업)
            hurst_raw, hurst_smoothed = Indicators.hurst(
                df, 
                config.hurst_window,
                config.hurst_rs_lag,
                config.hurst_smoothing,
                symbol
            )
            
            # 캐싱
            self.hurst_cache[symbol] = hurst_smoothed
            self.logger.debug(f"{symbol}: Hurst 지수 계산 완료")
            
        except Exception as e:
            self.logger.error(f"{symbol}: Hurst 계산 실패 - {e}")
    
    async def _on_candle_update(self, data: Dict) -> None:
        """캔들 업데이트 이벤트 처리"""
        symbol = data['symbol']
        
        # 60초마다 상태 출력 (요구사항)
        if int(asyncio.get_event_loop().time()) % 60 == 0:
            asyncio.create_task(self._print_status(symbol))
    
    async def _on_candle_closed(self, data: Dict) -> None:
        """
        캔들 종료 이벤트 처리 - 완전 수정 버전
        REST API로 최신 캔들들을 가져와서 업데이트
        """
        symbol = data['symbol']
        self.logger.info(f"{symbol}: 캔들 종료 처리 시작")
        
        try:
            # REST API로 최근 10개 캔들 가져오기 (여유있게)
            await asyncio.sleep(config.rest_api_delay)  # 캔들 완전 종료 대기
            
            recent_candles = await self.rest_manager.get_klines(symbol, 10)
            if not recent_candles or len(recent_candles) < 2:
                self.logger.error(f"{symbol}: 최근 캔들 조회 실패")
                return
            
            # 현재 시간
            import time
            current_time_ms = int(time.time() * 1000)
            
            # 완성된 캔들들만 필터링 (close_time이 현재 시간보다 이전)
            completed_candles = []
            for candle in recent_candles:
                if candle[6] < current_time_ms:  # close_time이 현재보다 이전
                    completed_candles.append(candle)
            
            if not completed_candles:
                self.logger.warning(f"{symbol}: 완성된 캔들이 없음")
                return
            
            # 가장 최근 완성된 캔들
            latest_completed = completed_candles[-1]
            
            # 캔들 추가 시도
            success = await self.candle_manager.add_completed_candle(symbol, latest_completed)
            
            if not success:
                self.logger.warning(f"{symbol}: 캔들 추가 실패, 전체 새로고침")
                # 전체 캔들 데이터 새로고침
                all_candles = await self.rest_manager.get_klines(symbol, config.candle_limit)
                if all_candles:
                    await self.candle_manager.initialize_candles(symbol, all_candles)
            
            # 디버그: 현재 캔들 상태 출력
            latest_candles = self.candle_manager.get_latest_candles(symbol, 5)
            if latest_candles is not None and len(latest_candles) > 0:
                last_candle = latest_candles.iloc[-1]
                self.logger.info(
                    f"{symbol}: 최신 REST 캔들 - "
                    f"시간: {last_candle['open_time'].strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"종가: {last_candle['close']:.4f}"
                )
            
            # Hurst 재계산
            await self._calculate_hurst(symbol)
            
            # 신호 생성
            await self._process_signals(symbol)
            
        except Exception as e:
            self.logger.error(f"{symbol}: 캔들 종료 처리 실패 - {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    async def _process_signals(self, symbol: str) -> None:
        """신호 처리"""
        try:
            # 쿨다운 체크
            if await self.position_manager.is_in_cooldown(symbol):
                return
            
            # 캔들 데이터 가져오기
            df = await self.candle_manager.get_candles_for_analysis(symbol)
            if df is None or len(df) < 100:
                return
            
            # 지표 계산 (Hurst 제외)
            indicators = Indicators.calculate_all(
                df,
                cci_length=config.cci_length,
                cci_smoothing=config.cci_smoothing,
                rsi_length=config.rsi_length,
                supertrend_atr_length=config.supertrend_length,
                supertrend_multiplier=config.supertrend_multiplier,
                psar_start=config.psar_start,
                psar_increment=config.psar_increment,
                psar_maximum=config.psar_maximum,
                vi_length=config.vi_length,
                mfi_length=config.mfi_length,
                atr_length=config.atr_length,
                adx_length=config.adx_length,
                adx_smoothing=config.adx_smoothing,
                symbol=symbol,
                exclude_hurst=True  # Hurst는 캐시 사용
            )
            
            # 캐시된 Hurst 추가
            if symbol in self.hurst_cache:
                indicators['hurst_smoothed'] = self.hurst_cache[symbol]
            
            # 지표 캐싱
            self.indicators_cache[symbol] = indicators
            
            # 현재 포지션 확인
            current_position = self.position_manager.get_position_side(symbol)
            
            # 신호 생성
            signal_gen = self.signal_generators[symbol]
            signal = signal_gen.generate_signal(indicators, current_position)
            
            if signal:
                await self._execute_signal(symbol, signal, indicators)
            
            # 리스크 체크 (포지션이 있는 경우)
            if current_position:
                current_price = float(df.iloc[-1]['close'])
                atr = indicators['atr'][-1] if len(indicators['atr']) > 0 else 0
                
                # 트레일링 스탑 업데이트
                self.risk_manager.update_trailing_stop(symbol, current_price, current_position, atr)
                
                # 리스크 한도 체크
                risk_action = await self.risk_manager.check_risk_limits(symbol, current_price)
                if risk_action:
                    await self._handle_risk_action(symbol, risk_action)
                    
        except Exception as e:
            self.logger.error(f"{symbol}: 신호 처리 실패 - {e}")
    
    async def _execute_signal(self, symbol: str, signal: Dict, indicators: Dict) -> None:
        """
        신호 실행 - 중복 방지 강화 버전
        """
        try:
            action = signal['action']
            
            # 진입 신호
            if 'entry' in action:
                # 1. 캐시 확인
                if self.position_manager.has_position(symbol):
                    self.logger.debug(f"{symbol}: 이미 포지션 존재 (캐시) - 신호 무시")
                    return
                
                # 2. REST API로 실시간 포지션 확인 (중요!)
                await self.position_manager.update_positions()
                if self.position_manager.has_position(symbol):
                    self.logger.debug(f"{symbol}: 이미 포지션 존재 (REST) - 신호 무시")
                    return
                
                # 3. 쿨다운 체크
                if await self.position_manager.is_in_cooldown(symbol):
                    self.logger.debug(f"{symbol}: 쿨다운 중 - 신호 무시")
                    return
                
                # 계정 잔고 확인
                balance_data = await self.rest_manager.get_account_balance()
                if not balance_data:
                    self.logger.error(f"{symbol}: 계정 잔고 조회 실패")
                    return
                
                # USDT 잔고 찾기
                usdt_balance = 0
                for asset in balance_data:
                    if asset.get('asset') == 'USDT':
                        usdt_balance = float(asset.get('balance', 0))
                        break
                
                if usdt_balance <= 0:
                    self.logger.error(f"{symbol}: USDT 잔고 부족 (${usdt_balance:.2f})")
                    return
                
                # 현재 가격 가져오기
                df = await self.candle_manager.get_candles_for_analysis(symbol)
                if df is None or len(df) == 0:
                    self.logger.error(f"{symbol}: 현재 가격 조회 실패")
                    return
                
                current_price = float(df.iloc[-1]['close'])
                
                # 포지션 크기 계산
                quantity = self.position_manager.calculate_position_size(symbol, usdt_balance, current_price)
                
                # 심볼 정보에서 정밀도 적용
                symbol_info = self.symbol_manager.get_symbol_info(symbol)
                if symbol_info:
                    precision = symbol_info.get('quantityPrecision', 3)
                    min_qty = symbol_info.get('minQty', 0.001)
                    
                    # 정밀도 적용
                    quantity = round(quantity, precision)
                    
                    # 최소 수량 체크
                    if quantity < min_qty:
                        self.logger.warning(f"{symbol}: 수량이 최소값보다 작음. 최소값으로 조정: {min_qty}")
                        quantity = min_qty
                
                # 포지션 오픈
                side = 'long' if 'long' in action else 'short'
                
                self.logger.info(
                    f"{symbol}: {side.upper()} 진입 시도 - "
                    f"잔고: ${usdt_balance:.2f}, "
                    f"현재가: ${current_price:.4f}, "
                    f"수량: {quantity}"
                )
                
                # open_position이 중복 체크를 포함하도록 수정됨
                success = await self.position_manager.open_position(symbol, side, quantity)
                
                if success:
                    # 리스크 한도 설정
                    entry_price = current_price
                    atr = indicators['atr'][-1] if len(indicators.get('atr', [])) > 0 else entry_price * 0.01
                    
                    self.risk_manager.set_initial_limits(symbol, entry_price, side, atr)
                    
                    # 텔레그램 알림
                    await self.telegram_bot.notify_trade(symbol, action, {
                        'price': entry_price,
                        'quantity': quantity,
                        'market_type': signal['market_type'],
                        'reason': signal.get('reason', ''),
                        'balance': usdt_balance
                    })
                else:
                    self.logger.debug(f"{symbol}: 포지션 오픈 실패 또는 중복")
            
            # 청산 신호
            elif 'exit' in action:
                success = await self.position_manager.close_position(symbol, signal.get('reason', 'signal'))
                
                if success:
                    # 리스크 한도 제거
                    self.risk_manager.clear_limits(symbol)
                    
                    # 텔레그램 알림
                    position = self.position_manager.get_position(symbol)
                    if position:
                        pnl_percent = (position['unrealized_pnl'] / position['margin']) * 100 if position['margin'] > 0 else 0
                        
                        await self.telegram_bot.notify_trade(symbol, action, {
                            'price': position['mark_price'],
                            'pnl': position['unrealized_pnl'],
                            'pnl_percent': pnl_percent,
                            'reason': signal.get('reason', '')
                        })
                        
        except Exception as e:
            self.logger.error(f"{symbol}: 신호 실행 실패 - {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    async def _handle_risk_action(self, symbol: str, action: str) -> None:
        """리스크 액션 처리"""
        try:
            success = await self.position_manager.close_position(symbol, action)
            
            if success:
                # 리스크 한도 제거
                self.risk_manager.clear_limits(symbol)
                
                # 텔레그램 알림
                position = self.position_manager.get_position(symbol)
                if position:
                    await self.telegram_bot.notify_trade(symbol, action, {
                        'price': position['mark_price'],
                        'pnl': position['unrealized_pnl'],
                        'pnl_percent': (position['unrealized_pnl'] / position['margin']) * 100,
                        'reason': action
                    })
                    
        except Exception as e:
            self.logger.error(f"{symbol}: 리스크 액션 처리 실패 - {e}")
    
    async def _print_status(self, symbol: str) -> None:
        """상태 출력 (60초 주기)"""
        try:
            # 최근 캔들 가져오기
            recent_candles = self.candle_manager.get_latest_candles(symbol, 5)
            if recent_candles is None:
                return
            
            # 현재 진행 캔들
            current = self.candle_manager.current_candles.get(symbol, {})
            
            # 지표 가져오기
            indicators = self.indicators_cache.get(symbol, {})
            
            # 추세장 판별
            signal_gen = self.signal_generators[symbol]
            is_trending = signal_gen.is_trending_market_by_slope(
                indicators.get('adx', np.array([])),
                indicators.get('hurst_smoothed', np.array([]))
            )
            
            # 출력
            print(f"\n{'='*60}")
            print(f"[{symbol}] 상태 - {'추세장' if is_trending else '횡보장'}")
            print(f"{'='*60}")
            
            # 과거 캔들
            print("과거 4개 캔들 (REST):")
            for i in range(-4, 0):
                candle = recent_candles.iloc[i]
                print(f"  #{i}: O:{candle['open']:.2f} H:{candle['high']:.2f} "
                      f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:.0f}")
            
            # 현재 캔들
            if current:
                print(f"\n현재 진행 캔들 (WebSocket):")
                print(f"  O:{current.get('open', 0):.2f} H:{current.get('high', 0):.2f} "
                      f"L:{current.get('low', 0):.2f} C:{current.get('close', 0):.2f} "
                      f"V:{current.get('volume', 0):.0f}")
            
            # 지표
            if indicators:
                print(f"\n지표:")
                print(f"  ADX: {indicators.get('adx', [0])[-1]:.2f}")
                print(f"  Hurst: {indicators.get('hurst_smoothed', [0])[-1]:.3f}")
                print(f"  RSI: {indicators.get('rsi', [0])[-1]:.1f}")
                print(f"  CCI: {indicators.get('cci', [0])[-1]:.1f}")
                print(f"  MFI: {indicators.get('mfi', [0])[-1]:.1f}")
                
                # 오실레이터 점수 (횡보장)
                if not is_trending:
                    score = signal_gen.calculate_score(
                        indicators.get('rsi', [50])[-1],
                        indicators.get('cci', [0])[-1],
                        indicators.get('mfi', [50])[-1]
                    )
                    print(f"  오실레이터 점수: {score:.2f}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"상태 출력 오류: {e}")
    
    async def run(self) -> None:
        """메인 실행 루프 - 수정"""
        self.running = True
        
        try:
            # 웹소켓 시작
            symbols = self.symbol_manager.active_symbols
            websocket_task = asyncio.create_task(self.websocket_manager.start(symbols))
            
            # 주기적 업데이트 태스크
            position_update_task = asyncio.create_task(self._periodic_position_update())
            symbol_update_task = asyncio.create_task(self.symbol_manager.periodic_update())
            
            # 메인 루프
            while self.running:
                await asyncio.sleep(1)
            
            # 태스크 종료
            websocket_task.cancel()
            position_update_task.cancel()
            symbol_update_task.cancel()
            if hasattr(self, 'status_print_task'):
                self.status_print_task.cancel()
            
        except Exception as e:
            self.logger.error(f"실행 오류: {e}")
            await self.telegram_bot.notify_error(str(e))
    
    async def _periodic_position_update(self) -> None:
        """주기적 포지션 업데이트"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 1분마다
                await self.position_manager.update_positions()
            except Exception as e:
                self.logger.error(f"포지션 업데이트 오류: {e}")

    async def stop(self) -> None:
        """시스템 종료"""
        self.logger.info("시스템 종료 시작...")
        self.running = False
        
        # 모든 포지션 종료
        for symbol in list(self.position_manager.positions.keys()):
            await self.position_manager.close_position(symbol, "system_shutdown")
        
        # 매니저 종료
        await self.websocket_manager.stop()
        await self.rest_manager.close()
        await self.telegram_bot.stop()
        
        self.logger.info("시스템 종료 완료")
    
    async def get_account_info(self) -> Optional[Dict]:
        """계정 정보 조회 (텔레그램용)"""
        try:
            balance_data = await self.rest_manager.get_account_balance()
            positions = await self.rest_manager.get_positions()
            
            if not balance_data:
                return None
            
            total_balance = sum(float(b['balance']) for b in balance_data)
            total_unrealized_pnl = sum(float(p['unRealizedProfit']) for p in positions) if positions else 0
            position_count = len(self.position_manager.positions)
            
            # 마진 계산
            total_margin = sum(
                float(p.get('isolatedWallet', 0) or p.get('positionInitialMargin', 0)) 
                for p in positions
            ) if positions else 0
            
            margin_ratio = (total_margin / total_balance * 100) if total_balance > 0 else 0
            
            return {
                'balance': total_balance,
                'unrealized_pnl': total_unrealized_pnl,
                'margin_ratio': margin_ratio,
                'position_count': position_count
            }
            
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return None
    
    async def get_all_status(self) -> Dict:
        """전체 심볼 상태 조회 (텔레그램용)"""
        status = {}
        
        for symbol in self.symbol_manager.active_symbols:
            status[symbol] = await self.get_symbol_status(symbol)
        
        return status
    
    async def get_symbol_status(self, symbol: str) -> Optional[Dict]:
        """특정 심볼 상태 조회 (텔레그램용)"""
        try:
            # 캔들 데이터
            df = await self.candle_manager.get_candles_for_analysis(symbol)
            if df is None or len(df) == 0:
                return None
            
            current_price = float(df.iloc[-1]['close'])
            
            # 지표
            indicators = self.indicators_cache.get(symbol, {})
            
            # 추세장 판별
            signal_gen = self.signal_generators.get(symbol)
            is_trending = 0
            if signal_gen and 'adx' in indicators and 'hurst_smoothed' in indicators:
                is_trending = signal_gen.is_trending_market_by_slope(
                    indicators['adx'],
                    indicators['hurst_smoothed']
                )
            
            # 기본 정보
            info = {
                'price': current_price,
                'market_type': '추세장' if is_trending else '횡보장'
            }
            
            # 지표 정보
            if indicators:
                info.update({
                    'adx': indicators.get('adx', [0])[-1],
                    'hurst': indicators.get('hurst_smoothed', [0])[-1],
                    'rsi': indicators.get('rsi', [0])[-1],
                    'cci': indicators.get('cci', [0])[-1],
                    'mfi': indicators.get('mfi', [0])[-1],
                    'vi_plus': indicators.get('vi_plus', [0])[-1],
                    'vi_minus': indicators.get('vi_minus', [0])[-1]
                })
                
                # 오실레이터 점수
                if signal_gen:
                    score = signal_gen.calculate_score(
                        info['rsi'], info['cci'], info['mfi']
                    )
                    info['oscillator_score'] = score
                
                # 추세 정보
                psar_trend = indicators.get('psar_trend', [0])[-1]
                supertrend_trend = indicators.get('trend_direction', [0])[-1]
                
                if psar_trend == 1 and supertrend_trend == 1:
                    info['trend'] = '상승'
                elif psar_trend == -1 and supertrend_trend == -1:
                    info['trend'] = '하락'
                else:
                    info['trend'] = '혼조'
            
            # 포지션 정보
            position = self.position_manager.get_position(symbol)
            if position:
                info['position'] = {
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'pnl': position['unrealized_pnl'],
                    'stop_loss': self.risk_manager.stop_losses.get(symbol),
                    'take_profit': self.risk_manager.take_profits.get(symbol)
                }
            else:
                info['position'] = None
            
            # 24시간 변동
            if len(df) >= 96:  # 24시간 = 96개 캔들 (15분)
                price_24h_ago = float(df.iloc[-96]['close'])
                change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                info['change_24h'] = change_24h
            
            # 거래량
            info['volume'] = float(df.iloc[-1]['volume']) if 'volume' in df.columns else 0
            
            return info
            
        except Exception as e:
            self.logger.error(f"{symbol}: 상태 조회 실패 - {e}")
            return None


async def main():
    """메인 함수"""
    logger = get_logger("main")
    
    # 시스템 생성
    system = TradingSystem()
    
    # 종료 시그널 핸들러
    def signal_handler(sig, frame):
        logger.info(f"종료 시그널 수신: {sig}")
        asyncio.create_task(system.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 초기화
        if not await system.initialize():
            logger.error("시스템 초기화 실패")
            return
        
        logger.info("트레이딩 시스템 시작")
        
        # 실행
        await system.run()
        
    except Exception as e:
        logger.error(f"시스템 오류: {e}")
        
    finally:
        await system.stop()
        logger.info("프로그램 종료")


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())