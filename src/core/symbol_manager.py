"""
Symbol list management based on volume and listing age - 개선된 버전
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import ccxt
import time
from src.utils.logger import get_logger
from src.utils.config import config

class SymbolManager:
    """Manages trading symbol selection with enhanced validation"""
    
    def __init__(self):
        self.logger = get_logger("symbol_manager")
        self.exchange = self._init_exchange()
        self.min_volume_usdt = 5000000  # 최소 500만 USDT 거래량
        self.quote_currency = 'USDT'
        self.max_symbols = config.symbol_count
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange instance with proper futures configuration"""
        try:
            self.logger.info("Initializing CCXT Binance exchange...")
            
            # 기본 설정
            exchange_config = {
                'apiKey': config.binance_api_key,
                'secret': config.binance_api_secret,
                'enableRateLimit': True,
                'timeout': 30000,
            }
            
            # testnet 설정
            if config.binance_testnet:
                exchange_config['sandbox'] = True
                self.logger.info("Using Binance testnet")
            
            # Exchange 인스턴스 생성
            exchange = ccxt.binance(exchange_config)
            
            # futures 시장으로 설정 (안전한 방식)
            if not hasattr(exchange, 'options'):
                exchange.options = {}
            
            # 여러 방식으로 futures 설정 시도
            exchange.options['defaultType'] = 'swap'  # 최신 방식
            
            self.logger.info(f"Exchange initialized with options: {exchange.options}")
            
            # 연결 테스트
            self._test_connection(exchange)
            
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise Exception(f"Exchange initialization failed: {e}")
    
    def _test_connection(self, exchange: ccxt.Exchange):
        """Test exchange connection and configuration"""
        try:
            self.logger.info("Testing exchange connection...")
            
            # 시장 정보 로딩 테스트
            markets = exchange.load_markets()
            self.logger.info(f"Successfully loaded {len(markets)} markets")
            
            # futures 시장 필터링 테스트
            futures_markets = self._filter_futures_markets(markets)
            self.logger.info(f"Found {len(futures_markets)} futures markets")
            
            if len(futures_markets) < 10:
                self.logger.warning("Very few futures markets found - configuration may be incorrect")
            
            # 샘플 futures 시장 로깅
            sample_futures = list(futures_markets.items())[:3]
            for symbol, market in sample_futures:
                self.logger.info(f"Sample futures market: {symbol} - Type: {market.get('type')}")
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise
    
    def _filter_futures_markets(self, markets: Dict) -> Dict:
        """Filter futures markets from all markets"""
        futures_markets = {}
        
        for symbol, market in markets.items():
            # 여러 조건으로 futures 마켓 확인
            market_type = market.get('type', '').lower()
            
            # futures 타입 확인 (여러 가능한 값)
            if market_type in ['future', 'swap', 'futures', 'derivative']:
                # USDT 페어만 선택
                if symbol.endswith(f'/{self.quote_currency}'):
                    # 활성 상태 확인
                    if market.get('active', False):
                        futures_markets[symbol] = market
        
        return futures_markets
    
    def _validate_symbol_data(self, symbol: str, ticker: Dict) -> bool:
        """Validate symbol data quality"""
        try:
            # 기본 데이터 존재 확인
            if not ticker:
                return False
            
            # 거래량 확인 (quoteVolume = USDT 거래량)
            quote_volume = ticker.get('quoteVolume', 0)
            if not quote_volume or quote_volume < self.min_volume_usdt:
                self.logger.debug(f"Low volume for {symbol}: {quote_volume}")
                return False
            
            # 가격 데이터 확인
            last_price = ticker.get('last', 0)
            if not last_price or last_price <= 0:
                self.logger.debug(f"Invalid price for {symbol}: {last_price}")
                return False
            
            # 24시간 변화율 확인 (극단적인 값 제외)
            percentage = ticker.get('percentage', 0)
            if abs(percentage or 0) > 50:  # 50% 이상 변동 제외
                self.logger.debug(f"Extreme price change for {symbol}: {percentage}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Error validating {symbol}: {e}")
            return False
    
    def _get_safe_symbols(self) -> List[str]:
        """Get manually curated safe symbols as backup"""
        safe_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
        ]
        
        # 설정된 개수만큼만 반환
        return safe_symbols[:self.max_symbols]
    
    def get_top_symbols(self) -> List[str]:
        """
        Get top symbols by 24h volume with enhanced validation
        
        Returns:
            List of symbol strings
        """
        try:
            self.logger.info("=== Starting symbol selection process ===")
            
            # 1. 시장 정보 로딩
            self.logger.info("Loading markets...")
            markets = self.exchange.load_markets()
            self.logger.info(f"Loaded {len(markets)} total markets")
            
            # 2. Futures 시장만 필터링
            futures_markets = self._filter_futures_markets(markets)
            self.logger.info(f"Filtered to {len(futures_markets)} futures markets")
            
            if len(futures_markets) < 5:
                raise Exception(f"Too few futures markets found: {len(futures_markets)}")
            
            # 3. 티커 데이터 가져오기
            self.logger.info("Fetching tickers...")
            
            # 재시도 로직 추가
            tickers = None
            for attempt in range(3):
                try:
                    tickers = self.exchange.fetch_tickers()
                    break
                except Exception as e:
                    self.logger.warning(f"Ticker fetch attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise
            
            self.logger.info(f"Fetched {len(tickers)} tickers")
            
            # 4. 유효한 심볼 선별
            eligible_symbols = []
            
            for symbol, market in futures_markets.items():
                try:
                    # 심볼 형식 변환 (CCXT 형식에서 바이낸스 형식으로)
                    binance_symbol = symbol.replace('/', '')
                    
                    # 티커 데이터 확인
                    ticker = tickers.get(symbol)
                    if not ticker:
                        # 다른 형식으로 시도
                        ticker = tickers.get(binance_symbol)
                    
                    if not ticker:
                        self.logger.debug(f"No ticker data for {symbol}")
                        continue
                    
                    # 데이터 검증
                    if not self._validate_symbol_data(symbol, ticker):
                        continue
                    
                    # 거래량 추출
                    quote_volume = ticker.get('quoteVolume', 0)
                    
                    eligible_symbols.append({
                        'symbol': binance_symbol,
                        'volume': quote_volume,
                        'price': ticker.get('last', 0),
                        'change': ticker.get('percentage', 0)
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error processing {symbol}: {e}")
                    continue
            
            self.logger.info(f"Found {len(eligible_symbols)} eligible symbols")
            
            # 5. 최소 개수 확인
            if len(eligible_symbols) < self.max_symbols:
                self.logger.warning(f"Only {len(eligible_symbols)} eligible symbols found, expected {self.max_symbols}")
                if len(eligible_symbols) == 0:
                    raise Exception("No eligible symbols found")
            
            # 6. 거래량 기준 정렬
            eligible_symbols.sort(key=lambda x: x['volume'], reverse=True)
            
            # 7. 상위 심볼 선택
            top_symbols = [s['symbol'] for s in eligible_symbols[:self.max_symbols]]
            
            # 8. 결과 로깅
            self.logger.info(f"Selected top {len(top_symbols)} symbols by volume:")
            for i, symbol_data in enumerate(eligible_symbols[:self.max_symbols]):
                self.logger.info(
                    f"  {i+1}. {symbol_data['symbol']}: "
                    f"Volume=${symbol_data['volume']:,.0f}, "
                    f"Price=${symbol_data['price']:.4f}, "
                    f"Change={symbol_data['change']:+.2f}%"
                )
            
            if len(top_symbols) == 0:
                raise Exception("No valid symbols selected")
            
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"=== Symbol selection failed ===")
            self.logger.error(f"Error: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 안전한 대체 심볼 사용 (하지만 로그에 명시)
            safe_symbols = self._get_safe_symbols()
            self.logger.error(f"Using safe backup symbols: {safe_symbols}")
            self.logger.error("*** WARNING: Using backup symbols - volume data may be outdated ***")
            
            return safe_symbols
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate symbols are still tradeable
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        try:
            self.logger.info(f"Validating {len(symbols)} symbols...")
            
            # 현재 시장 정보 가져오기
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            valid_symbols = []
            
            for symbol in symbols:
                # CCXT 형식으로 변환하여 확인
                ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}"  # BTCUSDT -> BTC/USDT
                
                market = markets.get(ccxt_symbol)
                ticker = tickers.get(ccxt_symbol)
                
                if market and ticker and market.get('active', False):
                    # 최소 거래량 확인
                    volume = ticker.get('quoteVolume', 0)
                    if volume >= self.min_volume_usdt:
                        valid_symbols.append(symbol)
                    else:
                        self.logger.warning(f"Low volume for {symbol}: {volume}")
                else:
                    self.logger.warning(f"Invalid or inactive symbol: {symbol}")
            
            self.logger.info(f"Validated {len(valid_symbols)}/{len(symbols)} symbols")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Symbol validation failed: {e}")
            return symbols  # 검증 실패 시 원본 반환
    
    def update_symbol_list(self) -> List[str]:
        """
        Update and return the current symbol list
        
        Returns:
            List of trading symbols
        """
        try:
            symbols = self.get_top_symbols()
            
            # 추가 검증
            if len(symbols) < 3:
                self.logger.error(f"Too few symbols returned: {len(symbols)}")
                return self._get_safe_symbols()
            
            # 중복 제거
            symbols = list(dict.fromkeys(symbols))  # 순서 유지하면서 중복 제거
            
            self.logger.info(f"Symbol list updated successfully: {symbols}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to update symbol list: {e}")
            return self._get_safe_symbols()