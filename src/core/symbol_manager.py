"""
Symbol list management based on volume and listing age - 개선된 버전
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
import time
from src.utils.logger import get_logger
from src.utils.config import config

class SymbolManager:
    """Manages trading symbol selection with enhanced validation"""
    
    def __init__(self):
        self.logger = get_logger("symbol_manager")

        self.min_volume_usdt = 1000000  # 최소 100만 USDT 거래량
        self.quote_currency = 'USDT'
        self.max_symbols = config.symbol_count

        # Binance API 설정
        self.base_url = "https://testnet.binancefuture.com" if config.binance_testnet else "https://fapi.binance.com"
        self.session = self._init_exchange()
        
    def _init_exchange(self) -> requests.Session:
        """Initialize requests session for Binance API calls"""
        try:
            self.logger.info("Initializing Binance API connection...")
            
            # requests 세션 생성
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; MTW-Bot/1.0)',
                'Content-Type': 'application/json'
            })
            
            # 연결 테스트
            self._test_connection(session)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API session: {e}")
            raise Exception(f"API session initialization failed: {e}")
    
    def _test_connection(self, session: requests.Session):
        """Test API connection"""
        try:
            self.logger.info("Testing Binance API connection...")
            
            # 서버 시간 확인
            url = f"{self.base_url}/fapi/v1/time"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            server_time = response.json()
            self.logger.info(f"Successfully connected to Binance API - Server time: {server_time['serverTime']}")
            
            # 거래소 정보 확인
            futures_markets = self._filter_futures_markets()
            self.logger.info(f"Found {len(futures_markets)} futures markets")
            
            if len(futures_markets) < 10:
                self.logger.warning("Very few futures markets found - API may have issues")
            
            # 샘플 futures 시장 로깅
            sample_futures = list(futures_markets.items())[:3]
            for symbol, market in sample_futures:
                self.logger.info(f"Sample futures market: {symbol} - Status: {market.get('status')}")
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise
    
    def _filter_futures_markets(self) -> Dict:
        """Filter futures markets using /fapi/v1/exchangeInfo"""
        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            exchange_info = response.json()
            futures_markets = {}
            
            for symbol_info in exchange_info.get('symbols', []):
                symbol = symbol_info.get('symbol', '')
                
                # USDT 페어만 선택
                if symbol.endswith(self.quote_currency):
                    # 활성 상태 확인
                    if (symbol_info.get('status') == 'TRADING' and
                        symbol_info.get('contractType') in ['PERPETUAL']):
                        futures_markets[symbol] = {
                            'symbol': symbol,
                            'status': symbol_info.get('status'),
                            'contractType': symbol_info.get('contractType'),
                            'baseAsset': symbol_info.get('baseAsset'),
                            'quoteAsset': symbol_info.get('quoteAsset')
                        }
            
            return futures_markets
            
        except Exception as e:
            self.logger.error(f"Error filtering futures markets: {e}")
            return {}
    
    def _validate_ticker_data(self, symbol: str, ticker: Dict) -> bool:
        """Validate ticker data quality"""
        try:
            # 기본 데이터 존재 확인
            if not ticker:
                return False
            
            # 거래량 확인 (quoteVolume = USDT 거래량)
            quote_volume = float(ticker.get('quoteVolume', 0))
            if not quote_volume or quote_volume < self.min_volume_usdt:
                self.logger.debug(f"Low volume for {symbol}: {quote_volume}")
                return False
            
            # 가격 데이터 확인
            last_price = float(ticker.get('lastPrice', 0))
            if not last_price or last_price <= 0:
                self.logger.debug(f"Invalid price for {symbol}: {last_price}")
                return False
            
            # 24시간 변화율 확인 (극단적인 값 제외)
            percentage = float(ticker.get('priceChangePercent', 0))
            if abs(percentage) > 50:  # 50% 이상 변동 제외
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
        Get top symbols by 24h volume using direct Binance API calls
        
        Returns:
            List of symbol strings
        """
        try:
            self.logger.info("=== Starting symbol selection process ===")
            
            # 1. Futures 시장 정보 가져오기
            self.logger.info("Loading futures markets...")
            futures_markets = self._filter_futures_markets()
            self.logger.info(f"Found {len(futures_markets)} futures markets")
            
            if len(futures_markets) < 5:
                raise Exception(f"Too few futures markets found: {len(futures_markets)}")
            
            # 2. 24시간 티커 데이터 가져오기
            self.logger.info("Fetching 24hr ticker data...")
            
            # 재시도 로직 추가
            tickers_data = None
            for attempt in range(3):
                try:
                    url = f"{self.base_url}/fapi/v1/ticker/24hr"
                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()
                    tickers_data = response.json()
                    break
                except Exception as e:
                    self.logger.warning(f"Ticker fetch attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise
            
            self.logger.info(f"Fetched ticker data for {len(tickers_data)} symbols")
            
            # 3. 유효한 심볼 선별
            eligible_symbols = []
            
            for ticker in tickers_data:
                try:
                    symbol = ticker.get('symbol', '')
                    
                    # USDT 페어만 선택
                    if not symbol.endswith(self.quote_currency):
                        continue
                    
                    # futures 시장에 있는지 확인
                    if symbol not in futures_markets:
                        continue
                    
                    # 데이터 검증
                    if not self._validate_ticker_data(symbol, ticker):
                        continue
                    
                    # 거래량 추출 (quoteVolume = USDT 거래량)
                    quote_volume = float(ticker.get('quoteVolume', 0))
                    
                    eligible_symbols.append({
                        'symbol': symbol,
                        'volume': quote_volume,
                        'price': float(ticker.get('lastPrice', 0)),
                        'change': float(ticker.get('priceChangePercent', 0))
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error processing ticker for {ticker.get('symbol', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"Found {len(eligible_symbols)} eligible symbols")
            
            # 4. 최소 개수 확인
            if len(eligible_symbols) < self.max_symbols:
                self.logger.warning(f"Only {len(eligible_symbols)} eligible symbols found, expected {self.max_symbols}")
                if len(eligible_symbols) == 0:
                    raise Exception("No eligible symbols found")
            
            # 5. 거래량 기준 정렬
            eligible_symbols.sort(key=lambda x: x['volume'], reverse=True)
            
            # 6. 상위 심볼 선택
            top_symbols = [s['symbol'] for s in eligible_symbols[:self.max_symbols]]
            
            # 7. 결과 로깅
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
        Validate symbols are still tradeable using direct API calls
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols
        """
        try:
            self.logger.info(f"Validating {len(symbols)} symbols...")
            
            # 현재 시장 정보 가져오기
            futures_markets = self._filter_futures_markets()
            
            # 24시간 티커 데이터 가져오기
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            tickers_data = response.json()
            
            # 티커 데이터를 딕셔너리로 변환
            tickers_dict = {ticker['symbol']: ticker for ticker in tickers_data}
            
            valid_symbols = []
            
            for symbol in symbols:
                if symbol in futures_markets and symbol in tickers_dict:
                    ticker = tickers_dict[symbol]
                    
                    # 최소 거래량 확인
                    volume = float(ticker.get('quoteVolume', 0))
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