# Binance Trading Bot

## Overview
Automated cryptocurrency trading bot for Binance Futures with advanced technical analysis, multiprocessing support, and risk management.

## Features
- **Multi-Symbol Trading**: Trades top symbols by 24h volume (up to 20 symbols)
- **Advanced Technical Analysis**: 
  - Trend Detection: ADX, Hurst Exponent with slope-based analysis
  - Trend Following: Parabolic SAR, SuperTrend, Vortex Indicator
  - Oscillators: RSI, CCI, MFI with normalization
  - Volatility: ATR for dynamic stop-loss
- **Market Adaptation**: Automatically switches between trending and ranging strategies
- **Risk Management**:
  - Fixed take-profit (default 5%)
  - Trailing stop-loss with ATR-based adjustment
  - Position cooldown after losses
  - Maximum position limits
- **Real-time Data Management**:
  - WebSocket for live price updates
  - REST API for historical data
  - Intelligent candle validation and synchronization
- **Multiprocessing Architecture**: Dedicated process per symbol for indicator calculation
- **Telegram Integration**: Monitor and control via Telegram bot
- **Docker Support**: Easy deployment with Docker

## System Architecture

### Data Flow
1. **Symbol Manager**: Selects top symbols by volume
2. **WebSocket Client**: Receives real-time tick data (centralized for all symbols)
3. **REST Client**: Fetches historical candles with 3-second rate limiting
4. **Data Manager**: Central shared memory for multiprocessing coordination
5. **Indicator Processors**: One process per symbol for technical analysis
6. **Signal Generator**: Creates entry/exit signals based on market conditions
7. **Trading Engine**: Executes orders with risk management
8. **Position Manager**: Tracks positions and trailing stops

### Strategy Logic

#### Trend Detection (8-candle slope analysis)
- **Trending Market**: ADX slope > 0 AND Hurst slope > 0
- **Ranging Market**: Otherwise

#### Trending Market Strategy
- **Long Entry**: PSAR uptrend + SuperTrend uptrend
- **Short Entry**: PSAR downtrend + SuperTrend downtrend
- **Exit Conditions**:
  - Trend disagreement with VI confirmation
  - Both indicators reverse
  - Take profit or trailing stop hit

#### Ranging Market Strategy
- **Long Entry**: Oscillator score ≤ -1.0 (oversold)
- **Short Entry**: Oscillator score ≥ 1.0 (overbought)
- **Exit**: Opposite extreme reached

## Installation

### Prerequisites
- Docker and Docker Compose
- Binance API credentials
- Telegram Bot Token (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/binance-trading-bot.git
cd binance-trading-bot
```

2. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials and preferences
```

3. Build and run with Docker:
```bash
docker-compose up --build
```

## Configuration

### Essential Settings (.env)
```env
# API Keys
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Parameters
LEVERAGE=10
SYMBOL_COUNT=5
TAKE_PROFIT_PERCENT=5.0

# Risk Management
STOP_TRAILING_ATR_MULTIPLIER=2.0
STOP_TRAILING_MIN_PERCENT=1.0
STOP_TRAILING_MAX_PERCENT=10.0
```

### Indicator Parameters
All technical indicators can be customized via environment variables. See `.env.example` for full list.

## Telegram Commands
- `/stop` - Stop the trading system
- `/balance` - Show account balance and PnL
- `/status` - Display all symbols with indicators and positions
- `/help` - Show available commands

## Console Output
The system displays comprehensive status every 30 seconds:
- Last 5 candles (4 closed + 1 current) for each symbol
- All calculated indicators
- Market type (trending/ranging)
- Current positions with PnL
- Latest signals

## Safety Features
- **Position Limits**: Maximum concurrent positions
- **Cooldown Periods**: Prevents immediate re-entry after losses
- **Candle Validation**: Ensures data integrity with 15-minute interval checks
- **Exchange Sync**: Periodic position synchronization with exchange
- **Graceful Shutdown**: Proper cleanup on stop signals

## Technical Details

### Multiprocessing Design
- **Main Process**: Coordination and data management
- **WebSocket Thread**: Single connection for all symbols
- **REST API Manager**: Sequential requests with 3-second intervals
- **Indicator Processes**: One per symbol (max 20)
- **Shared Memory**: Manager-based dictionaries and queues

### Hurst Exponent Optimization
- Calculated only on closed candles
- Results cached between candles
- Complex calculation isolated to reduce CPU load

### Data Integrity
- REST API fetches completed candles after 10-second delay
- WebSocket provides real-time updates for current candle
- Automatic full refresh if interval validation fails
- Last 50 candles checked for proper 15-minute spacing

## Performance Considerations
- Maximum 20 symbols to balance opportunity and system load
- 3-second REST API interval to avoid rate limits
- Efficient numpy-based indicator calculations
- Process pooling for parallel computation

## Monitoring
- Comprehensive logging with configurable levels
- Telegram notifications for important events
- Real-time console display
- Position tracking with PnL calculations

## Troubleshooting

### Common Issues
1. **WebSocket Disconnection**: Automatic reconnection with 5-second delay
2. **REST API Rate Limits**: Fixed 3-second interval prevents bans
3. **Missing Candles**: Automatic full refresh triggered
4. **Orphaned Positions**: Periodic sync with exchange

### Log Files
Logs are stored in the `logs/` directory:
- `main.log` - System coordination
- `binance_rest.log` - REST API operations
- `binance_ws.log` - WebSocket events
- `indicator_[SYMBOL].log` - Per-symbol calculations
- `trading_engine.log` - Trade execution

## Development

### Project Structure
```
binance-trading-bot/
├── src/
│   ├── core/           # Data and position management
│   ├── exchange/       # Binance API clients
│   ├── indicators/     # Technical analysis (provided)
│   ├── strategy/       # Trading logic and risk
│   ├── telegram/       # Bot interface
│   └── utils/          # Configuration and logging
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── main.py
└── .env
```

### Adding New Indicators
1. Implement calculation in `src/indicators/indicators.py`
2. Add parameters to `Config` class
3. Update `calculate_all()` method
4. Modify signal generation logic if needed

## Disclaimer
**USE AT YOUR OWN RISK**

This bot is for educational purposes. Cryptocurrency trading carries significant risk. Always test thoroughly with small amounts before increasing position sizes. The authors are not responsible for any financial losses.

## License
MIT License - See LICENSE file for details

## Support
For issues and questions, please open a GitHub issue or contact via Telegram.