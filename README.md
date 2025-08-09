# Binance Trading Bot

## 개요
바이낸스 선물 거래를 위한 자동 트레이딩 봇입니다. WebSocket과 REST API를 하이브리드로 사용하여 실시간 데이터 처리와 안정적인 캔들 관리를 구현했습니다.

## 주요 기능
- **하이브리드 데이터 관리**: REST API (과거 데이터) + WebSocket (실시간 데이터)
- **다중 심볼 지원**: 최대 20개 심볼 동시 거래
- **추세/횡보 판별**: ADX와 Hurst 지수 기반 시장 분석
- **리스크 관리**: Trailing Stop Loss와 고정 Take Profit
- **텔레그램 봇**: 실시간 알림 및 제어

## 설계 원칙
- **DRY (Don't Repeat Yourself)**: 코드 재사용성 극대화
- **KISS (Keep It Simple, Stupid)**: 단순하고 명확한 구조
- **YAGNI (You Aren't Gonna Need It)**: 필요한 기능만 구현

## 시스템 구조
```
trading-bot/
├── src/
│   ├── core/           # 핵심 관리 모듈
│   ├── trading/        # 거래 전략 모듈
│   ├── utils/          # 유틸리티
│   └── telegram/       # 텔레그램 봇
├── main.py             # 메인 실행 파일
├── .env                # 환경 변수
├── requirements.txt    # Python 패키지
├── Dockerfile         
└── docker-compose.yml
```

## 설치 및 실행

### 1. 환경 설정
```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
nano .env
```

### 2. Docker로 실행
```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 3. 로컬 실행
```bash
# 가상환경 생성
python -m venv venv

# 활성화
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt

# 실행
python main.py
```

## 텔레그램 명령어
- `/stop` - 봇 중지
- `/balance` - 계정 잔고 확인
- `/status` - 전체 심볼 상태
- `/status [SYMBOL]` - 특정 심볼 상세 정보
- `/help` - 도움말

## 거래 전략

### 추세장 (Trending Market)
- **진입 조건**: PSAR와 Supertrend 방향 일치
- **청산 조건**: 
  - 지표 반전 (PSAR ≠ Supertrend + VI 크로스)
  - 두 지표 모두 반대 방향

### 횡보장 (Ranging Market)
- **진입 조건**: 오실레이터 과매도/과매수
- **청산 조건**: 오실레이터 반대 신호

## 리스크 관리
- **레버리지**: 설정 가능 (기본 5배)
- **포지션 크기**: 계정의 10%
- **Stop Loss**: ATR 기반 Trailing Stop
- **Take Profit**: 고정 % (기본 5%)
- **쿨다운**: 거래 후 5분, 손절 시 60분

## 주요 설정값
```env
# 지표 파라미터
ADX_LENGTH=24           # ADX 기간
VI_LENGTH=48            # Vortex Indicator 기간
RSI_LENGTH=24           # RSI 기간
SUPERTREND_MULTIPLIER=5.0  # Supertrend ATR 배수

# 오실레이터 가중치 (합계 1.0)
WEIGHT_RSI=0.4
WEIGHT_CCI=0.3
WEIGHT_MFI=0.3

# 시장 판별
TREND_DETECTION_CANDLES=8  # 기울기 계산 캔들 수

# 리스크 관리
STOP_TRAILING_MULTIPLIER=2.0  # ATR 배수
TAKE_PROFIT_PERCENT=5.0        # 목표 수익률
```

## 주의사항
- 실제 거래 전 테스트넷에서 충분히 테스트
- API 키 권한은 필요한 것만 부여
- 충분한 자본과 리스크 관리 필수
- 24시간 모니터링 권장

## 라이센스
Private - 상업적 사용 금지