# Python 3.11 slim 이미지 사용 (KISS 원칙)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY main.py .

# 환경 변수 파일 복사 (빌드 시 .env 파일 필요)
COPY .env .

# 포트 노출 (필요한 경우)
# EXPOSE 8080

# 실행 명령
CMD ["python", "-u", "main.py"]