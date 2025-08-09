"""
로깅 유틸리티 (KISS 원칙 적용)
"""
import logging
import sys
from typing import Optional
from src.utils.config import config

# 로거 캐시 (DRY 원칙)
_loggers = {}

def get_logger(name: str) -> logging.Logger:
    """
    로거 인스턴스 반환 (캐싱 적용)
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 로거 인스턴스
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있으면 반환
    if logger.handlers:
        _loggers[name] = logger
        return logger
    
    # 로그 레벨 설정
    log_level = getattr(logging, config.log_level, logging.INFO)
    logger.setLevel(log_level)
    
    # 콘솔 핸들러 설정
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    # 캐시에 저장
    _loggers[name] = logger
    
    return logger