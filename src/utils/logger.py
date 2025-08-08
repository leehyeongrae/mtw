"""
Logging configuration module
"""
import logging
import sys
from pathlib import Path
import coloredlogs
from src.utils.config import config

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set log level
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Apply colored logs for console
        coloredlogs.install(
            level=log_level,
            logger=logger,
            fmt=console_format,
            stream=sys.stdout
        )
        
        # File handler if enabled
        if config.log_to_file:
            file_handler = logging.FileHandler(
                LOG_DIR / f"{name}.log",
                encoding='utf-8'
            )
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
    
    return logger