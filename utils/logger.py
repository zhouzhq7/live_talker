"""
Logging Configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "live_talker",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name (use None or "" for root logger)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Get root logger to configure all loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Get the specific logger (will inherit from root)
    logger = logging.getLogger(name) if name else root_logger
    logger.setLevel(getattr(logging, level.upper()))
    
    # Ensure propagation is enabled (default is True, but make it explicit)
    logger.propagate = True
    
    return logger

