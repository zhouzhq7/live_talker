#!/usr/bin/env python3
"""
Live Talker - Main Entry Point
实时语音对话系统主程序
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure root logger FIRST, before importing any other modules
# This ensures all child loggers inherit the configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
if root_logger.handlers:
    root_logger.handlers.clear()

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from .env")
except ImportError:
    pass  # python-dotenv not installed, skip

from config import load_config_from_env
from utils.logger import setup_logger

# Get logger for this module
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    print("=" * 70)
    print("Live Talker - 实时语音对话系统")
    print("=" * 70)
    print()
    
    # Load configuration
    config = load_config_from_env()
    
    # Reconfigure logger with config settings if needed
    if config.log_level:
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Add file handler if specified
        if config.log_file:
            from pathlib import Path
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info(f"Logger configured with level: {config.log_level}")
    
    try:
        from core.talker import LiveTalker
        
        # Initialize talker
        print("初始化 Live Talker...")
        talker = LiveTalker(config=config)
        
        print("\n" + "=" * 70)
        print("系统就绪！")
        print("=" * 70)
        print("\n使用说明:")
        print("  • 自然说话，系统会自动检测语音")
        print("  • 可以在系统回复时打断")
        print("  • 按 Ctrl+C 退出")
        print("\n" + "-" * 70)
        print()
        
        # Start conversation
        talker.start()
        
    except KeyboardInterrupt:
        print("\n\n正在退出...")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

