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

# Setup logging
logger = setup_logger(level="INFO")


def main():
    """Main entry point"""
    print("=" * 70)
    print("Live Talker - 实时语音对话系统")
    print("=" * 70)
    print()
    
    # Load configuration
    config = load_config_from_env()
    
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

