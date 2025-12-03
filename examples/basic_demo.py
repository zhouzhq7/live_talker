#!/usr/bin/env python3
"""
Basic Demo - Simple usage example
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config_from_env
from core.talker import LiveTalker
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(level="INFO")


def main():
    """Basic demo"""
    print("=" * 70)
    print("Live Talker - Basic Demo")
    print("=" * 70)
    print()
    
    # Load configuration
    config = load_config_from_env()
    
    # Create talker
    talker = LiveTalker(config=config)
    
    # Start conversation
    try:
        talker.start()
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

