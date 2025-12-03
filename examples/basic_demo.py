#!/usr/bin/env python3
"""
Basic Demo - Simple usage example
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure root logger FIRST
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.handlers:
    root_logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

from config import load_config_from_env
from core.talker import LiveTalker

# Get logger for this module
logger = logging.getLogger(__name__)


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

