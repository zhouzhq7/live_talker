#!/usr/bin/env python3
"""
Full Demo - Complete feature demonstration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TalkerConfig, ASRConfig, TTSConfig, LLMConfig, VADConfig, AudioConfig
from core.talker import LiveTalker
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(level="INFO")


def main():
    """Full demo with custom configuration"""
    print("=" * 70)
    print("Live Talker - Full Demo")
    print("=" * 70)
    print()
    
    # Create custom configuration
    config = TalkerConfig(
        audio=AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024
        ),
        asr=ASRConfig(
            engine="funasr",
            funasr_model="paraformer-zh",
            language="zh"
        ),
        tts=TTSConfig(
            engine="edge",
            edge_voice="zh-CN-XiaoxiaoNeural"
        ),
        llm=LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        ),
        vad=VADConfig(
            method="silero",
            threshold=0.5
        ),
        enable_interruption=True,
        max_conversation_history=10
    )
    
    # Create talker
    talker = LiveTalker(config=config)
    
    # Print configuration info
    print("\nConfiguration:")
    print(f"  ASR Engine: {config.asr.engine}")
    print(f"  TTS Engine: {config.tts.engine}")
    print(f"  LLM Provider: {config.llm.provider}")
    print(f"  VAD Method: {config.vad.method}")
    print(f"  Interruption: {'Enabled' if config.enable_interruption else 'Disabled'}")
    print()
    
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

