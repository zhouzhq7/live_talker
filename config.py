"""
Configuration for Live Talker
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000        # 16kHz standard for speech
    channels: int = 1               # Mono
    chunk_size: int = 1024          # Samples per chunk
    format: int = 8                 # pyaudio.paInt16


@dataclass
class ASRConfig:
    """ASR configuration"""
    engine: str = "funasr"          # funasr, whisper, fireredasr
    
    # FunASR settings
    funasr_model: str = "paraformer-zh"
    funasr_device: str = "cpu"
    funasr_enable_vad: bool = False  # VAD handled separately
    
    # Whisper settings
    whisper_model: str = "base"      # tiny, base, small, medium, large
    whisper_device: str = "cpu"
    
    # FireRedASR settings
    fireredasr_model: str = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
    fireredasr_device: str = "cpu"
    
    # Common settings
    language: str = "zh"            # zh, en, etc.


@dataclass
class TTSConfig:
    """TTS configuration"""
    engine: str = "edge"            # edge, pyttsx3
    
    # Edge-TTS settings
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    edge_rate: str = "+0%"
    edge_volume: str = "+0%"
    
    # Pyttsx3 settings
    pyttsx3_rate: int = 200
    pyttsx3_volume: float = 1.0


@dataclass
class VADConfig:
    """VAD configuration"""
    method: str = "silero"          # silero, webrtc, energy
    threshold: float = 0.5
    min_speech_duration: float = 0.25   # seconds
    min_silence_duration: float = 0.5   # seconds
    sample_rate: int = 16000


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "deepseek"      # deepseek, openai, local
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # Custom API endpoint
    model: str = "deepseek-chat"    # Model name
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = True             # Stream response


@dataclass
class TalkerConfig:
    """Main configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Model cache directory
    model_cache_dir: str = "D:\\models"  # Base directory for all model caches
    
    # Conversation settings
    enable_interruption: bool = True
    max_conversation_history: int = 10  # Keep last N turns
    welcome_message: str = "你好，我是你的AI小助手，有什么可以帮你的吗？"  # Initial greeting message
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


def load_config_from_env() -> TalkerConfig:
    """Load configuration from environment variables"""
    config = TalkerConfig()
    
    # Model cache directory
    config.model_cache_dir = os.getenv("MODEL_CACHE_DIR", config.model_cache_dir)
    
    # ASR engine
    config.asr.engine = os.getenv("ASR_ENGINE", config.asr.engine)
    
    # TTS engine
    config.tts.engine = os.getenv("TTS_ENGINE", config.tts.engine)
    
    # LLM settings
    config.llm.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    config.llm.api_base = os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_API_BASE")
    config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
    config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
    
    return config


# Default configuration
default_config = load_config_from_env()

