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
    engine: str = "sensevoice"      # sensevoice, funasr, whisper, fireredasr
    
    # SenseVoice settings (default)
    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cpu"
    sensevoice_language: str = "auto"  # auto, zh, en, yue, ja, ko, nospeech
    sensevoice_enable_vad: bool = True
    
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
    engine: str = "melotts"         # melotts (default), edge, pyttsx3
    
    # MeloTTS settings (default)
    melotts_language: str = "ZH"    # ZH, EN, ES, FR, JP, KR
    melotts_speaker: str = "ZH"     # Speaker ID
    melotts_speed: float = 1.0      # Speed: 0.5 - 2.0
    
    # Edge-TTS settings (optional)
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    edge_rate: str = "+0%"
    edge_volume: str = "+0%"
    
    # Pyttsx3 settings
    pyttsx3_rate: int = 200
    pyttsx3_volume: float = 1.0


@dataclass
class VADConfig:
    """VAD configuration"""
    method: str = "ten"             # ten (default), silero, webrtc, energy
    
    # TEN-VAD settings (default, 306KB, 32% faster)
    ten_hop_size: int = 256         # 16ms at 16kHz
    ten_threshold: float = 0.5
    
    # Silero settings (alternative)
    silero_threshold: float = 0.5
    
    # WebRTC settings
    webrtc_aggressiveness: int = 3  # 0-3
    
    # Common settings
    threshold: float = 0.5
    min_speech_duration: float = 0.25   # seconds
    min_silence_duration: float = 0.5   # seconds
    sample_rate: int = 16000


@dataclass
class LLMConfig:
    """LLM configuration - Multi-provider support with automatic failover"""
    
    # Provider selection
    provider: str = "deepseek"      # Primary provider: deepseek, zhipu, openai, moonshot
    enable_fallback: bool = True    # Enable automatic failover
    fallback_providers: list = field(default_factory=lambda: ["zhipu", "openai"])
    
    # Common settings
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = True
    system_prompt: str = "You are a helpful assistant."
    max_history: int = 10
    
    # DeepSeek settings
    deepseek_api_key: Optional[str] = None
    deepseek_api_base: Optional[str] = None
    deepseek_model: str = "deepseek-chat"  # deepseek-chat, deepseek-coder
    
    # Zhipu (智谱) settings
    zhipu_api_key: Optional[str] = None
    zhipu_api_base: Optional[str] = None
    zhipu_model: str = "glm-4-flash"  # glm-4-flash (free), glm-4-air, glm-4
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "gpt-4o-mini"  # gpt-4o-mini, gpt-4o, gpt-3.5-turbo
    
    # Moonshot (Kimi) settings
    moonshot_api_key: Optional[str] = None
    moonshot_api_base: Optional[str] = None
    moonshot_model: str = "moonshot-v1-8k"  # moonshot-v1-8k, 32k, 128k
    
    # Legacy compatibility (deprecated, use provider-specific settings)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "deepseek-chat"


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
    
    # LLM settings - Multi-provider
    config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
    config.llm.enable_fallback = os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true"
    
    # DeepSeek
    config.llm.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    config.llm.deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
    config.llm.deepseek_model = os.getenv("DEEPSEEK_MODEL", config.llm.deepseek_model)
    
    # Zhipu
    config.llm.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    config.llm.zhipu_api_base = os.getenv("ZHIPU_API_BASE")
    config.llm.zhipu_model = os.getenv("ZHIPU_MODEL", config.llm.zhipu_model)
    
    # OpenAI
    config.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.llm.openai_api_base = os.getenv("OPENAI_API_BASE")
    config.llm.openai_model = os.getenv("OPENAI_MODEL", config.llm.openai_model)
    
    # Moonshot
    config.llm.moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
    config.llm.moonshot_api_base = os.getenv("MOONSHOT_API_BASE")
    config.llm.moonshot_model = os.getenv("MOONSHOT_MODEL", config.llm.moonshot_model)
    
    # Legacy compatibility
    config.llm.api_key = config.llm.deepseek_api_key or config.llm.openai_api_key
    config.llm.api_base = config.llm.deepseek_api_base or config.llm.openai_api_base
    config.llm.model = config.llm.deepseek_model or config.llm.openai_model
    
    return config


# Default configuration
default_config = load_config_from_env()

