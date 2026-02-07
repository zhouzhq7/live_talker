"""
Pytest fixtures and configuration
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Generator, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir() -> Path:
    """Return test data directory path"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_audio_16k() -> np.ndarray:
    """Generate sample 16kHz audio data (1 second of sine wave)"""
    duration = 1.0  # seconds
    sample_rate = 16000
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    return (audio * 32767).astype(np.int16)


@pytest.fixture
def sample_audio_bytes_16k(sample_audio_16k) -> bytes:
    """Sample audio as bytes"""
    return sample_audio_16k.tobytes()


@pytest.fixture
def sample_silence_16k() -> bytes:
    """Generate 1 second of silence"""
    return np.zeros(16000, dtype=np.int16).tobytes()


@pytest.fixture
def sample_speech_chunks() -> list:
    """Generate sample speech-like audio chunks"""
    chunks = []
    # Simulate speech pattern: speech - silence - speech
    for i in range(5):
        # Speech chunk (sine wave)
        t = np.linspace(0, 0.2, int(16000 * 0.2))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        chunks.append((audio * 32767).astype(np.int16).tobytes())
        
        # Silence chunk
        if i < 4:
            chunks.append(np.zeros(int(16000 * 0.1), dtype=np.int16).tobytes())
    return chunks


@pytest.fixture
def temp_audio_file() -> Generator[str, None, None]:
    """Create a temporary audio file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_asr() -> MagicMock:
    """Mock ASR engine"""
    mock = MagicMock()
    mock.name = "MockASR"
    mock._is_initialized = True
    mock.transcribe.return_value = "你好，这是测试文本。"
    mock.transcribe_with_timing.return_value = {
        "text": "你好，这是测试文本。",
        "latency_ms": 150.0,
        "audio_duration_ms": 1000.0,
        "rtf": 0.15,
        "engine": "MockASR"
    }
    mock.get_info.return_value = {
        "name": "MockASR",
        "initialized": True,
        "warmup_done": True
    }
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_tts() -> MagicMock:
    """Mock TTS engine"""
    mock = MagicMock()
    mock.name = "MockTTS"
    mock._is_initialized = True
    # Return 1 second of fake PCM audio
    mock.synthesize.return_value = np.zeros(16000 * 2, dtype=np.int16).tobytes()
    mock.synthesize_to_file.return_value = True
    mock.get_info.return_value = {
        "name": "MockTTS",
        "initialized": True
    }
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM provider"""
    mock = MagicMock()
    mock.name = "MockLLM"
    mock._is_initialized = True
    mock.generate.return_value = "这是一个测试回复。"
    mock.generate_stream.return_value = iter(["这是", "一个", "测试", "回复。"])
    mock.chat.return_value = "这是一个测试对话回复。"
    mock.get_info.return_value = {
        "name": "MockLLM",
        "initialized": True,
        "history_length": 0
    }
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def mock_vad() -> MagicMock:
    """Mock VAD detector"""
    mock = MagicMock()
    mock.method = "mock"
    mock.threshold = 0.5
    mock.detect.return_value = True
    mock.update_state.return_value = {
        "speech_started": True,
        "speech_ended": False,
        "is_speaking": True
    }
    mock.reset.return_value = None
    return mock


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration"""
    from config import TalkerConfig, AudioConfig, ASRConfig, TTSConfig, VADConfig, LLMConfig
    
    config = TalkerConfig()
    config.model_cache_dir = tempfile.gettempdir()
    config.log_level = "DEBUG"
    return config


@pytest.fixture
def test_asr_config():
    """Test ASR configuration"""
    from config import ASRConfig
    return ASRConfig(
        engine="mock",
        language="zh"
    )


@pytest.fixture
def test_tts_config():
    """Test TTS configuration"""
    from config import TTSConfig
    return TTSConfig(
        engine="mock",
        edge_voice="zh-CN-XiaoxiaoNeural"
    )


# ============================================================================
# Phase 1: ASR Fixtures
# ============================================================================

@pytest.fixture
def sensevoice_config():
    """SenseVoice ASR configuration"""
    from config import ASRConfig
    return ASRConfig(
        engine="sensevoice",
        sensevoice_model="iic/SenseVoiceSmall",
        sensevoice_device="cpu",
        sensevoice_language="auto",
        sensevoice_enable_vad=True
    )


@pytest.fixture
def asr_test_cases():
    """ASR test cases with expected results"""
    return {
        "zh_simple": {
            "description": "中文简单句",
            "expected_keywords": ["你好", "谢谢"],
            "min_accuracy": 0.95
        },
        "zh_complex": {
            "description": "中文复杂句",
            "expected_keywords": ["今天", "天气", "不错"],
            "min_accuracy": 0.90
        },
        "en_simple": {
            "description": "英文简单句",
            "expected_keywords": ["hello", "thank"],
            "min_accuracy": 0.95
        },
        "mixed": {
            "description": "中英混合",
            "expected_keywords": ["AI", "人工智能"],
            "min_accuracy": 0.85
        }
    }


# ============================================================================
# Phase 2: TTS Fixtures
# ============================================================================

@pytest.fixture
def melotts_config():
    """MeloTTS configuration"""
    from config import TTSConfig
    return TTSConfig(
        engine="melotts",
        melotts_language="ZH",
        melotts_speaker="ZH",
        melotts_speed=1.0
    )


@pytest.fixture
def tts_test_texts():
    """TTS test texts"""
    return {
        "zh_short": "你好，世界。",
        "zh_long": "这是一个测试长文本，用于测试TTS引擎处理长文本的能力。需要确保语音合成自然流畅。",
        "en_short": "Hello, world.",
        "en_long": "This is a test sentence for English text to speech synthesis.",
        "mixed": "这是一个test，包含English和中文。",
        "numbers": "今天是2025年2月7日，温度是25度。",
        "special": "测试标点符号：，。！？；："
    }


# ============================================================================
# Phase 3: VAD Fixtures
# ============================================================================

@pytest.fixture
def ten_vad_config():
    """TEN-VAD configuration"""
    from config import VADConfig
    return VADConfig(
        method="ten",
        ten_hop_size=256,
        ten_threshold=0.5
    )


@pytest.fixture
def vad_test_audio():
    """Generate VAD test audio patterns"""
    sample_rate = 16000
    
    # Speech pattern (sine wave)
    t = np.linspace(0, 0.5, int(sample_rate * 0.5))
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
    
    # Silence pattern
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.int16)
    
    return {
        "pure_speech": speech.tobytes(),
        "pure_silence": silence.tobytes(),
        "speech_silence": np.concatenate([speech, silence]).tobytes(),
        "silence_speech": np.concatenate([silence, speech]).tobytes(),
        "alternating": np.concatenate([speech, silence, speech]).tobytes()
    }


# ============================================================================
# Phase 4: LLM Fixtures
# ============================================================================

@pytest.fixture
def ollama_config():
    """Ollama LLM configuration"""
    from config import LLMConfig
    return LLMConfig(
        provider="ollama",
        ollama_model="qwen2.5:7b",
        ollama_host="http://localhost:11434",
        ollama_timeout=30
    )


@pytest.fixture
def llm_test_prompts():
    """LLM test prompts"""
    return {
        "simple": "你好",
        "conversation": "请介绍一下你自己",
        "reasoning": "1+1等于几？",
        "chinese": "请用中文回答",
        "long_context": "请详细解释什么是人工智能。"
    }


# ============================================================================
# Phase 5: Streaming Fixtures
# ============================================================================

@pytest.fixture
def streaming_config():
    """Streaming pipeline configuration"""
    return {
        "asr_streaming": True,
        "llm_streaming": True,
        "tts_streaming": True,
        "buffer_size": 1024,
        "latency_target_ms": 1000
    }


@pytest.fixture
def sentence_test_cases():
    """Sentence splitting test cases"""
    return {
        "simple": {
            "text": "这是第一句。这是第二句。",
            "expected_sentences": ["这是第一句。", "这是第二句。"]
        },
        "with_punctuation": {
            "text": "你好！最近怎么样？我很好。",
            "expected_sentences": ["你好！", "最近怎么样？", "我很好。"]
        },
        "incomplete": {
            "text": "这句话还没说完",
            "expected_sentences": ["这句话还没说完"]
        }
    }


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance test thresholds"""
    return {
        "asr": {
            "rtf_max": 0.1,  # Real-time factor
            "latency_ms_max": 100,  # For 10s audio
            "memory_mb_max": 2048
        },
        "tts": {
            "first_chunk_ms_max": 500,
            "synthesis_10chars_ms_max": 1000,
            "memory_mb_max": 1024
        },
        "vad": {
            "detection_latency_ms_max": 100,
            "rtf_max": 0.01
        },
        "llm": {
            "first_token_ms_max": 500,  # Local LLM
            "tokens_per_sec_min": 10
        },
        "streaming": {
            "end_to_end_latency_ms_max": 1000
        }
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if self.start_time is None:
                return {}
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            self.metrics = {
                "duration_ms": (end_time - self.start_time) * 1000,
                "memory_used_mb": end_memory - self.start_memory,
                "peak_memory_mb": end_memory
            }
            return self.metrics
        
        def get_cpu_percent(self):
            return self.process.cpu_percent(interval=0.1)
    
    return PerformanceMonitor


# ============================================================================
# Skip Conditions
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "network: mark test as requiring network")


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_ollama():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
