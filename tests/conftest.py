"""
Pytest Configuration
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root():
    """Project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_audio_1s():
    """Generate 1 second of silent 16kHz 16-bit PCM audio"""
    import numpy as np
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples, dtype=np.int16)
    return audio.tobytes()


@pytest.fixture
def sample_audio_100ms():
    """Generate 100ms of silent 16kHz 16-bit PCM audio"""
    import numpy as np
    sample_rate = 16000
    duration = 0.1
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples, dtype=np.int16)
    return audio.tobytes()


@pytest.fixture
def sample_audio_speech():
    """Generate simulated speech audio (with some energy)"""
    import numpy as np
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    # Simulated speech with varying amplitude
    t = np.linspace(0, duration, num_samples)
    audio = (0.5 * np.sin(2 * np.pi * 200 * t) +
             0.3 * np.sin(2 * np.pi * 400 * t) +
             0.2 * np.sin(2 * np.pi * 600 * t)) * 32767
    audio = audio.astype(np.int16)
    return audio.tobytes()


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "threshold": 0.5,
        "min_speech_duration": 0.25,
        "min_silence_duration": 0.5,
        "sample_rate": 16000,
    }
