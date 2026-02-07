"""
Audio utilities for testing
"""

import numpy as np
import io
from typing import Tuple, Optional


def generate_test_audio(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    noise_level: float = 0.0
) -> np.ndarray:
    """
    Generate test audio data
    
    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of sine wave
        amplitude: Amplitude (0.0 to 1.0)
        noise_level: Noise level (0.0 to 1.0)
    
    Returns:
        Audio array as int16
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t) * amplitude
    
    # Add noise if requested
    if noise_level > 0:
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise
    
    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)
    
    # Convert to int16
    return (audio * 32767).astype(np.int16)


def generate_speech_like_audio(
    duration: float = 1.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Generate speech-like audio (modulated sine wave)
    
    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Audio array as int16
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency (vowel-like)
    base_freq = 150  # F0 for male voice
    
    # Modulation to simulate speech
    modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)  # 5Hz modulation
    
    # Harmonics
    audio = (
        np.sin(2 * np.pi * base_freq * t) * 0.5 +
        np.sin(2 * np.pi * base_freq * 2 * t) * 0.25 +
        np.sin(2 * np.pi * base_freq * 3 * t) * 0.125
    ) * modulation
    
    # Add some formant-like resonances
    formant = np.sin(2 * np.pi * 800 * t) * 0.1  # First formant
    audio = audio + formant
    
    # Apply envelope
    attack = int(0.05 * sample_rate)  # 50ms attack
    release = int(0.05 * sample_rate)  # 50ms release
    
    envelope = np.ones_like(audio)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    audio = audio * envelope
    
    return (audio * 32767).astype(np.int16)


def calculate_snr(audio: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        audio: Audio array
    
    Returns:
        SNR in dB
    """
    signal_power = np.mean(audio.astype(np.float64) ** 2)
    noise_power = np.var(audio.astype(np.float64))
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) of audio
    
    Args:
        audio: Audio array
    
    Returns:
        RMS value (0.0 to 1.0)
    """
    return np.sqrt(np.mean((audio / 32768.0) ** 2))


def compare_audio_quality(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """
    Compare two audio signals for quality metrics
    
    Args:
        audio1: Reference audio
        audio2: Test audio
        sample_rate: Sample rate
    
    Returns:
        Dictionary with quality metrics
    """
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Normalize
    audio1_float = audio1.astype(np.float64) / 32768.0
    audio2_float = audio2.astype(np.float64) / 32768.0
    
    # Calculate correlation
    correlation = np.corrcoef(audio1_float, audio2_float)[0, 1]
    
    # Calculate MSE
    mse = np.mean((audio1_float - audio2_float) ** 2)
    
    # Calculate SNR
    noise = audio1_float - audio2_float
    signal_power = np.mean(audio1_float ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return {
        "correlation": correlation,
        "mse": mse,
        "snr_db": snr,
        "similarity": max(0, correlation)  # Normalized similarity
    }


def create_wav_header(
    data_size: int,
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """
    Create WAV file header
    
    Args:
        data_size: Size of audio data in bytes
        sample_rate: Sample rate
        channels: Number of channels
        bits_per_sample: Bits per sample
    
    Returns:
        WAV header bytes
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = bytearray()
    
    # RIFF chunk
    header.extend(b'RIFF')
    header.extend((36 + data_size).to_bytes(4, 'little'))  # Chunk size
    header.extend(b'WAVE')
    
    # fmt sub-chunk
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # Sub-chunk size (16 for PCM)
    header.extend((1).to_bytes(2, 'little'))   # Audio format (1 = PCM)
    header.extend(channels.to_bytes(2, 'little'))
    header.extend(sample_rate.to_bytes(4, 'little'))
    header.extend(byte_rate.to_bytes(4, 'little'))
    header.extend(block_align.to_bytes(2, 'little'))
    header.extend(bits_per_sample.to_bytes(2, 'little'))
    
    # data sub-chunk
    header.extend(b'data')
    header.extend(data_size.to_bytes(4, 'little'))
    
    return bytes(header)


def create_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> bytes:
    """
    Create WAV file bytes from audio array
    
    Args:
        audio: Audio array (int16)
        sample_rate: Sample rate
    
    Returns:
        WAV file as bytes
    """
    data_bytes = audio.tobytes()
    header = create_wav_header(len(data_bytes), sample_rate)
    return header + data_bytes
