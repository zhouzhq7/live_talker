"""
Audio Processing Module
"""

from .sensor import AudioSensor
from .recorder import RealtimeRecorder
from .vad import VADDetector, InterruptionDetector
from .player import AudioPlayer
from .collector import AudioCollector, AudioConfig, SounddeviceCollector, create_collector
from .buffer import RingBuffer, AudioBuffer
from .aec import AECProcessor, WebRTCAEC, InterruptionDetector as AudioInterruptionDetector

__all__ = [
    "AudioSensor",
    "RealtimeRecorder",
    "VADDetector",
    "InterruptionDetector",
    "AudioPlayer",
    "AudioCollector",
    "AudioConfig",
    "SounddeviceCollector",
    "create_collector",
    "RingBuffer",
    "AudioBuffer",
    "AECProcessor",
    "WebRTCAEC",
    "AudioInterruptionDetector",
]

