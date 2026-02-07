"""
VAD (Voice Activity Detection) Module
独立 VAD 模块，支持多种检测引擎
"""

from .base import VADState, VADetector
from .silero import SileroVAD

__all__ = ["VADState", "VADetector", "SileroVAD"]
