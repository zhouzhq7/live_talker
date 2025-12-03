"""
ASR (Automatic Speech Recognition) Module
"""

from .base import BaseASR
from .funasr import FunASR
from .whisper import Whisper
from .fireredasr import FireRedASR

__all__ = [
    "BaseASR",
    "FunASR",
    "Whisper",
    "FireRedASR",
]

