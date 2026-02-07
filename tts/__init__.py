"""
TTS (Text-to-Speech) Module
"""

from .base import BaseTTS
from .melotts import MeloTTS
from .edge_tts import EdgeTTS
from .pyttsx3_tts import Pyttsx3TTS

__all__ = [
    "BaseTTS",
    "MeloTTS",
    "EdgeTTS",
    "Pyttsx3TTS",
]

