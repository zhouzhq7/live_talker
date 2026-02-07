"""
MeloTTS Implementation - Fully Open Source TTS
完全开源的多语言 TTS，无需 FFmpeg
"""

import logging
import numpy as np
import tempfile
import os
from typing import Optional
from .base import BaseTTS

logger = logging.getLogger(__name__)


class MeloTTS(BaseTTS):
    """
    MeloTTS - Fully open source multilingual TTS
    
    Features:
    - Fully open source, no FFmpeg required
    - Support for ZH, EN, ES, FR, JP, KR
    - Chinese-English mixed synthesis
    - Speed adjustment
    - Completely offline
    
    Note: First-time use will download model (~300MB)
    """
    
    # Language mapping
    SUPPORTED_LANGUAGES = ["ZH", "EN", "ES", "FR", "JP", "KR"]
    
    def __init__(
        self,
        language: str = "ZH",
        speaker: Optional[str] = None,
        speed: float = 1.0,
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MeloTTS
        
        Args:
            language: Language code (ZH, EN, ES, FR, JP, KR)
            speaker: Speaker ID (None = use default for language)
            speed: Speech speed (0.5 - 2.0)
            model_cache_dir: Model cache directory (not used, for compatibility)
            **kwargs: Additional arguments
        """
        super().__init__(name=f"MeloTTS-{language}", **kwargs)
        
        self.language = language.upper()
        self.speaker = speaker
        self.speed = max(0.5, min(2.0, speed))  # Clamp between 0.5 and 2.0
        self.model_cache_dir = model_cache_dir
        
        self.model = None
        self.speaker_ids = None
        
        # Validate language
        if self.language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"[{self.name}] Language '{language}' not in supported list: {self.SUPPORTED_LANGUAGES}. Using 'ZH'")
            self.language = "ZH"
        
        # Try to load model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load MeloTTS model
        
        Returns:
            True if successful, False otherwise
        """
        if self._is_initialized:
            return True
        
        try:
            from melo.api import TTS
            
            logger.info(f"[{self.name}] Loading MeloTTS model (language: {self.language})...")
            logger.info(f"[{self.name}] First-time download: ~300MB")
            
            # Load model with auto device selection
            self.model = TTS(language=self.language, device='auto')
            
            # Get available speakers
            self.speaker_ids = self.model.hps.data.spk2id
            
            # Set default speaker if not specified or invalid
            if self.speaker is None or self.speaker not in self.speaker_ids:
                self.speaker = list(self.speaker_ids.keys())[0]
                logger.info(f"[{self.name}] Using default speaker: {self.speaker}")
            
            self._is_initialized = True
            logger.info(f"[{self.name}] ✓ Model loaded successfully with speaker: {self.speaker}")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] melotts not installed. Install with: pip install melotts")
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            output_file: Optional file path to save audio
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz, mono)
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return b''
        
        if not text or not text.strip():
            logger.debug(f"[{self.name}] Empty text, returning empty audio")
            return b''
        
        try:
            import torchaudio
            import torch
            
            logger.debug(f"[{self.name}] Synthesizing: {text[:50]}...")
            
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_path = tmp.name
            
            try:
                # Synthesize to file
                self.model.tts_to_file(
                    text,
                    self.speaker_ids[self.speaker],
                    temp_path,
                    speed=self.speed
                )
                
                # Load audio file
                waveform, sample_rate = torchaudio.load(temp_path)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Convert to int16 PCM
                pcm_data = (waveform * 32767).short().numpy().tobytes()
                
                logger.debug(f"[{self.name}] Synthesis complete - PCM size: {len(pcm_data)} bytes")
                
                # Save to file if requested
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(pcm_data)
                    logger.debug(f"[{self.name}] Saved to: {output_file}")
                
                return pcm_data
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    def get_info(self) -> dict:
        """
        Get MeloTTS-specific information
        
        Returns:
            Dictionary with engine information
        """
        info = super().get_info()
        info.update({
            "language": self.language,
            "speaker": self.speaker,
            "speed": self.speed,
            "available_speakers": list(self.speaker_ids.keys()) if self.speaker_ids else [],
            "supported_languages": self.SUPPORTED_LANGUAGES
        })
        return info
    
    @classmethod
    def list_supported_languages(cls) -> list:
        """List supported languages"""
        return cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def list_speakers(cls, language: str = "ZH") -> list:
        """
        List available speakers for a language
        
        Args:
            language: Language code
            
        Returns:
            List of speaker IDs
        """
        try:
            from melo.api import TTS
            model = TTS(language=language.upper(), device='cpu')
            return list(model.hps.data.spk2id.keys())
        except Exception as e:
            logger.error(f"Failed to list speakers: {e}")
            return []
