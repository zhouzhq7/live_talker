"""
FireRedASR Implementation
Streaming ASR based on Zipformer architecture
参考 voice_benchmark/asr/fireredasr.py
"""

import logging
import numpy as np
from typing import Optional, List
from .base import BaseASR

logger = logging.getLogger(__name__)


class FireRedASR(BaseASR):
    """
    FireRedASR implementation
    
    Features:
    - Fast streaming ASR
    - Zipformer architecture
    - Low latency
    - Good for real-time applications
    
    Note: This is a wrapper for sherpa-onnx based models
    """
    
    def __init__(
        self,
        model_name: str = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize FireRedASR
        
        Args:
            model_name: Model name or path
            device: Device to run on (cpu, cuda)
            **kwargs: Additional arguments
        """
        super().__init__(name="FireRedASR", **kwargs)
        
        self.model_name = model_name
        self.device = device
        self.recognizer = None
        
        # Try to load model
        self.load_model()
    
    def load_model(self) -> bool:
        """Load FireRedASR model"""
        if self._is_initialized:
            return True
        
        try:
            import sherpa_onnx
            
            logger.info(f"[{self.name}] Loading FireRedASR model...")
            logger.info(f"[{self.name}] Note: FireRedASR requires model files to be downloaded separately")
            logger.info(f"[{self.name}] Download from: https://github.com/k2-fsa/sherpa-onnx")
            logger.info(f"[{self.name}] Model: {self.model_name}")
            
            # Configure recognizer
            # Note: This is a simplified config - adjust based on your model files
            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                feat_config=sherpa_onnx.FeatureExtractorConfig(
                    sampling_rate=16000,
                    feature_dim=80,
                ),
                model_config=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder_filename="encoder.onnx",
                        decoder_filename="decoder.onnx",
                        joiner_filename="joiner.onnx",
                    ),
                    tokens="tokens.txt",
                    num_threads=4,
                    provider="cpu",
                ),
                decoding_method="greedy_search",
                max_active_paths=4,
            )
            
            self.recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
            
            self._is_initialized = True
            logger.info(f"[{self.name}] ✓ Model loaded successfully!")
            return True
            
        except ImportError:
            logger.error(
                f"[{self.name}] sherpa-onnx not installed. "
                "Install with: pip install sherpa-onnx"
            )
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            logger.warning(
                f"[{self.name}] FireRedASR requires model files. "
                "Download from: https://github.com/k2-fsa/sherpa-onnx"
            )
            import traceback
            traceback.print_exc()
            return False
    
    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> str:
        """
        Transcribe audio using FireRedASR
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate (should be 16000)
            language: Language code (currently supports zh)
            
        Returns:
            Transcribed text
        """
        if not self._is_initialized or not self.recognizer:
            logger.error(f"[{self.name}] Model not initialized")
            return ""
        
        if not audio_data or len(audio_data) == 0:
            return ""
        
        try:
            # Convert bytes to float32 numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Create stream
            stream = self.recognizer.create_stream()
            
            # Accept waveform
            stream.accept_waveform(sample_rate, audio_float)
            
            # Decode
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            
            # Get result
            result = self.recognizer.get_result(stream)
            text = result.text.strip()
            
            if text:
                logger.debug(f"[{self.name}] Transcribed: {text[:50]}...")
            
            return text
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def transcribe_streaming(
        self,
        audio_chunks: List[bytes],
        sample_rate: int = 16000
    ) -> List[str]:
        """
        Streaming transcription
        
        Args:
            audio_chunks: List of audio chunks
            sample_rate: Sample rate
            
        Returns:
            List of partial results
        """
        if not self._is_initialized or not self.recognizer:
            logger.error(f"[{self.name}] Model not initialized")
            return []
        
        results = []
        
        try:
            stream = self.recognizer.create_stream()
            
            for chunk in audio_chunks:
                # Convert chunk
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Accept waveform
                stream.accept_waveform(sample_rate, audio_float)
                
                # Decode
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                
                # Get partial result
                result = self.recognizer.get_result(stream)
                if result.text:
                    results.append(result.text)
            
            return results
            
        except Exception as e:
            logger.error(f"[{self.name}] Streaming transcription failed: {e}")
            return results
    
    def get_info(self) -> dict:
        """Get FireRedASR-specific information"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "features": [
                "streaming",
                "low_latency",
                "zipformer_architecture"
            ]
        })
        return info

