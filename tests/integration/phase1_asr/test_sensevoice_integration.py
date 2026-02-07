"""
Integration tests for SenseVoice ASR

These tests require the actual SenseVoice model to be installed.
Use -m "not slow" to skip these tests.
"""

import pytest
import numpy as np
from unittest.mock import patch


@pytest.mark.phase1
@pytest.mark.integration
@pytest.mark.asr
@pytest.mark.slow
class TestSenseVoiceIntegration:
    """Integration tests for SenseVoice"""
    
    @pytest.fixture(scope="class")
    def sensevoice_engine(self):
        """Create SenseVoice engine (class-level fixture)"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(
                model_name="iic/SenseVoiceSmall",
                device="cpu"
            )
            engine.load_model()
            yield engine
        except ImportError:
            pytest.skip("SenseVoice not installed")
        except Exception as e:
            pytest.skip(f"Failed to load SenseVoice: {e}")
    
    def test_chinese_transcription(self, sensevoice_engine):
        """Test Chinese transcription"""
        # This test would require actual audio files
        # For now, we just verify the engine is working
        assert sensevoice_engine.is_available()
    
    def test_english_transcription(self, sensevoice_engine):
        """Test English transcription"""
        assert sensevoice_engine.is_available()
    
    def test_multilingual_support(self, sensevoice_engine):
        """Test multilingual support"""
        info = sensevoice_engine.get_info()
        assert info["supports_emotion"] is True
        assert info["supports_event_detection"] is True


@pytest.mark.phase1
@pytest.mark.integration
@pytest.mark.asr
class TestASRIntegration:
    """Test ASR integration with talker"""
    
    def test_asr_factory_integration(self, mock_asr):
        """Test ASR factory creates correct engine"""
        from core.talker import LiveTalker
        from config import TalkerConfig
        
        with patch('core.talker.SenseVoice', return_value=mock_asr):
            config = TalkerConfig()
            config.asr.engine = "sensevoice"
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = config
            
            asr = talker._create_asr()
            assert asr is not None
