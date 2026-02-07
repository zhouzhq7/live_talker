"""
Unit tests for ASR factory and integration
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.phase1
@pytest.mark.unit
@pytest.mark.asr
class TestASRFactory:
    """Test ASR factory pattern"""
    
    def test_create_asr_sensevoice(self):
        """Test creating SenseVoice ASR through factory"""
        with patch('core.talker.SenseVoice') as mock_sensevoice:
            mock_instance = MagicMock()
            mock_sensevoice.return_value = mock_instance
            
            # Import talker after patching
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.asr.engine = "sensevoice"
            talker.config.asr.sensevoice_model = "iic/SenseVoiceSmall"
            talker.config.asr.sensevoice_device = "cpu"
            talker.config.asr.sensevoice_language = "auto"
            talker.config.asr.sensevoice_enable_vad = True
            talker.config.model_cache_dir = "/tmp/models"
            
            result = talker._create_asr()
            
            mock_sensevoice.assert_called_once()
            assert result == mock_instance
    
    def test_create_asr_funasr(self):
        """Test creating FunASR through factory"""
        with patch('core.talker.FunASR') as mock_funasr:
            mock_instance = MagicMock()
            mock_funasr.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.asr.engine = "funasr"
            talker.config.asr.funasr_model = "paraformer-zh"
            talker.config.asr.funasr_device = "cpu"
            talker.config.asr.funasr_enable_vad = False
            talker.config.model_cache_dir = "/tmp/models"
            
            result = talker._create_asr()
            
            mock_funasr.assert_called_once()
            assert result == mock_instance
    
    def test_create_asr_whisper(self):
        """Test creating Whisper ASR through factory"""
        with patch('core.talker.Whisper') as mock_whisper:
            mock_instance = MagicMock()
            mock_whisper.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.asr.engine = "whisper"
            talker.config.asr.whisper_model = "base"
            talker.config.asr.whisper_device = "cpu"
            talker.config.model_cache_dir = "/tmp/models"
            
            result = talker._create_asr()
            
            mock_whisper.assert_called_once()
            assert result == mock_instance
    
    def test_create_asr_invalid_engine(self):
        """Test creating ASR with invalid engine defaults to SenseVoice"""
        with patch('core.talker.SenseVoice') as mock_sensevoice:
            mock_instance = MagicMock()
            mock_sensevoice.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.asr.engine = "invalid"
            talker.config.model_cache_dir = "/tmp/models"
            
            result = talker._create_asr()
            
            # Should fallback to SenseVoice
            mock_sensevoice.assert_called_once()
            assert result == mock_instance


@pytest.mark.phase1
@pytest.mark.unit
@pytest.mark.asr
class TestASRBaseClass:
    """Test BaseASR abstract class"""
    
    def test_base_asr_init(self):
        """Test BaseASR initialization"""
        from asr.base import BaseASR
        
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseASR(name="test")
    
    def test_asr_interface(self):
        """Test ASR interface compliance"""
        from asr.base import BaseASR
        
        # Check required methods
        required_methods = ['load_model', 'transcribe']
        for method in required_methods:
            assert hasattr(BaseASR, method)
            assert callable(getattr(BaseASR, method))
