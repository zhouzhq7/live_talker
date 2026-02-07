"""
Unit tests for TTS factory
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestTTSFactory:
    """Test TTS factory pattern"""
    
    def test_create_tts_melotts(self):
        """Test creating MeloTTS through factory"""
        with patch('core.talker.MeloTTS') as mock_melotts:
            mock_instance = MagicMock()
            mock_melotts.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.tts.engine = "melotts"
            talker.config.tts.melotts_language = "ZH"
            talker.config.tts.melotts_speaker = "ZH"
            talker.config.tts.melotts_speed = 1.0
            
            result = talker._create_tts()
            
            mock_melotts.assert_called_once()
            assert result == mock_instance
    
    def test_create_tts_edge(self):
        """Test creating Edge-TTS through factory"""
        with patch('core.talker.EdgeTTS') as mock_edge:
            mock_instance = MagicMock()
            mock_edge.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.tts.engine = "edge"
            talker.config.tts.edge_voice = "zh-CN-XiaoxiaoNeural"
            
            result = talker._create_tts()
            
            mock_edge.assert_called_once()
            assert result == mock_instance
    
    def test_create_tts_pyttsx3(self):
        """Test creating Pyttsx3 TTS through factory"""
        with patch('core.talker.Pyttsx3TTS') as mock_pyttsx3:
            mock_instance = MagicMock()
            mock_pyttsx3.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.tts.engine = "pyttsx3"
            
            result = talker._create_tts()
            
            mock_pyttsx3.assert_called_once()
            assert result == mock_instance
    
    def test_create_tts_invalid_engine(self):
        """Test creating TTS with invalid engine defaults to MeloTTS"""
        with patch('core.talker.MeloTTS') as mock_melotts:
            mock_instance = MagicMock()
            mock_melotts.return_value = mock_instance
            
            from core.talker import LiveTalker
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = MagicMock()
            talker.config.tts.engine = "invalid"
            
            result = talker._create_tts()
            
            # Should fallback to MeloTTS
            mock_melotts.assert_called_once()
            assert result == mock_instance


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestTTSBaseClass:
    """Test BaseTTS abstract class"""
    
    def test_base_tts_init(self):
        """Test BaseTTS initialization"""
        from tts.base import BaseTTS
        
        with pytest.raises(TypeError):
            BaseTTS(name="test")
    
    def test_tts_interface(self):
        """Test TTS interface compliance"""
        from tts.base import BaseTTS
        
        required_methods = ['synthesize']
        for method in required_methods:
            assert hasattr(BaseTTS, method)
            assert callable(getattr(BaseTTS, method))
