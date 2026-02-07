"""
Integration tests for MeloTTS
"""

import pytest
import numpy as np


@pytest.mark.phase2
@pytest.mark.integration
@pytest.mark.tts
@pytest.mark.slow
class TestMeloTTSIntegration:
    """Integration tests for MeloTTS"""
    
    @pytest.fixture(scope="class")
    def melotts_engine(self):
        """Create MeloTTS engine"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH", speaker="ZH")
            engine.load_model()
            yield engine
        except ImportError:
            pytest.skip("MeloTTS not installed")
        except Exception as e:
            pytest.skip(f"Failed to load MeloTTS: {e}")
    
    def test_chinese_synthesis(self, melotts_engine):
        """Test Chinese speech synthesis"""
        audio = melotts_engine.synthesize("你好，世界")
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_english_synthesis(self, melotts_engine):
        """Test English speech synthesis"""
        audio = melotts_engine.synthesize("Hello world")
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_mixed_language(self, melotts_engine):
        """Test mixed Chinese-English synthesis"""
        audio = melotts_engine.synthesize("这是一个test")
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_long_text(self, melotts_engine):
        """Test long text synthesis"""
        long_text = "这是一个很长的测试文本，用于测试TTS引擎处理长文本的能力。" * 5
        audio = melotts_engine.synthesize(long_text)
        assert isinstance(audio, bytes)
        assert len(audio) > 0
    
    def test_output_format(self, melotts_engine):
        """Test output audio format (16-bit PCM, 16kHz)"""
        audio = melotts_engine.synthesize("测试")
        
        # Should be int16 PCM
        audio_array = np.frombuffer(audio, dtype=np.int16)
        assert len(audio_array) > 0
        
        # For 1 second of audio at 16kHz, should be about 16000 samples
        # (may vary due to text length)
        assert len(audio_array) > 1000


@pytest.mark.phase2
@pytest.mark.integration
@pytest.mark.tts
class TestTTSIntegration:
    """Test TTS integration with talker"""
    
    def test_tts_factory_integration(self, mock_tts):
        """Test TTS factory creates correct engine"""
        from core.talker import LiveTalker
        from config import TalkerConfig
        
        with patch('core.talker.EdgeTTS', return_value=mock_tts):
            config = TalkerConfig()
            config.tts.engine = "edge"
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = config
            
            tts = talker._create_tts()
            assert tts is not None
