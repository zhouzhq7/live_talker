"""
Unit tests for SenseVoice ASR
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open


@pytest.mark.phase1
@pytest.mark.unit
@pytest.mark.asr
class TestSenseVoice:
    """Test SenseVoice ASR implementation"""
    
    @pytest.fixture
    def sensevoice_class(self):
        """Import SenseVoice class"""
        try:
            from asr.sensevoice import SenseVoice
            return SenseVoice
        except ImportError:
            pytest.skip("SenseVoice not implemented yet")
    
    @pytest.fixture
    def mock_funasr(self):
        """Mock FunASR AutoModel"""
        with patch('funasr.AutoModel') as mock:
            model_instance = MagicMock()
            model_instance.generate.return_value = [{"text": "你好世界"}]
            mock.return_value = model_instance
            yield mock
    
    def test_sensevoice_init(self, sensevoice_class, mock_funasr):
        """Test SenseVoice initialization"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(
                model_name="iic/SenseVoiceSmall",
                device="cpu"
            )
            assert asr.name == "SenseVoice"
            assert asr.model_name == "iic/SenseVoiceSmall"
            assert asr.device == "cpu"
            # When mocked, model loads successfully during init
            assert asr._is_initialized is True
    
    def test_sensevoice_init_with_cache_dir(self, sensevoice_class, mock_funasr):
        """Test initialization with custom cache directory"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(
                model_name="iic/SenseVoiceSmall",
                device="cpu",
                model_cache_dir="/tmp/models"
            )
            assert asr.model_cache_dir == "/tmp/models"
    
    def test_load_model_success(self, sensevoice_class, mock_funasr):
        """Test successful model loading"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            result = asr.load_model()
            assert result is True
            assert asr._is_initialized is True
            mock_funasr.assert_called_once()
    
    def test_load_model_failure(self, sensevoice_class):
        """Test model loading failure"""
        with patch('funasr.AutoModel', side_effect=Exception("Model not found")):
            asr = sensevoice_class(device="cpu")
            result = asr.load_model()
            assert result is False
            assert asr._is_initialized is False
    
    def test_transcribe_success(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test successful transcription"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            text = asr.transcribe(sample_audio_bytes_16k, sample_rate=16000)
            assert isinstance(text, str)
    
    def test_transcribe_not_initialized(self, sensevoice_class, sample_audio_bytes_16k):
        """Test transcription when not initialized"""
        with patch.dict('os.environ', {}, clear=True):
            with patch('funasr.AutoModel', side_effect=Exception("Model not found")):
                asr = sensevoice_class(device="cpu")
                # Model failed to load
                assert asr._is_initialized is False
                text = asr.transcribe(sample_audio_bytes_16k)
                assert text == ""
    
    def test_transcribe_with_language(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test transcription with specific language"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            text = asr.transcribe(
                sample_audio_bytes_16k,
                sample_rate=16000,
                language="zh"
            )
            assert isinstance(text, str)
    
    def test_clean_emotion_tags(self, sensevoice_class, mock_funasr):
        """Test emotion tag cleaning"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            
            # Test different emotion tags
            test_cases = [
                ("<|EMO_UNKNOWN|><|Event_unknow|>你好", "你好"),
                ("<|HAPPY|><|Speech|>开心", "开心"),
                ("<|SAD|><|Applause|>难过", "难过"),
                ("正常文本", "正常文本"),
                ("<|ANGRY|>", ""),
            ]
            
            for input_text, expected in test_cases:
                result = asr._clean_emotion_tags(input_text)
                assert result == expected, f"Failed for input: {input_text}"
    
    def test_transcribe_with_emotion(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test transcription with emotion tags"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            # Mock result with emotion tags
            mock_funasr.return_value.generate.return_value = [{
                "text": "<|HAPPY|><|Speech|>你好世界"
            }]
            
            text = asr.transcribe(sample_audio_bytes_16k)
            assert "<|" not in text  # Tags should be cleaned
            assert "你好" in text
    
    def test_get_info(self, sensevoice_class, mock_funasr):
        """Test get_info method"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(
                model_name="iic/SenseVoiceSmall",
                device="cpu"
            )
            asr.load_model()
            
            info = asr.get_info()
            assert info["name"] == "SenseVoice"
            assert info["model_name"] == "iic/SenseVoiceSmall"
            assert info["device"] == "cpu"
            assert info["supports_emotion"] is True
            assert info["supports_event_detection"] is True
            assert info["initialized"] is True
    
    def test_transcribe_with_timing(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test transcribe_with_timing method"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            result = asr.transcribe_with_timing(
                sample_audio_bytes_16k,
                sample_rate=16000
            )
            
            assert "text" in result
            assert "latency_ms" in result
            assert "audio_duration_ms" in result
            assert "rtf" in result
            assert "engine" in result
            assert result["engine"] == "SenseVoice"
    
    def test_transcribe_empty_audio(self, sensevoice_class, mock_funasr):
        """Test transcription with empty audio"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            text = asr.transcribe(b"")
            assert text == ""
    
    def test_transcribe_failure(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test transcription failure handling"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            # Make model.generate raise an exception
            asr.model.generate.side_effect = Exception("Generation failed")
            
            text = asr.transcribe(sample_audio_bytes_16k)
            assert text == ""
    
    def test_warmup(self, sensevoice_class, mock_funasr):
        """Test warmup method"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            asr.warmup(num_iterations=2)
            assert asr._warmup_done is True
    
    def test_is_available(self, sensevoice_class, mock_funasr):
        """Test is_available method"""
        with patch.dict('os.environ', {}, clear=True):
            # With mock, model loads successfully
            asr = sensevoice_class(device="cpu")
            assert asr.is_available() is True
            
            # Test after explicit load
            asr.load_model()
            assert asr.is_available() is True
    
    def test_is_available_not_initialized(self, sensevoice_class):
        """Test is_available when model fails to load"""
        with patch.dict('os.environ', {}, clear=True):
            with patch('funasr.AutoModel', side_effect=Exception("Model not found")):
                asr = sensevoice_class(device="cpu")
                assert asr.is_available() is False
    
    def test_transcribe_with_emotion_detection(self, sensevoice_class, mock_funasr, sample_audio_bytes_16k):
        """Test transcribe with emotion detection"""
        with patch.dict('os.environ', {}, clear=True):
            asr = sensevoice_class(device="cpu")
            asr.load_model()
            
            # Mock result with emotion tags
            mock_funasr.return_value.generate.return_value = [{
                "text": "<|HAPPY|><|Speech|>你好"
            }]
            
            result = asr.transcribe_with_emotion(sample_audio_bytes_16k)
            
            assert "text" in result
            assert "raw_text" in result
            assert "emotion" in result
            assert "event" in result
            assert result["emotion"] == "HAPPY"
            assert result["event"] == "Speech"


@pytest.mark.phase1
@pytest.mark.unit
@pytest.mark.asr
class TestASRConfig:
    """Test ASR configuration"""
    
    def test_asr_config_defaults(self):
        """Test default ASR configuration"""
        from config import ASRConfig
        
        config = ASRConfig()
        assert config.engine == "sensevoice"  # Updated default
    
    def test_sensevoice_config(self):
        """Test SenseVoice-specific configuration"""
        from config import ASRConfig
        
        config = ASRConfig(
            engine="sensevoice",
            sensevoice_model="iic/SenseVoiceSmall",
            sensevoice_device="cpu",
            sensevoice_language="auto"
        )
        
        assert config.engine == "sensevoice"
        assert config.sensevoice_model == "iic/SenseVoiceSmall"
        assert config.sensevoice_device == "cpu"
        assert config.sensevoice_language == "auto"
