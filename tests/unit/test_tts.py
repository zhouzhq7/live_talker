"""
Unit Tests for TTS Streaming Module
"""

import pytest
import sys
import os
import asyncio
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.base import BaseTTS, TTSResult


class TestTTSResult:
    """Test cases for TTSResult"""

    def test_creation(self):
        """Test TTS result creation"""
        audio_data = b'\x00\x01\x02\x03'
        result = TTSResult(
            audio_chunk=audio_data,
            text="你好",
            is_final=True,
            sample_rate=16000
        )

        assert result.audio_chunk == audio_data
        assert result.text == "你好"
        assert result.is_final is True
        assert result.sample_rate == 16000

    def test_defaults(self):
        """Test default values"""
        audio_data = b'\x00\x01\x02\x03'
        result = TTSResult(
            audio_chunk=audio_data,
            text="测试",
            is_final=False
        )

        assert result.sample_rate == 16000

    def test_unpacking(self):
        """Test NamedTuple unpacking"""
        audio_data = b'\x00\x01\x02\x03'
        result = TTSResult(
            audio_chunk=audio_data,
            text="测试",
            is_final=True,
            sample_rate=16000
        )

        chunk, text, is_final, sr = result
        assert chunk == audio_data
        assert text == "测试"
        assert is_final is True
        assert sr == 16000


class TestBaseTTS:
    """Test cases for BaseTTS"""

    def test_init(self):
        """Test BaseTTS initialization"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test")
        assert tts.name == "test"
        assert tts._is_initialized is False
        assert tts.config == {}

    def test_is_available_default(self):
        """Test default availability"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test")
        assert tts.is_available() is False

    def test_get_info(self):
        """Test get_info method"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test", voice="test-voice")
        info = tts.get_info()

        assert "name" in info
        assert "initialized" in info
        assert "config" in info

    def test_synthesize_to_file(self):
        """Test synthesize_to_file method"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b"test audio data"

        tts = TestTTS(name="test")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name

        try:
            result = tts.synthesize_to_file("test text", tmp_path)
            assert result is True

            with open(tmp_path, 'rb') as f:
                content = f.read()
            assert content == b"test audio data"
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_split_sentences(self):
        """Test sentence splitting"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test")

        # Test basic splitting
        text = "你好。世界。你好吗？"
        sentences = tts._split_sentences(text)
        assert len(sentences) >= 2

        # Test with newlines
        text2 = "第一句。第二句\n第三句。"
        sentences2 = tts._split_sentences(text2)
        assert len(sentences2) >= 2


class TestEdgeTTS:
    """Test cases for EdgeTTS"""

    @pytest.fixture
    def edge_tts(self):
        """Create EdgeTTS instance"""
        pytest.importorskip("edge_tts")
        from tts.edge_tts import EdgeTTS

        tts = EdgeTTS(voice="zh-CN-XiaoxiaoNeural")
        return tts

    def test_init(self, edge_tts):
        """Test initialization"""
        assert edge_tts.name == "Edge-TTS"
        assert edge_tts.voice == "zh-CN-XiaoxiaoNeural"

    def test_get_info(self, edge_tts):
        """Test get_info method"""
        info = edge_tts.get_info()

        assert "name" in info
        assert "voice" in info
        assert "rate" in info
        assert "volume" in info

    def test_synthesize(self, edge_tts):
        """Test basic synthesis"""
        result = edge_tts.synthesize("你好")

        # Result should be bytes (may be empty if FFmpeg not available)
        assert isinstance(result, bytes)


class TestPyttsx3TTS:
    """Test cases for Pyttsx3TTS"""

    @pytest.fixture
    def pyttsx3_tts(self):
        """Create Pyttsx3TTS instance"""
        pytest.importorskip("pyttsx3")
        from tts.pyttsx3_tts import Pyttsx3TTS

        tts = Pyttsx3TTS()
        return tts

    def test_init(self, pyttsx3_tts):
        """Test initialization"""
        assert pyttsx3_tts.name == "Pyttsx3"

    def test_get_info(self, pyttsx3_tts):
        """Test get_info method"""
        info = pyttsx3_tts.get_info()

        assert "name" in info
        assert "rate" in info
        assert "volume" in info


class TestTTSInterfaceCompatibility:
    """Test TTS interface compatibility"""

    def test_base_methods(self):
        """Test that BaseTTS has required methods"""
        required_methods = [
            'synthesize',
            'synthesize_stream',
            'get_info',
            'is_available',
            '_split_sentences'
        ]

        for method in required_methods:
            assert hasattr(BaseTTS, method)

    def test_synthesize_compatibility(self):
        """Test synthesize method signature"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b"test"

        tts = TestTTS(name="test")
        result = tts.synthesize("test prompt")

        assert isinstance(result, bytes)

    def test_tts_result_namedtuple(self):
        """Test TTSResult as NamedTuple"""
        audio = b'\x00\x01\x02\x03'
        result = TTSResult(
            audio_chunk=audio,
            text="test",
            is_final=True,
            sample_rate=16000
        )

        # Test unpacking
        chunk, text, is_final, sr = result
        assert chunk == audio
        assert text == "test"
        assert is_final is True
        assert sr == 16000

        # Test indexing
        assert result[0] == audio
        assert result[1] == "test"

    def test_synthesize_stream_interface(self):
        """Test synthesize_stream returns async generator"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b"test"

        tts = TestTTS(name="test")

        # Test that synthesize_stream exists and is callable
        assert hasattr(tts, 'synthesize_stream')
        assert callable(tts.synthesize_stream)


class TestTTSSentenceSplitting:
    """Test sentence splitting functionality"""

    def test_chinese_sentences(self):
        """Test Chinese sentence splitting"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test")

        text = "你好，我叫小明。今天天气很好。你吃了吗？"
        sentences = tts._split_sentences(text)

        # Should split on Chinese punctuation
        assert any("你好" in s for s in sentences)

    def test_short_text(self):
        """Test short text handling"""
        class TestTTS(BaseTTS):
            def synthesize(self, text: str, output_file: Optional[str] = None) -> bytes:
                return b""

        tts = TestTTS(name="test")

        text = "hi"
        sentences = tts._split_sentences(text)

        # Short text should not be split
        assert len(sentences) == 1 or "hi" in sentences[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
