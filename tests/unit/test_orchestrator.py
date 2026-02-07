"""
Unit Tests for StreamOrchestrator
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.state import PipelineState, PipelineEvent
from pipeline.orchestrator import StreamOrchestrator, PipelineConfig


class MockASR:
    """Mock ASR engine for testing"""

    def __init__(self, result_text="测试文本"):
        self.result_text = result_text
        self.call_count = 0

    def transcribe(self, audio_data, sample_rate=16000):
        self.call_count += 1
        return self.result_text

    async def stream_recognize(self, audio_stream):
        """Async stream recognize"""
        self.call_count += 1
        yield {"text": "部分", "is_final": False}
        yield {"text": self.result_text, "is_final": True}


class MockLLM:
    """Mock LLM engine for testing"""

    def __init__(self, response="这是一个测试回复"):
        self.response = response
        self.call_count = 0

    def chat(self, text):
        self.call_count += 1
        return self.response

    async def chat_stream(self, text):
        """Async chat stream"""
        self.call_count += 1
        for char in self.response:
            yield char


class MockTTS:
    """Mock TTS engine for testing"""

    def __init__(self, audio_data=b"audio_data"):
        self.audio_data = audio_data
        self.call_count = 0

    def synthesize(self, text):
        self.call_count += 1
        return self.audio_data

    async def synthesize_stream(self, text_stream):
        """Async synthesize stream"""
        self.call_count += 1
        yield {"audio_chunk": self.audio_data, "is_final": True, "text": "test"}


class TestStreamOrchestrator:
    """Test cases for StreamOrchestrator"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PipelineConfig(enable_parallel=True)

    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator instance"""
        return StreamOrchestrator(
            config=config,
            asr_engine=MockASR(),
            llm_engine=MockLLM(),
            tts_engine=MockTTS(),
        )

    def test_init(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.state == PipelineState.IDLE
        assert orchestrator._running is False
        assert orchestrator._interrupted is False

    def test_start_stop(self, orchestrator):
        """Test start and stop"""
        async def test():
            await orchestrator.start()
            assert orchestrator._running is True
            assert orchestrator.state == PipelineState.LISTENING

            await orchestrator.stop()
            assert orchestrator._running is False
            assert orchestrator.state == PipelineState.IDLE

        asyncio.run(test())

    def test_interrupt(self, orchestrator):
        """Test interruption"""
        async def test():
            await orchestrator.start()

            # Should be able to interrupt
            result = await orchestrator.interrupt()
            assert result is True
            assert orchestrator._interrupted is True
            assert orchestrator.state == PipelineState.INTERRUPTED

        asyncio.run(test())

    def test_interrupt_idle(self, orchestrator):
        """Test interrupting when idle"""
        async def test():
            # Cannot interrupt when idle
            result = await orchestrator.interrupt()
            assert result is False

        asyncio.run(test())

    def test_reset(self, orchestrator):
        """Test reset"""
        async def test():
            await orchestrator.start()
            await orchestrator.interrupt()

            orchestrator.reset()

            assert orchestrator._interrupted is False
            # reset() should transition to IDLE
            assert orchestrator.state == PipelineState.IDLE

        asyncio.run(test())

    def test_callbacks(self, orchestrator):
        """Test callback registration"""
        state_changes = []
        results = []
        errors = []

        def on_state_change(state):
            state_changes.append(state)

        def on_result(result):
            results.append(result)

        def on_error(error):
            errors.append(error)

        orchestrator.set_callbacks(
            on_state_change=on_state_change,
            on_result=on_result,
            on_error=on_error,
        )

        assert orchestrator._on_state_change is not None
        assert orchestrator._on_result is not None
        assert orchestrator._on_error is not None

    def test_get_info(self, orchestrator):
        """Test get_info method"""
        info = orchestrator.get_info()

        assert "state" in info
        assert "running" in info
        assert "interrupted" in info
        assert "engines" in info
        assert "metrics" in info

    def test_process_without_engines(self):
        """Test process when no engines are configured"""
        orchestrator = StreamOrchestrator()

        async def test():
            await orchestrator.start()

            audio_data = b"test_audio"
            result = await orchestrator.process(audio_data)

            # Should complete without error
            assert result.success is True

        asyncio.run(test())


class TestPipelineConfig:
    """Test cases for PipelineConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PipelineConfig()

        assert config.enable_parallel is True
        assert config.buffer_pre_speech == 1.0
        assert config.max_latency == 10.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = PipelineConfig(
            enable_parallel=False,
            buffer_pre_speech=2.0,
            max_latency=5.0,
        )

        assert config.enable_parallel is False
        assert config.buffer_pre_speech == 2.0
        assert config.max_latency == 5.0


class TestOrchestratorIntegration:
    """Integration tests for StreamOrchestrator"""

    @pytest.fixture
    def mock_engines(self):
        """Create mock engines"""
        return MockASR(), MockLLM(), MockTTS()

    def test_full_processing_flow(self, mock_engines):
        """Test complete processing flow"""
        asr, llm, tts = mock_engines

        orchestrator = StreamOrchestrator(
            asr_engine=asr,
            llm_engine=llm,
            tts_engine=tts,
        )

        async def test():
            await orchestrator.start()

            # Process audio
            audio_data = b"test_audio_data"
            result = await orchestrator.process(audio_data)

            # Verify results
            assert result.success is True
            assert result.text == "测试文本"
            assert result.response == "这是一个测试回复"
            assert result.audio == b"audio_data"
            assert result.latency >= 0

            # Verify engines were called
            assert asr.call_count > 0
            assert llm.call_count > 0
            assert tts.call_count > 0

            await orchestrator.stop()

        asyncio.run(test())

    def test_interruption_during_processing(self, mock_engines):
        """Test interruption during processing"""
        # This test verifies that interruption can be triggered
        # Note: Full interruption testing requires async mock engines
        orchestrator = StreamOrchestrator()

        async def test():
            await orchestrator.start()

            # Verify interruption can be triggered
            result = await orchestrator.interrupt()
            assert result is True
            assert orchestrator._interrupted is True
            assert orchestrator.state == PipelineState.INTERRUPTED

            await orchestrator.stop()

        asyncio.run(test())


class TestTextStreamGenerator:
    """Test text stream generator"""

    def test_sentence_splitting(self):
        """Test sentence splitting in text stream"""
        from pipeline.orchestrator import StreamOrchestrator

        orchestrator = StreamOrchestrator()

        text = "你好世界！这是一个测试。多个句子怎么办？"
        expected = [
            "你好世界！",
            "这是一个测试。",
            "多个句子怎么办？"
        ]

        # Convert generator to list
        result = list(orchestrator._text_stream_generator(text))

        assert len(result) == 3

    def test_single_sentence(self):
        """Test single sentence"""
        from pipeline.orchestrator import StreamOrchestrator

        orchestrator = StreamOrchestrator()
        text = "只有一个句子"

        result = list(orchestrator._text_stream_generator(text))
        assert len(result) == 1
        assert result[0] == "只有一个句子"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
