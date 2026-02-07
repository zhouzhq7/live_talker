"""
Unit Tests for LLM Streaming Module
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.base import BaseLLM, LLMResult


class TestLLMResult:
    """Test cases for LLMResult"""

    def test_creation(self):
        """Test LLM result creation"""
        result = LLMResult(
            token="你好",
            is_final=False,
            confidence=0.95,
            stop_reason=None
        )

        assert result.token == "你好"
        assert result.is_final is False
        assert result.confidence == 0.95
        assert result.stop_reason is None

    def test_final_result(self):
        """Test final LLM result with stop reason"""
        result = LLMResult(
            token="再见",
            is_final=True,
            confidence=0.98,
            stop_reason="stop"
        )

        assert result.is_final is True
        assert result.stop_reason == "stop"

    def test_defaults(self):
        """Test default values"""
        result = LLMResult(token="测试", is_final=False)

        assert result.confidence == 0.0
        assert result.stop_reason is None


class TestBaseLLM:
    """Test cases for BaseLLM"""

    def test_init(self):
        """Test BaseLLM initialization"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return "test"

            def generate_stream(self, prompt: str, **kwargs):
                yield "test"

            async def _chat_stream_impl(self, messages):
                yield "test"

        llm = TestLLM(name="test")
        assert llm.name == "test"
        assert llm._is_initialized is False
        assert llm.conversation_history == []

    def test_history_management(self):
        """Test conversation history management"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return "response"

            def generate_stream(self, prompt: str, **kwargs):
                yield "response"

            async def _chat_stream_impl(self, messages):
                yield "response"

        llm = TestLLM(name="test")
        llm.conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}
        ]

        assert len(llm.conversation_history) == 2

        llm.clear_history()
        assert len(llm.conversation_history) == 0

    def test_format_messages(self):
        """Test message formatting"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return ""

            def generate_stream(self, prompt: str, **kwargs):
                yield ""

            async def _chat_stream_impl(self, messages):
                yield ""

        llm = TestLLM(name="test")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        formatted = llm._format_messages(messages)
        assert "system: You are helpful" in formatted
        assert "user: Hello" in formatted
        assert "assistant: Hi!" in formatted

    def test_is_available_default(self):
        """Test default availability"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return ""

            def generate_stream(self, prompt: str, **kwargs):
                yield ""

            async def _chat_stream_impl(self, messages):
                yield ""

        llm = TestLLM(name="test")
        assert llm.is_available() is False

    def test_get_info(self):
        """Test get_info method"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return ""

            def generate_stream(self, prompt: str, **kwargs):
                yield ""

            async def _chat_stream_impl(self, messages):
                yield ""

        llm = TestLLM(name="test", model="test-model")
        info = llm.get_info()

        assert "name" in info
        assert "initialized" in info
        assert "history_length" in info
        assert "config" in info


class TestDeepseekLLM:
    """Test cases for DeepseekLLM"""

    @pytest.fixture
    def deepseek_llm(self):
        """Create DeepseekLLM instance"""
        import os
        # Skip if no API key
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

        from llm.deepseek import DeepseekLLM

        llm = DeepseekLLM(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat"
        )
        return llm

    def test_init(self, deepseek_llm):
        """Test initialization"""
        assert deepseek_llm.name == "Deepseek"
        assert deepseek_llm.model == "deepseek-chat"

    def test_get_info(self, deepseek_llm):
        """Test get_info method"""
        info = deepseek_llm.get_info()

        assert "name" in info
        assert "model" in info
        assert "api_base" in info
        assert "temperature" in info
        assert "has_api_key" in info

    def test_chat_non_streaming(self, deepseek_llm):
        """Test non-streaming chat"""
        response = deepseek_llm.chat(
            user_message="Hello",
            stream=False
        )

        # Response should be a string (may be empty if no API key)
        assert isinstance(response, str)


class TestLLMInterfaceCompatibility:
    """Test LLM interface compatibility"""

    def test_base_methods(self):
        """Test that BaseLLM has required abstract methods"""
        from llm.base import BaseLLM

        required_methods = [
            'generate',
            'generate_stream',
            '_chat_stream_impl'
        ]

        for method in required_methods:
            assert hasattr(BaseLLM, method)

    def test_generate_compatibility(self):
        """Test generate method signature"""
        from llm.base import BaseLLM

        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return "test"

            def generate_stream(self, prompt: str, **kwargs):
                yield "test"

            async def _chat_stream_impl(self, messages):
                yield "test"

        llm = TestLLM(name="test")
        result = llm.generate("test prompt")

        assert isinstance(result, str)

    def test_chat_stream_interface(self):
        """Test async chat_stream interface"""
        from llm.base import BaseLLM

        class TestLLM(BaseLLM):
            def generate(self, prompt: str, stream: bool = False, **kwargs):
                return "test"

            def generate_stream(self, prompt: str, **kwargs):
                yield "test"

            async def _chat_stream_impl(self, messages):
                yield "Hello"
                yield " world"

        llm = TestLLM(name="test")

        # Test that chat_stream exists
        assert hasattr(llm, 'chat_stream')
        assert callable(llm.chat_stream)

        # Test that calling it returns an async generator object
        result = llm.chat_stream("test")
        import inspect
        assert inspect.isasyncgen(result)

    def test_llm_result_namedtuple(self):
        """Test LLMResult as NamedTuple"""
        result = LLMResult(
            token="test",
            is_final=True,
            confidence=0.9,
            stop_reason="stop"
        )

        # Test unpacking
        token, is_final, confidence, stop_reason = result
        assert token == "test"
        assert is_final is True
        assert confidence == 0.9
        assert stop_reason == "stop"

        # Test indexing
        assert result[0] == "test"
        assert result[1] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
