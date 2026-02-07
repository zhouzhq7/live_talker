"""
Unit tests for LLM Provider architecture

Tests:
- LLMProvider base class
- LLMProviderPool failover
- OpenAICompatibleProvider
- Individual providers (DeepSeek, Zhipu, etc.)
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from llm.provider import (
    LLMProvider,
    LLMProviderPool,
    LLMResponse,
    ProviderStatus,
    ProviderHealth
)
from llm.openai_compatible import (
    OpenAICompatibleProvider,
    DeepSeekProvider,
    ZhipuProvider,
    OpenAIProvider,
    MoonshotProvider
)
from llm.conversation import ConversationManager, ConversationConfig


class TestProviderHealth:
    """Test ProviderHealth dataclass"""
    
    def test_default_initialization(self):
        """Test default health initialization"""
        health = ProviderHealth()
        assert health.status == ProviderStatus.UNKNOWN
        assert health.success_rate == 1.0
        assert health.total_requests == 0
        
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        health = ProviderHealth(
            total_requests=100,
            successful_requests=95
        )
        assert health.success_rate == 0.95
        
    def test_success_rate_with_zero_requests(self):
        """Test success rate when no requests made"""
        health = ProviderHealth(total_requests=0)
        assert health.success_rate == 1.0


class TestLLMResponse:
    """Test LLMResponse dataclass"""
    
    def test_basic_response(self):
        """Test basic response creation"""
        response = LLMResponse(
            text="Hello",
            provider="test",
            model="test-model"
        )
        assert response.text == "Hello"
        assert response.provider == "test"
        assert response.model == "test-model"
        
    def test_response_with_tokens(self):
        """Test response with token usage"""
        response = LLMResponse(
            text="Hello world",
            provider="test",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=100.5
        )
        assert response.total_tokens == 15
        assert response.latency_ms == 100.5


class MockProvider(LLMProvider):
    """Mock provider for testing"""
    
    def initialize(self) -> bool:
        self._is_initialized = True
        return True
    
    def chat_completion(self, messages, stream=False, **kwargs):
        return LLMResponse(
            text="Mock response",
            provider=self.name,
            model=self.model
        )
    
    def chat_completion_stream(self, messages, **kwargs):
        yield "Mock"
        yield " response"


class TestLLMProvider:
    """Test LLMProvider base class"""
    
    def test_initialization(self):
        """Test provider initialization"""
        provider = MockProvider(
            name="test",
            api_key="test-key",
            model="test-model"
        )
        assert provider.name == "test"
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
        
    def test_is_available(self):
        """Test availability check"""
        provider = MockProvider(name="test")
        assert not provider.is_available()
        
        provider.initialize()
        assert provider.is_available()
        
    def test_get_info(self):
        """Test get_info method"""
        provider = MockProvider(name="test", model="test-model")
        provider.initialize()
        
        info = provider.get_info()
        assert info["name"] == "test"
        assert info["model"] == "test-model"
        assert info["initialized"] is True
        
    def test_health_check_healthy(self):
        """Test health check with healthy provider"""
        provider = MockProvider(name="test")
        provider.initialize()
        
        health = provider.check_health()
        assert health.status == ProviderStatus.HEALTHY
        
    def test_health_check_uninitialized(self):
        """Test health check with uninitialized provider"""
        provider = MockProvider(name="test")
        # Don't initialize
        
        health = provider.check_health()
        assert health.status == ProviderStatus.UNAVAILABLE


class TestLLMProviderPool:
    """Test LLMProviderPool"""
    
    def test_initialization(self):
        """Test pool initialization"""
        p1 = MockProvider(name="p1")
        p2 = MockProvider(name="p2")
        
        pool = LLMProviderPool(providers=[p1, p2])
        
        assert len(pool.providers) == 2
        assert "p1" in pool.providers
        assert "p2" in pool.providers
        
    def test_fallback_chain(self):
        """Test fallback chain configuration"""
        p1 = MockProvider(name="p1")
        p2 = MockProvider(name="p2")
        
        pool = LLMProviderPool(
            providers=[p1, p2],
            fallback_chain=["p2", "p1"]
        )
        
        assert pool.fallback_chain == ["p2", "p1"]
        
    def test_get_available_providers(self):
        """Test getting available providers"""
        p1 = MockProvider(name="p1")
        p2 = MockProvider(name="p2")
        p1.initialize()
        p2.initialize()
        
        pool = LLMProviderPool(providers=[p1, p2])
        available = pool.get_available_providers()
        
        assert "p1" in available
        assert "p2" in available
        
    def test_chat_completion_success(self):
        """Test successful chat completion"""
        p1 = MockProvider(name="p1")
        p1.initialize()
        
        pool = LLMProviderPool(providers=[p1])
        response = pool.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.text == "Mock response"
        assert response.provider == "p1"
        
    def test_chat_completion_failover(self):
        """Test chat completion with failover"""
        # Create a failing provider
        failing = MockProvider(name="failing")
        failing.initialize()
        failing.chat_completion = Mock(side_effect=Exception("Failed"))
        
        # Create a working provider
        working = MockProvider(name="working")
        working.initialize()
        
        pool = LLMProviderPool(
            providers=[failing, working],
            fallback_chain=["failing", "working"]
        )
        
        response = pool.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.text == "Mock response"
        assert response.provider == "working"
        assert pool.stats["failover_count"] == 1
        
    def test_chat_completion_all_fail(self):
        """Test when all providers fail"""
        p1 = MockProvider(name="p1")
        p1.initialize()
        p1.chat_completion = Mock(side_effect=Exception("Failed"))
        
        pool = LLMProviderPool(providers=[p1])
        response = pool.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.text == ""
        assert response.provider == "none"
        assert pool.stats["failed_requests"] == 1
        
    def test_streaming_completion(self):
        """Test streaming completion"""
        p1 = MockProvider(name="p1")
        p1.initialize()
        
        pool = LLMProviderPool(providers=[p1])
        chunks = list(pool.chat_completion_stream(
            messages=[{"role": "user", "content": "Hello"}]
        ))
        
        assert "".join(chunks) == "Mock response"
        
    def test_get_stats(self):
        """Test getting pool statistics"""
        p1 = MockProvider(name="p1")
        p1.initialize()
        
        pool = LLMProviderPool(providers=[p1])
        pool.chat_completion(messages=[{"role": "user", "content": "Hi"}])
        
        stats = pool.get_stats()
        assert stats["stats"]["total_requests"] == 1
        assert stats["stats"]["successful_requests"] == 1
        assert "p1" in stats["providers"]


class TestOpenAICompatibleProvider:
    """Test OpenAICompatibleProvider base class"""
    
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai):
        """Test OpenAI client initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        provider = OpenAICompatibleProvider(
            name="test",
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model"
        )
        
        result = provider.initialize()
        
        assert result is True
        assert provider._is_initialized is True
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.test.com",
            timeout=30
        )
        
    def test_initialization_no_key(self):
        """Test initialization without API key"""
        provider = OpenAICompatibleProvider(
            name="test",
            api_key=None
        )
        
        result = provider.initialize()
        
        assert result is False
        assert provider._is_initialized is False
        
    def test_initialization_no_openai_package(self):
        """Test graceful handling of missing openai package"""
        provider = OpenAICompatibleProvider(name="test")
        provider._OpenAI = None
        
        result = provider.initialize()
        
        assert result is False


class TestDeepSeekProvider:
    """Test DeepSeekProvider"""
    
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_from_env(self, mock_openai):
        """Test initialization from environment variable"""
        mock_openai.return_value = Mock()
        
        provider = DeepSeekProvider()
        
        assert provider.name == "deepseek"
        assert provider.model == "deepseek-chat"
        mock_openai.assert_called()
        
    @patch('openai.OpenAI')
    def test_custom_model(self, mock_openai):
        """Test with custom model"""
        mock_openai.return_value = Mock()
        
        provider = DeepSeekProvider(
            api_key="test-key",
            model="deepseek-coder"
        )
        
        assert provider.model == "deepseek-coder"


class TestZhipuProvider:
    """Test ZhipuProvider"""
    
    @patch.dict(os.environ, {"ZHIPU_API_KEY": "test-key"})
    @patch('openai.OpenAI')
    def test_from_env(self, mock_openai):
        """Test initialization from environment variable"""
        mock_openai.return_value = Mock()
        
        provider = ZhipuProvider()
        
        assert provider.name == "zhipu"
        assert provider.model == "glm-4-flash"
        
    @patch('openai.OpenAI')
    def test_flash_model(self, mock_openai):
        """Test GLM-4-Flash (free model)"""
        mock_openai.return_value = Mock()
        
        provider = ZhipuProvider(api_key="test-key")
        
        assert provider.model == "glm-4-flash"


class TestConversationConfig:
    """Test ConversationConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ConversationConfig()
        
        assert config.primary_provider == "deepseek"
        assert config.enable_fallback is True
        assert "zhipu" in config.fallback_providers
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = ConversationConfig(
            primary_provider="zhipu",
            zhipu_model="glm-4-air",
            temperature=0.5
        )
        
        assert config.primary_provider == "zhipu"
        assert config.zhipu_model == "glm-4-air"
        assert config.temperature == 0.5


class TestConversationManager:
    """Test ConversationManager"""
    
    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "",
        "ZHIPU_API_KEY": "",
        "OPENAI_API_KEY": "",
        "MOONSHOT_API_KEY": ""
    }, clear=True)
    def test_initialization_no_providers(self):
        """Test initialization with no providers available"""
        config = ConversationConfig(
            deepseek_api_key=None,
            zhipu_api_key=None,
            openai_api_key=None,
            moonshot_api_key=None
        )
        
        manager = ConversationManager(config)
        
        assert manager.is_ready() is False
        
    def test_clear_history(self):
        """Test clearing conversation history"""
        config = ConversationConfig()
        manager = ConversationManager(config)
        
        # Simulate some history
        manager.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        manager.clear_history()
        
        assert len(manager.conversation_history) == 0
        
    def test_get_stats_empty(self):
        """Test getting stats with no activity"""
        config = ConversationConfig()
        manager = ConversationManager(config)
        
        stats = manager.get_stats()
        
        assert stats["messages_sent"] == 0
        assert stats["messages_received"] == 0
        assert stats["avg_latency_ms"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
