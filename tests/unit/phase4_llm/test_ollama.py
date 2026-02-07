"""
Unit tests for Ollama LLM
"""

import pytest
from unittest.mock import MagicMock, patch, Mock


@pytest.mark.phase4
@pytest.mark.unit
@pytest.mark.llm
class TestOllamaLLM:
    """Test Ollama LLM implementation"""
    
    @pytest.fixture
    def ollama_class(self):
        """Import OllamaLLM class"""
        try:
            from llm.ollama import OllamaLLM
            return OllamaLLM
        except ImportError:
            pytest.skip("OllamaLLM not implemented yet")
    
    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama API response"""
        return {
            "response": "这是一个测试回复",
            "done": True
        }
    
    def test_ollama_init(self, ollama_class):
        """Test Ollama initialization"""
        llm = ollama_class(
            model="qwen2.5:7b",
            host="http://localhost:11434",
            timeout=30
        )
        
        assert llm.name == "Ollama-qwen2.5:7b"
        assert llm.model == "qwen2.5:7b"
        assert llm.host == "http://localhost:11434"
        assert llm.timeout == 30
        assert not llm._is_initialized
    
    def test_load_model_success(self, ollama_class):
        """Test successful model loading"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "qwen2.5:7b"}]
            }
            
            llm = ollama_class(model="qwen2.5:7b")
            result = llm.load_model()
            
            assert result is True
            assert llm._is_initialized is True
    
    def test_load_model_service_not_running(self, ollama_class):
        """Test model loading when service not running"""
        with patch('requests.get', side_effect=Exception("Connection refused")):
            llm = ollama_class(model="qwen2.5:7b")
            result = llm.load_model()
            
            assert result is False
            assert llm._is_initialized is False
    
    def test_generate_success(self, ollama_class, mock_ollama_response):
        """Test successful generation"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_ollama_response
            
            llm = ollama_class(model="qwen2.5:7b")
            llm._is_initialized = True
            
            response = llm.generate("你好")
            
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_generate_not_initialized(self, ollama_class):
        """Test generation when not initialized"""
        llm = ollama_class(model="qwen2.5:7b")
        # Don't initialize
        
        with pytest.raises(RuntimeError):
            llm.generate("你好")
    
    def test_generate_stream(self, ollama_class):
        """Test streaming generation"""
        def mock_stream_response():
            chunks = [
                b'{"response": "这是"}',
                b'{"response": "一个"}',
                b'{"response": "测试"}',
                b'{"response": "回复", "done": true}'
            ]
            for chunk in chunks:
                yield chunk
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = mock_stream_response()
            mock_post.return_value = mock_response
            
            llm = ollama_class(model="qwen2.5:7b")
            llm._is_initialized = True
            
            chunks = list(llm.generate_stream("你好"))
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chat(self, ollama_class, mock_ollama_response):
        """Test chat interface"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_ollama_response
            
            llm = ollama_class(model="qwen2.5:7b")
            llm._is_initialized = True
            
            response = llm.chat(
                user_message="你好",
                system_prompt="你是一个助手"
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert len(llm.conversation_history) == 2
    
    def test_chat_with_streaming(self, ollama_class):
        """Test chat with streaming"""
        def mock_stream_response():
            chunks = [
                b'{"response": "你好"}',
                b'{"response": "！", "done": true}'
            ]
            for chunk in chunks:
                yield chunk
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = mock_stream_response()
            mock_post.return_value = mock_response
            
            llm = ollama_class(model="qwen2.5:7b")
            llm._is_initialized = True
            
            response = llm.chat("你好", stream=True)
            
            assert isinstance(response, str)
            assert "你好" in response
    
    def test_clear_history(self, ollama_class):
        """Test clearing conversation history"""
        llm = ollama_class(model="qwen2.5:7b")
        llm.conversation_history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}
        ]
        
        llm.clear_history()
        
        assert len(llm.conversation_history) == 0
    
    def test_health_check(self, ollama_class):
        """Test health check"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            llm = ollama_class(model="qwen2.5:7b")
            is_healthy = llm.health_check()
            
            assert is_healthy is True
    
    def test_health_check_failure(self, ollama_class):
        """Test health check failure"""
        with patch('requests.get', side_effect=Exception("Connection error")):
            llm = ollama_class(model="qwen2.5:7b")
            is_healthy = llm.health_check()
            
            assert is_healthy is False
    
    def test_get_info(self, ollama_class):
        """Test get_info method"""
        llm = ollama_class(
            model="qwen2.5:7b",
            host="http://localhost:11434",
            timeout=30
        )
        llm._is_initialized = True
        llm.conversation_history = [{"role": "user", "content": "test"}]
        
        info = llm.get_info()
        
        assert info["name"] == "Ollama-qwen2.5:7b"
        assert info["initialized"] is True
        assert info["history_length"] == 1


@pytest.mark.phase4
@pytest.mark.unit
@pytest.mark.llm
class TestLLMConfig:
    """Test LLM configuration"""
    
    def test_llm_config_defaults(self):
        """Test default LLM configuration"""
        from config import LLMConfig
        
        config = LLMConfig()
        assert config.provider == "deepseek"
    
    def test_ollama_config(self):
        """Test Ollama-specific configuration"""
        from config import LLMConfig
        
        config = LLMConfig(
            provider="ollama",
            ollama_model="qwen2.5:7b",
            ollama_host="http://localhost:11434",
            ollama_timeout=30
        )
        
        assert config.provider == "ollama"
        assert config.ollama_model == "qwen2.5:7b"
        assert config.ollama_host == "http://localhost:11434"
        assert config.ollama_timeout == 30


@pytest.mark.phase4
@pytest.mark.unit
@pytest.mark.llm
class TestLLMBaseClass:
    """Test BaseLLM abstract class"""
    
    def test_base_llm_init(self):
        """Test BaseLLM initialization"""
        from llm.base import BaseLLM
        
        with pytest.raises(TypeError):
            BaseLLM(name="test")
    
    def test_llm_interface(self):
        """Test LLM interface compliance"""
        from llm.base import BaseLLM
        
        required_methods = ['generate', 'generate_stream']
        for method in required_methods:
            assert hasattr(BaseLLM, method)
            assert callable(getattr(BaseLLM, method))
