"""
Integration tests for Ollama LLM

These tests require Ollama service to be running.
Use -m "not local_only" to skip these tests.
"""

import pytest
import time


@pytest.mark.phase4
@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.local_only
class TestOllamaIntegration:
    """Integration tests for Ollama"""
    
    @pytest.fixture(scope="class")
    def ollama_engine(self):
        """Create Ollama engine (requires running service)"""
        try:
            from llm.ollama import OllamaLLM
            engine = OllamaLLM(model="qwen2.5:7b")
            
            if not engine.load_model():
                pytest.skip("Ollama service not available")
            
            yield engine
        except ImportError:
            pytest.skip("OllamaLLM not installed")
    
    def test_simple_generation(self, ollama_engine):
        """Test simple text generation"""
        response = ollama_engine.generate("你好")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_streaming_generation(self, ollama_engine):
        """Test streaming generation"""
        chunks = []
        for chunk in ollama_engine.generate_stream("你好"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    def test_chat_conversation(self, ollama_engine):
        """Test chat with conversation history"""
        # First message
        response1 = ollama_engine.chat("你好")
        assert isinstance(response1, str)
        
        # Second message (should remember context)
        response2 = ollama_engine.chat("今天天气怎么样？")
        assert isinstance(response2, str)
        
        # Verify history is maintained
        assert len(ollama_engine.conversation_history) == 4
    
    def test_system_prompt(self, ollama_engine):
        """Test system prompt"""
        response = ollama_engine.chat(
            user_message="你好",
            system_prompt="你是一个专业的程序员"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.phase4
@pytest.mark.integration
@pytest.mark.llm
class TestLLMIntegration:
    """Test LLM integration with talker"""
    
    def test_llm_factory_integration(self, mock_llm):
        """Test LLM factory creates correct engine"""
        from core.talker import LiveTalker
        from config import TalkerConfig
        
        with patch('core.talker.DeepseekLLM', return_value=mock_llm):
            config = TalkerConfig()
            config.llm.provider = "deepseek"
            
            talker = LiveTalker.__new__(LiveTalker)
            talker.config = config
            
            llm = talker._create_llm()
            assert llm is not None
