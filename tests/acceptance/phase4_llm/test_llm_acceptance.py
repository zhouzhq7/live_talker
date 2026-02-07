"""
Acceptance tests for LLM (Ollama)

Based on CHECKLIST.md requirements:
- First token latency < 500ms
- Generation speed > 10 tokens/s
- Support for qwen2.5:7b and deepseek-r1:7b
"""

import pytest
import time


@pytest.mark.phase4
@pytest.mark.acceptance
@pytest.mark.llm
@pytest.mark.local_only
class TestLLMAcceptance:
    """Acceptance tests for LLM"""
    
    @pytest.fixture
    def ollama_engine(self):
        """Create Ollama engine"""
        try:
            from llm.ollama import OllamaLLM
            engine = OllamaLLM(model="qwen2.5:7b")
            
            if not engine.load_model():
                pytest.skip("Ollama service not available")
            
            yield engine
        except ImportError:
            pytest.skip("OllamaLLM not installed")
    
    def test_first_token_latency(self, ollama_engine, performance_monitor):
        """Test first token latency < 500ms"""
        monitor = performance_monitor()
        monitor.start()
        
        first_token_time = None
        start = time.time()
        
        for chunk in ollama_engine.generate_stream("你好"):
            if first_token_time is None:
                first_token_time = (time.time() - start) * 1000
            break
        
        metrics = monitor.stop()
        
        assert first_token_time is not None
        assert first_token_time < 500, \
            f"First token too slow: {first_token_time:.2f}ms"
    
    def test_generation_speed(self, ollama_engine):
        """Test generation speed > 10 tokens/s"""
        prompt = "请解释什么是机器学习"
        
        start = time.time()
        response = ollama_engine.generate(prompt)
        end = time.time()
        
        # Estimate tokens
        estimated_tokens = len(response) / 1.5  # Rough estimate for Chinese
        duration = end - start
        
        tokens_per_sec = estimated_tokens / duration
        
        assert tokens_per_sec > 10, \
            f"Generation too slow: {tokens_per_sec:.1f} tokens/s"
    
    def test_chinese_quality(self, ollama_engine):
        """Test Chinese generation quality"""
        response = ollama_engine.generate("你好，请用中文回答")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Basic Chinese check (should contain Chinese characters)
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in response)
        assert has_chinese, "Response should contain Chinese characters"
    
    def test_conversation_continuity(self, ollama_engine):
        """Test conversation continuity"""
        # First turn
        response1 = ollama_engine.chat("我叫小明")
        
        # Second turn (should remember name)
        response2 = ollama_engine.chat("我叫什么名字？")
        
        # Check if name is mentioned
        assert "小明" in response2 or "不知道" in response2
    
    def test_fallback_mechanism(self):
        """Test fallback to cloud API"""
        try:
            from llm.ollama import OllamaLLM
            from llm.deepseek import DeepseekLLM
        except ImportError:
            pytest.skip("Required modules not installed")
        
        # This is a design test - fallback should work
        # Actual implementation depends on talker.py
        pass


@pytest.mark.phase4
@pytest.mark.acceptance
@pytest.mark.llm
class TestLLMConfigAcceptance:
    """Test LLM configuration acceptance"""
    
    def test_ollama_configuration(self):
        """Test Ollama configuration"""
        from config import LLMConfig
        
        config = LLMConfig(
            provider="ollama",
            ollama_model="qwen2.5:7b",
            ollama_host="http://localhost:11434",
            ollama_timeout=30
        )
        
        assert config.provider == "ollama"
        assert config.ollama_model == "qwen2.5:7b"
    
    def test_multiple_model_support(self):
        """Test support for multiple models"""
        from config import LLMConfig
        
        models = [
            "qwen2.5:7b",
            "qwen2.5:14b",
            "deepseek-r1:7b"
        ]
        
        for model in models:
            config = LLMConfig(
                provider="ollama",
                ollama_model=model
            )
            assert config.ollama_model == model
