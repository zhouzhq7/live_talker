"""
OpenAI-compatible LLM Provider Base Class

Supports any provider using OpenAI API format:
- DeepSeek
- Zhipu (through OpenAI compatibility layer)
- OpenAI
- Moonshot (Kimi)
"""

import time
import logging
from typing import Optional, List, Dict, Any, Iterator

from .provider import LLMProvider, LLMResponse, ProviderStatus

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """
    Generic OpenAI-compatible API provider
    
    Works with any service that implements OpenAI's chat completion API.
    """
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible provider
        
        Args:
            name: Provider name
            api_key: API key
            base_url: API base URL (e.g., https://api.deepseek.com)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            timeout: Request timeout
            **kwargs: Additional config
        """
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # Import OpenAI here to allow graceful import error handling
        try:
            from openai import OpenAI
            self._OpenAI = OpenAI
        except ImportError:
            logger.error(f"[{name}] openai package not installed. Install with: pip install openai")
            self._OpenAI = None
            
    def initialize(self) -> bool:
        """Initialize OpenAI client"""
        if self._OpenAI is None:
            logger.error(f"[{self.name}] Cannot initialize: openai package not available")
            return False
            
        if not self.api_key:
            logger.error(f"[{self.name}] Cannot initialize: API key required")
            return False
            
        try:
            self._client = self._OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self._is_initialized = True
            logger.info(f"[{self.name}] Initialized (model={self.model}, base_url={self.base_url})")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            return False
            
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        OpenAI-compatible chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse
        """
        if not self._is_initialized:
            return LLMResponse(
                text="",
                provider=self.name,
                model=self.model,
                latency_ms=0
            )
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            text = response.choices[0].message.content or ""
            
            # Extract token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
            
            return LLMResponse(
                text=text.strip(),
                provider=self.name,
                model=self.model,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Chat completion failed: {e}")
            self.health.consecutive_failures += 1
            return LLMResponse(
                text="",
                provider=self.name,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000
            )
            
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[str]:
        """
        Streaming chat completion
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return
        
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"[{self.name}] Streaming failed: {e}")
            self.health.consecutive_failures += 1
            raise


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider"""
    
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        **kwargs
    ):
        """
        Initialize DeepSeek provider
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model name (deepseek-chat, deepseek-coder)
            **kwargs: Additional config
        """
        import os
        
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = kwargs.pop("base_url", None) or os.getenv(
            "DEEPSEEK_API_BASE", 
            self.DEFAULT_BASE_URL
        )
        
        super().__init__(
            name="deepseek",
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
        
        # Auto-initialize
        self.initialize()


class ZhipuProvider(OpenAICompatibleProvider):
    """Zhipu AI (智谱) API provider"""
    
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-flash",
        **kwargs
    ):
        """
        Initialize Zhipu provider
        
        Args:
            api_key: Zhipu API key (or set ZHIPU_API_KEY env var)
            model: Model name (glm-4-flash, glm-4-air, glm-4)
            **kwargs: Additional config
        """
        import os
        
        api_key = api_key or os.getenv("ZHIPU_API_KEY")
        base_url = kwargs.pop("base_url", None) or os.getenv(
            "ZHIPU_API_BASE",
            self.DEFAULT_BASE_URL
        )
        
        super().__init__(
            name="zhipu",
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
        
        # Auto-initialize
        self.initialize()


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider"""
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        **kwargs
    ):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            **kwargs: Additional config
        """
        import os
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = kwargs.pop("base_url", None) or os.getenv(
            "OPENAI_API_BASE",
            self.DEFAULT_BASE_URL
        )
        
        super().__init__(
            name="openai",
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
        
        # Auto-initialize
        self.initialize()


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot (Kimi) API provider"""
    
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",
        **kwargs
    ):
        """
        Initialize Moonshot provider
        
        Args:
            api_key: Moonshot API key (or set MOONSHOT_API_KEY env var)
            model: Model name (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k)
            **kwargs: Additional config
        """
        import os
        
        api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        base_url = kwargs.pop("base_url", None) or os.getenv(
            "MOONSHOT_API_BASE",
            self.DEFAULT_BASE_URL
        )
        
        super().__init__(
            name="moonshot",
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
        
        # Auto-initialize
        self.initialize()
