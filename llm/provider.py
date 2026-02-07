"""
LLM Provider Interface and Pool Management

Supports multiple LLM providers with automatic failover:
- DeepSeek
- Zhipu (智谱)  
- OpenAI
- Google Gemini
- Xinghuo (讯飞星火)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ProviderHealth:
    """Provider health metrics"""
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_check: float = 0
    response_time_ms: float = 0
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass  
class LLMResponse:
    """Standardized LLM response"""
    text: str
    provider: str
    model: str
    latency_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Any = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers with OpenAI-compatible interface
    
    All LLM implementations should inherit from this class.
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
        Initialize LLM provider
        
        Args:
            name: Provider identifier
            api_key: API key for authentication
            base_url: API base URL
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific config
        """
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.config = kwargs
        
        self._client = None
        self._is_initialized = False
        self.health = ProviderHealth()
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider client
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        OpenAI-compatible chat completion API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
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
        pass
    
    def check_health(self) -> ProviderHealth:
        """
        Check provider health status
        
        Returns:
            ProviderHealth object
        """
        if not self._is_initialized:
            self.health.status = ProviderStatus.UNAVAILABLE
            return self.health
            
        start_time = time.time()
        try:
            # Simple health check with minimal request
            test_response = self.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            
            self.health.response_time_ms = (time.time() - start_time) * 1000
            self.health.last_check = time.time()
            
            if test_response.text:
                self.health.status = ProviderStatus.HEALTHY
                self.health.consecutive_failures = 0
            else:
                self.health.consecutive_failures += 1
                if self.health.consecutive_failures >= 3:
                    self.health.status = ProviderStatus.UNAVAILABLE
                else:
                    self.health.status = ProviderStatus.DEGRADED
                    
        except Exception as e:
            self.health.response_time_ms = (time.time() - start_time) * 1000
            self.health.last_check = time.time()
            self.health.consecutive_failures += 1
            
            if self.health.consecutive_failures >= 3:
                self.health.status = ProviderStatus.UNAVAILABLE
            else:
                self.health.status = ProviderStatus.DEGRADED
                
            logger.debug(f"[{self.name}] Health check failed: {e}")
            
        return self.health
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self._is_initialized and self.health.status != ProviderStatus.UNAVAILABLE
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.name,
            "model": self.model,
            "initialized": self._is_initialized,
            "available": self.is_available(),
            "health": {
                "status": self.health.status.value,
                "success_rate": self.health.success_rate,
                "response_time_ms": self.health.response_time_ms,
                "consecutive_failures": self.health.consecutive_failures
            },
            "config": {
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"


class LLMProviderPool:
    """
    Pool of LLM providers with automatic failover
    
    Manages multiple providers and routes requests to healthy ones.
    """
    
    def __init__(
        self,
        providers: List[LLMProvider],
        fallback_chain: Optional[List[str]] = None,
        health_check_interval: int = 60
    ):
        """
        Initialize provider pool
        
        Args:
            providers: List of LLMProvider instances
            fallback_chain: Ordered list of provider names for fallback
            health_check_interval: Seconds between health checks
        """
        self.providers: Dict[str, LLMProvider] = {}
        for provider in providers:
            self.providers[provider.name] = provider
            
        # Set fallback chain (default to provider order)
        if fallback_chain:
            self.fallback_chain = fallback_chain
        else:
            self.fallback_chain = [p.name for p in providers]
            
        self.health_check_interval = health_check_interval
        self._last_health_check = 0
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "failover_count": 0
        }
        
        logger.info(f"[ProviderPool] Initialized with {len(providers)} providers: {list(self.providers.keys())}")
        
    def _check_health_if_needed(self):
        """Run health checks if interval has passed"""
        current_time = time.time()
        if current_time - self._last_health_check > self.health_check_interval:
            self._run_health_checks()
            
    def _run_health_checks(self):
        """Run health checks on all providers"""
        logger.debug("[ProviderPool] Running health checks...")
        for name, provider in self.providers.items():
            health = provider.check_health()
            logger.debug(f"[ProviderPool] {name}: {health.status.value} (success_rate: {health.success_rate:.2%})")
        self._last_health_check = time.time()
        
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        self._check_health_if_needed()
        return [
            name for name in self.fallback_chain
            if name in self.providers and self.providers[name].is_available()
        ]
        
    def get_primary_provider(self) -> Optional[LLMProvider]:
        """Get the first available provider in fallback chain"""
        available = self.get_available_providers()
        if available:
            return self.providers[available[0]]
        return None
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion with automatic failover
        
        Args:
            messages: List of message dicts
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse
        """
        self.stats["total_requests"] += 1
        
        # Get available providers
        available = self.get_available_providers()
        
        if not available:
            logger.error("[ProviderPool] No providers available!")
            self.stats["failed_requests"] += 1
            return LLMResponse(
                text="",
                provider="none",
                model="none",
                latency_ms=0
            )
        
        # Try each available provider
        last_error = None
        for provider_name in available:
            provider = self.providers[provider_name]
            start_time = time.time()
            
            try:
                logger.debug(f"[ProviderPool] Trying {provider_name}...")
                response = provider.chat_completion(messages, stream=stream, **kwargs)
                
                if response.text:
                    response.latency_ms = (time.time() - start_time) * 1000
                    self.stats["successful_requests"] += 1
                    provider.health.successful_requests += 1
                    provider.health.total_requests += 1
                    
                    if provider_name != available[0]:
                        self.stats["failover_count"] += 1
                        logger.info(f"[ProviderPool] Failover successful: {available[0]} -> {provider_name}")
                        
                    return response
                else:
                    raise Exception("Empty response")
                    
            except Exception as e:
                last_error = e
                provider.health.consecutive_failures += 1
                provider.health.total_requests += 1
                logger.warning(f"[ProviderPool] {provider_name} failed: {e}")
                continue
        
        # All providers failed
        logger.error(f"[ProviderPool] All providers failed. Last error: {last_error}")
        self.stats["failed_requests"] += 1
        return LLMResponse(
            text="",
            provider="none",
            model="none",
            latency_ms=0
        )
        
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[str]:
        """
        Streaming chat completion with failover
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        self.stats["total_requests"] += 1
        
        available = self.get_available_providers()
        
        if not available:
            logger.error("[ProviderPool] No providers available!")
            self.stats["failed_requests"] += 1
            return
        
        for provider_name in available:
            provider = self.providers[provider_name]
            
            try:
                logger.debug(f"[ProviderPool] Streaming with {provider_name}...")
                
                for chunk in provider.chat_completion_stream(messages, **kwargs):
                    yield chunk
                    
                self.stats["successful_requests"] += 1
                provider.health.successful_requests += 1
                provider.health.total_requests += 1
                
                if provider_name != available[0]:
                    self.stats["failover_count"] += 1
                    logger.info(f"[ProviderPool] Failover successful: {available[0]} -> {provider_name}")
                    
                return
                
            except Exception as e:
                provider.health.consecutive_failures += 1
                provider.health.total_requests += 1
                logger.warning(f"[ProviderPool] {provider_name} streaming failed: {e}")
                continue
        
        logger.error("[ProviderPool] All providers failed for streaming")
        self.stats["failed_requests"] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "stats": self.stats.copy(),
            "providers": {
                name: provider.get_info()
                for name, provider in self.providers.items()
            },
            "fallback_chain": self.fallback_chain,
            "available_providers": self.get_available_providers()
        }
        
    def __repr__(self) -> str:
        return f"LLMProviderPool(providers={list(self.providers.keys())})"
