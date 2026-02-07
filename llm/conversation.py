"""
Conversation Manager with Multi-Provider LLM Support

Manages conversation flow with automatic provider failover.
"""

import time
import logging
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, field

from .provider import LLMProviderPool, LLMResponse
from .openai_compatible import (
    DeepSeekProvider,
    ZhipuProvider,
    OpenAIProvider,
    MoonshotProvider
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """Conversation configuration"""
    # Provider settings
    primary_provider: str = "deepseek"
    fallback_providers: List[str] = field(default_factory=lambda: ["zhipu", "openai"])
    enable_fallback: bool = True
    
    # Generation settings
    system_prompt: str = "You are a helpful assistant."
    max_history: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # API Keys (loaded from env if not provided)
    deepseek_api_key: Optional[str] = None
    zhipu_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None
    
    # Model settings
    deepseek_model: str = "deepseek-chat"
    zhipu_model: str = "glm-4-flash"
    openai_model: str = "gpt-4o-mini"
    moonshot_model: str = "moonshot-v1-8k"


class ConversationManager:
    """
    Manages conversation with multi-provider LLM support
    
    Features:
    - Automatic provider failover
    - Conversation history management
    - Token usage tracking
    - Streaming support
    """
    
    def __init__(self, config: Optional[ConversationConfig] = None):
        """
        Initialize conversation manager
        
        Args:
            config: Conversation configuration
        """
        self.config = config or ConversationConfig()
        self.pool: Optional[LLMProviderPool] = None
        self.conversation_history: List[Dict[str, str]] = []
        
        # Statistics
        self.session_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "total_tokens": 0,
            "total_latency_ms": 0
        }
        
        # Initialize providers
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize LLM provider pool"""
        providers = []
        
        # DeepSeek
        if self.config.deepseek_api_key or self._get_env("DEEPSEEK_API_KEY"):
            provider = DeepSeekProvider(
                api_key=self.config.deepseek_api_key,
                model=self.config.deepseek_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            if provider.is_available():
                providers.append(provider)
                logger.info(f"[Conversation] DeepSeek provider ready")
        
        # Zhipu
        if self.config.zhipu_api_key or self._get_env("ZHIPU_API_KEY"):
            provider = ZhipuProvider(
                api_key=self.config.zhipu_api_key,
                model=self.config.zhipu_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            if provider.is_available():
                providers.append(provider)
                logger.info(f"[Conversation] Zhipu provider ready")
        
        # OpenAI
        if self.config.openai_api_key or self._get_env("OPENAI_API_KEY"):
            provider = OpenAIProvider(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            if provider.is_available():
                providers.append(provider)
                logger.info(f"[Conversation] OpenAI provider ready")
        
        # Moonshot (Kimi)
        if self.config.moonshot_api_key or self._get_env("MOONSHOT_API_KEY"):
            provider = MoonshotProvider(
                api_key=self.config.moonshot_api_key,
                model=self.config.moonshot_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            if provider.is_available():
                providers.append(provider)
                logger.info(f"[Conversation] Moonshot provider ready")
        
        if not providers:
            logger.error("[Conversation] No LLM providers available!")
            logger.error("Please set at least one API key:")
            logger.error("  - DEEPSEEK_API_KEY")
            logger.error("  - ZHIPU_API_KEY")
            logger.error("  - OPENAI_API_KEY")
            logger.error("  - MOONSHOT_API_KEY")
            return
        
        # Build fallback chain
        fallback_chain = [self.config.primary_provider]
        for provider_name in self.config.fallback_providers:
            if provider_name not in fallback_chain:
                fallback_chain.append(provider_name)
        
        # Filter to only available providers
        available_names = [p.name for p in providers]
        fallback_chain = [p for p in fallback_chain if p in available_names]
        
        # Create pool
        self.pool = LLMProviderPool(
            providers=providers,
            fallback_chain=fallback_chain,
            health_check_interval=60
        )
        
        logger.info(f"[Conversation] Provider pool initialized with {len(providers)} providers")
        logger.info(f"[Conversation] Fallback chain: {fallback_chain}")
        
    def _get_env(self, key: str) -> Optional[str]:
        """Get environment variable"""
        import os
        return os.getenv(key)
        
    def chat(
        self,
        message: str,
        stream: bool = False,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Send message and get response
        
        Args:
            message: User message
            stream: Whether to stream response
            system_prompt: Optional system prompt override
            
        Returns:
            Assistant response
        """
        if not self.pool:
            return "Error: No LLM providers available"
        
        # Build messages
        messages = []
        
        # System prompt
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Log request
        logger.info("=" * 70)
        logger.info(f"🤖 [Conversation] Sending message")
        logger.info(f"   - Message length: {len(message)} chars")
        logger.info(f"   - History length: {len(self.conversation_history)} messages")
        logger.info(f"   - Stream: {stream}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        if stream:
            # Collect streaming response
            response_text = ""
            for chunk in self.pool.chat_completion_stream(messages):
                response_text += chunk
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Update history
            self._update_history(message, response_text)
            
            logger.info(f"[Conversation] Streaming response: {len(response_text)} chars ({elapsed_ms:.0f}ms)")
            
            return response_text
        else:
            # Non-streaming
            response = self.pool.chat_completion(messages, stream=False)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.text:
                # Update history
                self._update_history(message, response.text)
                
                # Update stats
                self.session_stats["messages_sent"] += 1
                self.session_stats["messages_received"] += 1
                self.session_stats["total_tokens"] += response.total_tokens
                self.session_stats["total_latency_ms"] += elapsed_ms
                
                logger.info("=" * 70)
                logger.info(f"✅ [Conversation] Response received")
                logger.info(f"   - Provider: {response.provider}")
                logger.info(f"   - Model: {response.model}")
                logger.info(f"   - Latency: {elapsed_ms:.0f}ms")
                logger.info(f"   - Tokens: {response.total_tokens}")
                logger.info(f"   - Response length: {len(response.text)} chars")
                logger.info(f"   - Preview: {response.text[:100]}{'...' if len(response.text) > 100 else ''}")
                logger.info("=" * 70)
                
                return response.text
            else:
                logger.error("[Conversation] Empty response from all providers")
                return "Sorry, I couldn't generate a response. Please try again."
    
    def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        Send message and get streaming response
        
        Args:
            message: User message
            system_prompt: Optional system prompt override
            
        Yields:
            Response chunks
        """
        if not self.pool:
            yield "Error: No LLM providers available"
            return
        
        # Build messages
        messages = []
        
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})
        
        # Collect full response for history
        full_response = ""
        
        for chunk in self.pool.chat_completion_stream(messages):
            full_response += chunk
            yield chunk
        
        # Update history after streaming completes
        if full_response:
            self._update_history(message, full_response)
            self.session_stats["messages_sent"] += 1
            self.session_stats["messages_received"] += 1
    
    def _update_history(self, user_msg: str, assistant_msg: str):
        """Update conversation history"""
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})
        
        # Trim history to max size
        max_hist = self.config.max_history * 2  # *2 for user+assistant pairs
        if len(self.conversation_history) > max_hist:
            self.conversation_history = self.conversation_history[-max_hist:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("[Conversation] History cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        stats = self.session_stats.copy()
        
        if stats["messages_received"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["messages_received"]
        else:
            stats["avg_latency_ms"] = 0
        
        if self.pool:
            stats["providers"] = self.pool.get_stats()
        
        return stats
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        if not self.pool:
            return []
        return self.pool.get_available_providers()
    
    def is_ready(self) -> bool:
        """Check if conversation manager is ready"""
        return self.pool is not None and len(self.get_available_providers()) > 0
