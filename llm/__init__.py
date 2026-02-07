"""
LLM (Large Language Model) Module

Multi-provider LLM support with automatic failover:
- DeepSeek
- Zhipu (智谱)
- OpenAI
- Moonshot (Kimi)
"""

# Base classes
from .base import BaseLLM
from .provider import LLMProvider, LLMProviderPool, LLMResponse, ProviderStatus

# OpenAI-compatible providers
from .openai_compatible import (
    OpenAICompatibleProvider,
    DeepSeekProvider,
    ZhipuProvider,
    OpenAIProvider,
    MoonshotProvider
)

# Conversation management
from .conversation import ConversationManager, ConversationConfig

# Legacy support
from .deepseek import DeepseekLLM

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMProvider",
    "LLMProviderPool",
    "LLMResponse",
    "ProviderStatus",
    
    # Providers
    "OpenAICompatibleProvider",
    "DeepSeekProvider",
    "ZhipuProvider",
    "OpenAIProvider",
    "MoonshotProvider",
    
    # Conversation
    "ConversationManager",
    "ConversationConfig",
    
    # Legacy
    "DeepseekLLM",
]
