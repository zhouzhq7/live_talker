"""
LLM (Large Language Model) Module
"""

from .base import BaseLLM
from .deepseek import DeepseekLLM

__all__ = [
    "BaseLLM",
    "DeepseekLLM",
]

