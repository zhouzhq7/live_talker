"""
Base LLM Interface
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterator
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers
    
    All LLM implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize LLM provider
        
        Args:
            name: Provider name for identification
            **kwargs: Provider-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._is_initialized = False
        self.conversation_history: List[Dict[str, str]] = []
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            stream: Whether to stream response
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate streaming response
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        pass
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Chat interface with conversation history
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            stream: Whether to stream response
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Generate response
        if stream:
            response = ""
            for chunk in self.generate_stream(self._format_messages(messages)):
                response += chunk
            return response
        else:
            response = self.generate(self._format_messages(messages), stream=False)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for prompt
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.debug(f"[{self.name}] Conversation history cleared")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM provider
        
        Returns:
            Dictionary with provider information
        """
        return {
            "name": self.name,
            "initialized": self._is_initialized,
            "history_length": len(self.conversation_history),
            "config": self.config
        }
    
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available
        
        Returns:
            True if available, False otherwise
        """
        return self._is_initialized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

