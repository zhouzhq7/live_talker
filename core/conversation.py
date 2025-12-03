"""
Conversation Manager
Manages conversation history and context
"""

import logging
from typing import List, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context
    
    Features:
    - Maintain conversation history
    - Limit history length
    - Format messages for LLM
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history * 2)  # user + assistant pairs
        self.system_prompt: Optional[str] = None
    
    def add_user_message(self, message: str):
        """Add user message to history"""
        self.history.append({
            "role": "user",
            "content": message
        })
        logger.debug(f"[Conversation] Added user message: {message[:50]}...")
    
    def add_assistant_message(self, message: str):
        """Add assistant message to history"""
        self.history.append({
            "role": "assistant",
            "content": message
        })
        logger.debug(f"[Conversation] Added assistant message: {message[:50]}...")
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get formatted messages for LLM
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            List of message dicts
        """
        messages = []
        
        if include_system and self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.extend(list(self.history))
        
        return messages
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        logger.info("[Conversation] History cleared")
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt"""
        self.system_prompt = prompt
        logger.debug(f"[Conversation] System prompt set: {prompt[:50]}...")
    
    def get_history_length(self) -> int:
        """Get current history length"""
        return len(self.history)
    
    def get_info(self) -> dict:
        """Get conversation info"""
        return {
            "history_length": len(self.history),
            "max_history": self.max_history,
            "has_system_prompt": self.system_prompt is not None
        }

