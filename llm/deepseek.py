"""
Deepseek LLM Integration
参考 eva_development_log/memory/03_LLM方案与Deepseek接入.md
"""

import logging
from typing import Optional, Iterator, List, Dict
from .base import BaseLLM

logger = logging.getLogger(__name__)


class DeepseekLLM(BaseLLM):
    """
    Deepseek LLM implementation using OpenAI-compatible API
    
    Features:
    - OpenAI-compatible API
    - Streaming support
    - Conversation history management
    - Cost-effective Chinese LLM
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize Deepseek LLM
        
        Args:
            api_key: API key (or set DEEPSEEK_API_KEY env var)
            api_base: API base URL (default: Deepseek official)
            model: Model name (deepseek-chat, deepseek-coder)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        super().__init__(name="Deepseek", **kwargs)
        
        import os
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            logger.warning(f"[{self.name}] No API key provided. Set DEEPSEEK_API_KEY env var.")
        
        # Try to initialize
        self.load_model()
    
    def load_model(self) -> bool:
        """Initialize Deepseek client"""
        try:
            from openai import OpenAI
            
            if not self.api_key:
                logger.error(f"[{self.name}] API key required")
                return False
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            self._is_initialized = True
            logger.info(f"[{self.name}] Initialized (model={self.model})")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] openai not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            return False
    
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
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return ""
        
        if stream:
            # Collect streaming chunks
            response = ""
            for chunk in self.generate_stream(prompt, **kwargs):
                response += chunk
            return response
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False
            )
            
            text = response.choices[0].message.content
            return text.strip() if text else ""
            
        except Exception as e:
            logger.error(f"[{self.name}] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate streaming response
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they are generated
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"[{self.name}] Streaming generation failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Chat interface with conversation history (OpenAI format)
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            stream: Whether to stream response
            
        Returns:
            Assistant's response
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return ""
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            if stream:
                # Collect streaming chunks
                response = ""
                stream_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response += content
                
                # Add to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                return response
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                text = response.choices[0].message.content.strip()
                
                # Add to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": text
                })
                
                return text
                
        except Exception as e:
            logger.error(f"[{self.name}] Chat failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_info(self) -> dict:
        """Get Deepseek-specific information"""
        info = super().get_info()
        info.update({
            "model": self.model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_api_key": bool(self.api_key)
        })
        return info

