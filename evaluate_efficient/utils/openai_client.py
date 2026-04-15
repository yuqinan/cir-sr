"""
OpenAI API client for teacher/student model evaluation.
Provides a unified interface for both local and OpenAI API models with async support.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI, AsyncOpenAI
import time
import json

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for interacting with OpenAI API models.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI client with both sync and async capabilities.
        
        Args:
            model_name: Name of the OpenAI model (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        
        # Check if this is an OpenRouter model
        self.is_openrouter = model_name.startswith("gpt-oss") or model_name.startswith("qwen3-30b-a3b-thinking-2507")
        if self.is_openrouter:
            if model_name.startswith("gpt-oss"):
                self.model_name = "openai/" + self.model_name
            else:
                self.model_name = "qwen/" + self.model_name
        
        if self.is_openrouter:
            # Initialize OpenRouter client
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key is None:
                raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            self.async_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=openrouter_key
            )
        else:
            # Initialize standard OpenAI client
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        
        # Concurrent request limit
        self.concurrent_requests = 50   # Max concurrent async requests (increased)
        self.semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        # Rate limiting
        self.requests_per_minute = 3500  # Conservative limit for most OpenAI tiers
        self.request_timestamps = []
        
        logger.info(f"Initialized OpenAI client (sync + async) for model: {model_name}")
        
        # Cost tracking (thread-safe)
        self._lock = asyncio.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        
        # OpenAI pricing (per 1M tokens) - update as needed
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "o1-preview": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o3-mini": {"input": 1.25, "output": 5.00},
            'gpt-5-mini': {"input": 0.25, "output": 2},
        }
    
    def calculate_total_cost(self) -> Dict[str, float]:
        """Calculate total cost based on token usage."""
        model_pricing = self.pricing.get(self.model_name, {"input": 1.0, "output": 2.0})
        
        prompt_cost = (self.total_prompt_tokens / 1_000_000) * model_pricing["input"]
        completion_cost = (self.total_completion_tokens / 1_000_000) * model_pricing["output"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }
    
    def print_cost_summary(self, checkpoint_name: str = ""):
        """Print final cost summary."""
        cost_info = self.calculate_total_cost()
        
        print(f"\n🔸 OpenAI Cost Summary{f' - {checkpoint_name}' if checkpoint_name else ''}:")
        print(f"   Model: {self.model_name} | Requests: {self.total_requests:,}")
        print(f"   Tokens: {cost_info['total_tokens']:,} | Cost: ${cost_info['total_cost']:.4f}")
    
    
    def generate_single(self, prompt: str, temperature: float = 1.0, max_tokens: int = 1024, 
                       top_p: float = 1.0, **kwargs) -> Dict[str, Any]:
        """
        Generate a single response using OpenAI API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary with text and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                #temperature=temperature,
               #max_tokens=max_tokens,
                #top_p=top_p,
                #**kwargs
            )
            
            if self.is_openrouter:
                # For OpenRouter models, format as <think>reasoning</think><answer>content</answer>
                content = response.choices[0].message.content or ""
                reasoning = getattr(response.choices[0].message, 'reasoning', '')
                
                generated_text = f"<think>{reasoning}</think>\n<answer>{content}</answer>"
            else:
                # Standard OpenAI response
                generated_text = response.choices[0].message.content
            
            finish_reason = response.choices[0].finish_reason
            
            result = {
                'text': generated_text,
                'finish_reason': finish_reason,
                'num_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'text': "",
                'finish_reason': "error",
                'num_tokens': 0,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'error': str(e)
            }
    
    
    async def generate_individual_async(self, prompts: List[str], temperature: float = 1.0, 
                                      max_tokens: int = 1024, top_p: float = 1.0, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate individual responses for each prompt using separate API calls for better accuracy.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments
            
        Returns:
            List of response dictionaries
        """
        async def run_prompt(prompt: str):
            """Run a single prompt asynchronously."""
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    #temperature=temperature,
                    #max_tokens=max_tokens,
                    #top_p=top_p,
                    #**kwargs
                )
                
                if self.is_openrouter:
                    # For OpenRouter models, format as <think>reasoning</think><answer>content</answer>
                    print(response.choices[0].message)
                    content = response.choices[0].message.content or ""
                    reasoning = getattr(response.choices[0].message, 'reasoning', '')
                    
                    generated_text = f"<think>{reasoning}</think>\n<answer>{content}</answer>"
                else:
                    # Standard OpenAI response
                    generated_text = response.choices[0].message.content
                
                finish_reason = response.choices[0].finish_reason
                
                result = {
                    'text': generated_text,
                    'finish_reason': finish_reason,
                    'num_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating individual response: {str(e)}")
                return {
                    'text': "",
                    'finish_reason': "error",
                    'num_tokens': 0,
                    'total_tokens': 0,
                    'prompt_tokens': 0,
                    'error': str(e)
                }
        
        logger.info(f"Generating {len(prompts)} individual responses concurrently")
        
        # Create a task for each prompt
        tasks = [run_prompt(prompt) for prompt in prompts]
        
        # Run them all concurrently
        results = await asyncio.gather(*tasks)
        
        # Update cost tracking after all requests complete
        for result in results:
            if 'prompt_tokens' in result and 'num_tokens' in result:
                self.total_prompt_tokens += result.get('prompt_tokens', 0)
                self.total_completion_tokens += result.get('num_tokens', 0)
                self.total_requests += 1
        
        return results

