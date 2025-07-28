"""
Run models on scenario evals.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from prefill_evals.models import ScenarioEval, to_anthropic_format, to_openai_format

from pathlib import Path

from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
import time
import logging
from functools import wraps
from datetime import datetime

from dotenv import load_dotenv

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from prefill_evals.models import ModelSpec

# Configure logging
logger = logging.getLogger(__name__)

def exponential_backoff_retry(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries + 1} attempts: {str(e)}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

@dataclass
class ResponseGrading:
    """
    Standardized grading of a model response.
    """
    score: float
    data: Optional[Dict[str, Any]] = None

class ResponseGrader(ABC):
    """
    Grade a model response.
    """
    name: str

    @abstractmethod
    async def grade(self, response: str, eval: ScenarioEval) -> ResponseGrading:
        pass

@dataclass
class EvalResult:
    """
    Result of running a scenario eval on a single model.
    """
    model: ModelSpec
    num_runs: int
    responses: List[str]
    grades: List[ResponseGrading]

class ScenarioEvaluator:
    """
    Run models on scenario evals.
    """

    def __init__(
        self,
        eval: ScenarioEval,
        runs_per_model: int = 1,
        graders: List[ResponseGrader] = None,
        cache_dir: Optional[Path] = None,
        dotenv_path: Optional[Path] = None,
    ):
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        if cache_dir:
            raise NotImplementedError("Evaluation caching not yet implemented")
        
        self.eval = eval
        self.runs_per_model = runs_per_model
        self.cache_dir = cache_dir
        self.graders = graders

        self.anthropic_client = AsyncAnthropic()
        self.openai_client = AsyncOpenAI()

    @exponential_backoff_retry(max_retries=5, base_delay=1.0, max_delay=60.0)
    async def _call_anthropic_api(self, params: dict) -> str:
        """Make a single API call to Anthropic with retry logic."""
        response = await self.anthropic_client.messages.create(**params)
        
        # Extract only text content from the response
        response_text = ""
        for content_block in response.content:
            if content_block.type == "text":
                response_text += content_block.text
        
        return response_text.strip()

    async def run_anthropic_model(self, model_id: str, num_runs: int = 1, max_response_tokens: Optional[int] = None) -> List[str]:
        """Run Anthropic model on the scenario eval."""
        responses = []
        
        # Convert messages to Anthropic format
        messages = to_anthropic_format(self.eval.messages)
        
        # Convert tools to Anthropic format if present
        tools = None
        if self.eval.tools:
            tools = [tool.to_anthropic_format() for tool in self.eval.tools]
        
        logger.debug(f"Running Anthropic model {model_id} with max response tokens {max_response_tokens}")

        for run_idx in range(num_runs):
            # Create the API call parameters
            params = {
                "model": model_id,
                "messages": messages,
                "max_completion_tokens": 4096 if max_response_tokens is None else max_response_tokens,
            }
            
            # Add system prompt if present
            if self.eval.system:
                params["system"] = self.eval.system
                
            # Add tools if present
            if tools:
                params["tools"] = tools
            
            try:
                # Make the API call with retry logic
                response_text = await self._call_anthropic_api(params)
                responses.append(response_text)
            except Exception as e:
                logger.error(f"Failed to get response from {model_id} (run {run_idx + 1}/{num_runs}): {str(e)}")
                raise
        
        return responses
    
    @exponential_backoff_retry(max_retries=5, base_delay=1.0, max_delay=60.0)
    async def _call_openai_api(self, params: dict) -> List[str]:
        """Make a single API call to OpenAI with retry logic."""

        response = await self.openai_client.chat.completions.create(**params)
        
        # Extract only text content from the response
        response_texts = [choice.message.content or "" for choice in response.choices]
        
        return response_texts

    async def run_openai_model(self, model_id: str, num_runs: int = 1, max_response_tokens: Optional[int] = None) -> List[str]:
        """Run OpenAI model on the scenario eval."""
        # Convert messages to OpenAI format
        messages = to_openai_format(self.eval.messages)
        
        # Add system message if present
        if self.eval.system:
            messages.insert(0, {"role": "system", "content": self.eval.system})
        
        # Convert tools to OpenAI format if present
        tools = None
        if self.eval.tools:
            tools = [tool.to_openai_format() for tool in self.eval.tools]

        params = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 4096 if max_response_tokens is None else max_response_tokens,
            "n": num_runs,
        }
            
        # Add tools if present
        if tools:
            params["tools"] = tools
        
        logger.debug(f"Running OpenAI model {model_id} with max response tokens {max_response_tokens}")

        try:
            # Make the API call with retry logic
            responses = await self._call_openai_api(params)
        except Exception as e:
            logger.error(f"Failed to get response from {model_id}: {str(e)}")
            raise
        
        return responses
    
    async def run_model(self, provider: str, model_id: str, max_response_tokens: Optional[int] = None, num_runs: int = 1) -> List[str]:
        if provider == "anthropic":
            return await self.run_anthropic_model(model_id, num_runs, max_response_tokens)
        elif provider == "openai":
            return await self.run_openai_model(model_id, num_runs, max_response_tokens)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    async def grade_response(self, response: str, eval: ScenarioEval) -> List[ResponseGrading]:
        # Use gather with return_exceptions=True to handle grading errors gracefully
        grades = await asyncio.gather(
            *[grader.grade(response, eval) for grader in self.graders],
            return_exceptions=True
        )
        
        # Process results, logging errors but returning None for failed grades
        processed_grades = []
        for i, grade in enumerate(grades):
            if isinstance(grade, Exception):
                logger.error(f"Grading error for grader {i}: {str(grade)}")
                processed_grades.append(None)
            else:
                processed_grades.append(grade)
        
        return processed_grades
    
    async def run_eval(self, model: ModelSpec, num_runs: int = 1) -> EvalResult:
        responses = await self.run_model(model.provider, model.model_id, num_runs)
        
        if self.graders:
            grades = await asyncio.gather(*[self.grade_response(response, self.eval) for response in responses])
        else:
            grades = [[None] for _ in responses]
            
        return EvalResult(model, num_runs, responses, grades)