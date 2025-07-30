"""
Progressive results saving for evaluation outputs.

This module provides classes for saving evaluation results in different formats
(JSON, JSONL) with batching and background flushing for performance.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from prefill_evals.models import AgentMessage, ModelSpec
from prefill_evals.evaluator import EvalResult, ResponseGrading
from prefill_evals.config import EvalConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Serialization Functions
# =============================================================================

def serialize_agent_message(message: AgentMessage) -> Dict[str, Any]:
    """Convert AgentMessage to JSON-serializable dict."""
    return message.to_dict()


def serialize_response_grading(grading: Optional[ResponseGrading]) -> Optional[Dict[str, Any]]:
    """Convert ResponseGrading to JSON-serializable dict."""
    if grading is None:
        return None
    return {
        "score": grading.score,
        "data": grading.data or {}
    }


def serialize_model_spec(model: ModelSpec) -> Dict[str, Any]:
    """Convert ModelSpec to JSON-serializable dict."""
    data = {
        "provider": model.provider,
        "model_id": model.model_id
    }
    
    if model.max_response_tokens is not None:
        data["max_response_tokens"] = model.max_response_tokens
    
    if model.alias is not None:
        data["alias"] = model.alias
    
    return data


def serialize_eval_result(result: EvalResult, scenario_path: Path) -> Dict[str, Any]:
    """Convert EvalResult to JSON-serializable dict with scenario info."""
    # Serialize responses - each response is a list of messages
    serialized_responses = []
    for response_messages in result.responses:
        serialized_messages = [serialize_agent_message(msg) for msg in response_messages]
        serialized_responses.append({
            "messages": serialized_messages,
            "rendered": "\n".join(msg.render() for msg in response_messages)
        })
    
    return {
        "scenario_path": str(scenario_path),
        "model": serialize_model_spec(result.model),
        "num_runs": result.num_runs,   
        "responses": serialized_responses,
        "grades": [
            [serialize_response_grading(grade) for grade in response_grades]
            for response_grades in result.grades
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def create_error_result(model: ModelSpec, error: Exception, scenario_path: Path) -> Dict[str, Any]:
    """Create an error result entry."""
    return {
        "scenario_path": str(scenario_path),
        "model": serialize_model_spec(model),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# =============================================================================
# Base Results Saver
# =============================================================================

class BaseResultsSaver(ABC):
    """Abstract base class for all results savers."""
    
    def __init__(self, output_path: Path, config: Optional[EvalConfig] = None,
                 batch_size: int = 10, flush_interval: float = 5.0):
        """
        Initialize the results saver.
        
        Args:
            output_path: Path to save results
            config: Optional evaluation configuration
            batch_size: Number of results to batch before flushing
            flush_interval: Time interval between automatic flushes
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Async coordination
        self.lock = asyncio.Lock()
        self.pending_results = []
        self.last_flush_time = time.time()
        self.flush_task = None
        self.should_stop = False
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self._background_flush())
    
    async def _background_flush(self):
        """Background task that periodically flushes pending results."""
        while not self.should_stop:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_if_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")
    
    async def _flush_if_needed(self):
        """Check if flush is needed based on batch size or time."""
        async with self.lock:
            current_time = time.time()
            time_since_flush = current_time - self.last_flush_time
            
            if (len(self.pending_results) >= self.batch_size or 
                (len(self.pending_results) > 0 and time_since_flush >= self.flush_interval)):
                await self._flush_to_disk()
    
    @abstractmethod
    async def _flush_to_disk(self):
        """Flush pending results to disk. Must be called with lock held."""
        pass
    
    async def save_result(self, result: EvalResult, scenario_path: Path):
        """Save a single evaluation result."""
        async with self.lock:
            serialized = serialize_eval_result(result, scenario_path)
            self.pending_results.append(serialized)
            
            if len(self.pending_results) >= self.batch_size:
                await self._flush_to_disk()
    
    async def save_error(self, error_result: Dict[str, Any]):
        """Save an error result."""
        async with self.lock:
            self.pending_results.append(error_result)
            
            if len(self.pending_results) >= self.batch_size:
                await self._flush_to_disk()
    
    async def save_batch(self, results_with_paths: List[Tuple[EvalResult, Path]]):
        """Save multiple results at once."""
        async with self.lock:
            for result, scenario_path in results_with_paths:
                serialized = serialize_eval_result(result, scenario_path)
                self.pending_results.append(serialized)
            
            # Always flush after batch save
            await self._flush_to_disk()
    
    async def finalize(self):
        """Ensure all pending results are written and clean up."""
        self.should_stop = True
        
        # Cancel background task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        async with self.lock:
            await self._flush_to_disk()


# =============================================================================
# JSON Results Saver
# =============================================================================

class JSONResultsSaver(BaseResultsSaver):
    """Save results in JSON format with metadata."""
    
    def __init__(self, output_path: Path, config: Optional[EvalConfig] = None,
                 batch_size: int = 10, flush_interval: float = 5.0):
        super().__init__(output_path, config, batch_size, flush_interval)
        
        # Initialize data structure
        self.data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "config": self._serialize_config(config) if config else {}
            },
            "results": []
        }
        
        # Load existing results if file exists
        if self.output_path.exists():
            self._load_existing_results()
    
    def _serialize_config(self, config: EvalConfig) -> Dict[str, Any]:
        """Serialize evaluation configuration for metadata."""
        return {
            "models": [serialize_model_spec(m) for m in config.models],
            "runs_per_model": config.runs_per_model,
            "num_scenarios": len(config.scenarios),
            "autograders": [g.name for g in config.autograders]
        }
    
    def _load_existing_results(self):
        """Load existing results from file."""
        try:
            with open(self.output_path, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict) and "results" in existing_data:
                    self.data = existing_data
                    print(f"Loaded {len(self.data['results'])} existing results from {self.output_path}")
                elif isinstance(existing_data, list):
                    # Old format - convert to new format
                    self.data["results"] = existing_data
                    print(f"Converted {len(existing_data)} existing results to new format")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    async def _flush_to_disk(self):
        """Flush pending results to disk with atomic write."""
        if not self.pending_results:
            return
        
        # Add pending results to data
        self.data["results"].extend(self.pending_results)
        self.data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Generate unique temp file
        temp_dir = self.output_path.parent
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir, text=True)
        
        try:
            # Write to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename
            Path(temp_path).replace(self.output_path)
            
            # Clear pending results
            self.pending_results = []
            self.last_flush_time = time.time()
            
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e


# =============================================================================
# JSONL Results Saver
# =============================================================================

class JSONLResultsSaver(BaseResultsSaver):
    """Save results in JSONL format (one JSON object per line)."""
    
    def __init__(self, output_path: Path, config: Optional[EvalConfig] = None,
                 batch_size: int = 10, flush_interval: float = 5.0):
        super().__init__(output_path, config, batch_size, flush_interval)
        
        # For JSONL, we append to existing file
        if self.output_path.exists():
            print(f"Will append to existing JSONL file: {self.output_path}")
    
    async def _flush_to_disk(self):
        """Flush pending results by appending to JSONL file."""
        if not self.pending_results:
            return
        
        try:
            # Append to JSONL file
            with open(self.output_path, 'a') as f:
                for result in self.pending_results:
                    json.dump(result, f)
                    f.write('\n')
            
            # Clear pending results
            self.pending_results = []
            self.last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing to JSONL: {e}")
            raise


# =============================================================================
# Factory Function
# =============================================================================

def create_results_saver(output_path: Path, config: Optional[EvalConfig] = None, 
                        format: str = "json", **kwargs) -> BaseResultsSaver:
    """
    Create a results saver instance based on the specified format.
    
    Args:
        output_path: Path to save results
        config: Optional evaluation configuration
        format: Output format ("json" or "jsonl")
        **kwargs: Additional arguments passed to saver constructor
    
    Returns:
        Results saver instance
    
    Raises:
        ValueError: If format is not supported
    """
    if format == "json":
        return JSONResultsSaver(output_path, config, **kwargs)
    elif format == "jsonl":
        return JSONLResultsSaver(output_path, config, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {format}. Choose 'json' or 'jsonl'.")