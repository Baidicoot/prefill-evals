#!/usr/bin/env python3
"""
Model-based feedback providers for generated scenarios.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import sys
import os

# Add safety-tooling to path if it's a sibling directory
safety_tooling_path = Path(__file__).parent.parent / "safety-tooling"
if safety_tooling_path.exists():
    sys.path.insert(0, str(safety_tooling_path))

from prefill_evals.models import ScenarioEval

# Import safety-tooling components  
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils


class ScenarioFeedbackProvider(ABC):
    """Base class for providing feedback on generated scenarios."""
    
    @abstractmethod
    async def provide_feedback(self, scenario: ScenarioEval) -> Dict[str, Any]:
        """
        Provide feedback on a generated scenario.
        
        Args:
            scenario: The generated scenario to evaluate
            
        Returns:
            Dict with 'feedback' (str) and optionally 'score' (float)
        """
        pass


class ModelBasedScenarioFeedback(ScenarioFeedbackProvider):
    """Provide feedback on scenarios using a language model."""
    
    def __init__(self, name: str, model_config: Dict[str, str], template_file: Path, inference_api: Optional[InferenceAPI] = None):
        """
        Initialize the feedback provider.
        
        Args:
            name: Name of this feedback provider
            model_config: Dict with 'provider' and 'model_id'
            template_file: Path to feedback prompt template
            inference_api: Optional InferenceAPI instance (creates one if not provided)
        """
        self.name = name
        self.model_config = model_config
        self.template_file = Path(template_file)
        
        if not self.template_file.exists():
            raise FileNotFoundError(f"Feedback template file not found: {self.template_file}")
        
        # Setup InferenceAPI if not provided
        if inference_api is None:
            utils.setup_environment()
            cache_dir = Path(os.environ.get("CACHE_DIR", ".cache"))
            self.inference_api = InferenceAPI(cache_dir=cache_dir)
        else:
            self.inference_api = inference_api
        
        # Load template at initialization
        with open(self.template_file, 'r') as f:
            self.template = f.read()
    
    async def provide_feedback(self, scenario: ScenarioEval) -> Dict[str, Any]:
        """
        Get feedback on a scenario from the model.
        
        Args:
            scenario: The generated scenario to evaluate
            
        Returns:
            Dict with 'feedback' (str) and optionally 'score' (float)
        """
        # Format the template with scenario data
        format_kwargs = {
            'transcript': self._render_transcript(scenario),
            'tools': self._render_tools(scenario.tools) if scenario.tools else "No tools defined",
        }
        
        # Add extra items if present
        if scenario.extra_items:
            for item_name, item_content in scenario.extra_items.items():
                format_kwargs[item_name] = item_content
        
        # Format the prompt
        try:
            prompt = self.template.format(**format_kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required field in feedback template: {e}")
        
        # Get feedback from model
        response = await self._call_model(prompt)
        
        # Extract and parse feedback
        return self._extract_feedback(response)
    
    def _render_transcript(self, scenario: ScenarioEval) -> str:
        """Render scenario transcript in XML format."""
        from prefill_evals.parser import render_transcript
        return render_transcript(scenario)
    
    def _render_tools(self, tools: List[Any]) -> str:
        """Render tool definitions."""
        if not tools:
            return ""
        
        parts = []
        for tool in tools:
            parts.append(f"Tool: {tool.name}")
            if tool.description:
                parts.append(f"  Description: {tool.description}")
            if tool.parameters:
                parts.append("  Parameters:")
                for param in tool.parameters:
                    opt_str = " (optional)" if param.optional else ""
                    parts.append(f"    - {param.name} ({param.type}){opt_str}: {param.description}")
        
        return "\n".join(parts)
    
    async def _call_model(self, prompt: str) -> str:
        """
        Call the model to get feedback using InferenceAPI.
        """
        # Create prompt with single user message
        inference_prompt = Prompt(messages=[
            ChatMessage(content=prompt, role=MessageRole.user)
        ])
        
        # Call the model
        responses = await self.inference_api(
            model_id=self.model_config['model_id'],
            prompt=inference_prompt,
            max_tokens=2048,
            force_provider=self.model_config['provider'] if self.model_config['provider'] in ['anthropic', 'openai'] else None,
        )
        
        return responses[0].completion
    
    def _extract_feedback(self, response: str) -> Dict[str, Any]:
        """
        Extract feedback from model response.
        
        Expected format:
        <feedback>
        ... feedback text ...
        </feedback>
        
        Optional:
        <score>8.5</score>
        """
        result = {}
        
        # Extract feedback
        feedback_match = re.search(r'<feedback>(.*?)</feedback>', response, re.DOTALL)
        if feedback_match:
            result['feedback'] = feedback_match.group(1).strip()
        else:
            # If no tags, use entire response as feedback
            result['feedback'] = response.strip()
        
        # Extract score if present
        score_match = re.search(r'<score>(\d+(?:\.\d+)?)</score>', response)
        if score_match:
            result['score'] = float(score_match.group(1))
        
        return result


def format_feedback_for_revision(feedback_results: List[Dict[str, Any]]) -> str:
    """
    Format feedback results into a string for the revision prompt.
    
    Args:
        feedback_results: List of dicts with provider name and feedback
        
    Returns:
        Formatted feedback string
    """
    if not feedback_results:
        return "No feedback provided."
    
    parts = ["=== Feedback on Generated Scenario ==="]
    
    for result in feedback_results:
        parts.append(f"\n[{result['provider_name']}]")
        if 'score' in result:
            parts.append(f"Score: {result['score']}")
        parts.append(result['feedback'])
        parts.append("")  # Empty line between providers
    
    parts.append("=== End of Feedback ===")
    
    return "\n".join(parts)