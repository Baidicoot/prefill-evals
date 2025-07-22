#!/usr/bin/env python3
"""
Model-based feedback providers for generated scenarios.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from prefill_evals.models import ScenarioEval


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
    
    def __init__(self, name: str, model_config: Dict[str, str], template_file: Path):
        """
        Initialize the feedback provider.
        
        Args:
            name: Name of this feedback provider
            model_config: Dict with 'provider' and 'model_id'
            template_file: Path to feedback prompt template
        """
        self.name = name
        self.model_config = model_config
        self.template_file = Path(template_file)
        
        if not self.template_file.exists():
            raise FileNotFoundError(f"Feedback template file not found: {self.template_file}")
        
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
        from prefill_evals.models import TextMessage, ToolCall, ToolResult
        
        parts = []
        
        # Add system prompt if present
        if scenario.system:
            parts.append(f"<system>\n{scenario.system}\n</system>")
        
        # Add messages
        for msg in scenario.messages:
            if isinstance(msg, TextMessage):
                if msg.role == 'user':
                    parts.append(f"<user>\n{msg.content}\n</user>")
                elif msg.role == 'assistant':
                    parts.append(f"<agent>\n{msg.content}\n</agent>")
            elif isinstance(msg, ToolCall):
                tool_parts = [f"<tool_call:{msg.name}>"]
                for param_name, param_value in msg.params.items():
                    tool_parts.append(f"  <argument:{param_name}>{param_value}</argument:{param_name}>")
                tool_parts.append(f"</tool_call:{msg.name}>")
                parts.append("\n".join(tool_parts))
            elif isinstance(msg, ToolResult):
                parts.append(f"<tool_result>\n{msg.content}\n</tool_result>")
        
        return "\n\n".join(parts)
    
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
        Call the model to get feedback.
        
        This is a placeholder - in practice, this would use the appropriate API
        based on self.model_config['provider'].
        """
        # Import the appropriate client based on provider
        if self.model_config['provider'] == 'anthropic':
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model=self.model_config['model_id'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return response.content[0].text
        elif self.model_config['provider'] == 'openai':
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model_config['model_id'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.model_config['provider']}")
    
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