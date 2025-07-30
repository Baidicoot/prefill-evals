#!/usr/bin/env python3
"""
Model-based grading for scenario and response evaluation.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from prefill_evals.models import AgentMessage, TextMessage, ToolCall, ToolResult

# Add safety-tooling to path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt


def parse_grader_prompt(content: str) -> Dict[str, Optional[str]]:
    """
    Parse a grader prompt file into system and user prompts.
    
    Args:
        content: Raw prompt content
        
    Returns:
        Dict with 'system' and 'user' keys
    """
    # Initialize result
    result = {
        "system": None,
        "user": None
    }
    
    # Split by sections
    lines = content.strip().split('\n')
    current_section = None
    current_content = []
    
    headers = ["System:", "User:"]
    
    for line in lines:
        # Check if this line is a section header
        line_starts_with_header = any(line.strip().startswith(header) for header in headers)
        if line_starts_with_header:
            # Save previous section if exists
            if current_section and current_content:
                content_text = '\n'.join(current_content).strip()
                if current_section == "System:":
                    result["system"] = content_text
                elif current_section == "User:":
                    result["user"] = content_text
            
            # Start new section
            current_section = next(header for header in headers if line.strip().startswith(header))
            current_content = [line.strip()[len(current_section):].strip()]
        else:
            # Add line to current section
            current_content.append(line)
    
    # Don't forget the last section
    if current_section and current_content:
        content_text = '\n'.join(current_content).strip()
        if current_section == "System:":
            result["system"] = content_text
        elif current_section == "User:":
            result["user"] = content_text
    
    # If no headers found, treat entire content as user prompt
    if result["system"] is None and result["user"] is None:
        result["user"] = content.strip()
    
    return result


def render_scenario_to_text(scenario: Dict) -> str:
    """
    Render a parsed scenario dict to text format.
    
    Args:
        scenario: Parsed scenario dict with 'system' and 'messages'
        
    Returns:
        Formatted text representation
    """
    from prefill_evals.models import ScenarioEval
    from prefill_evals.parser import render_transcript
    
    # Convert dict to ScenarioEval if needed
    if isinstance(scenario, dict):
        # Create a ScenarioEval object from the dict
        scenario_obj = ScenarioEval(
            system=scenario.get("system"),
            messages=scenario.get("messages", []),
            tools=scenario.get("tools", []),
            extra_items=scenario.get("extra_items", {})
        )
    else:
        scenario_obj = scenario
    
    # Use the parser's render_transcript function
    return render_transcript(scenario_obj)


class ModelBasedGrader:
    """Grade responses/transcripts using another model."""
    
    def __init__(self, grader_model: str, prompt_file: Path, name: str):
        """
        Initialize the grader.
        
        Args:
            grader_model: Model ID to use for grading
            prompt_file: Path to grading prompt template
            name: Name of this grader (e.g., "hacking", "naturalness")
        """
        self.model = grader_model
        self.prompt_file = Path(prompt_file)
        self.name = name
        
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Grader prompt file not found: {self.prompt_file}")
        
        # Load and parse prompt at initialization
        with open(self.prompt_file, 'r') as f:
            prompt_content = f.read()
        
        self.parsed_prompt = parse_grader_prompt(prompt_content)
    
    async def grade(
        self, 
        api: InferenceAPI,
        scenario: Dict, 
        response: Optional[str] = None,
        input_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Grade a transcript or response.
        
        Args:
            api: InferenceAPI instance
            scenario: Parsed scenario dict with 'system' and 'messages'
            response: Optional model response to grade (for response graders)
            input_data: Optional additional data (e.g., target)
            
        Returns:
            Dict with 'grade' (int) and 'feedback' (str)
        """
        # Render scenario to text
        transcript = render_scenario_to_text(scenario)
        
        # Build format kwargs
        format_kwargs = {"transcript": transcript}
        if response is not None:
            format_kwargs["response"] = response
        if input_data:
            format_kwargs.update(input_data)
        
        # Format both system and user prompts
        messages = []
        
        # Add system prompt if present
        if self.parsed_prompt["system"]:
            try:
                system_content = self.parsed_prompt["system"].format(**format_kwargs)
                messages.append(ChatMessage(role=MessageRole.system, content=system_content))
            except KeyError as e:
                raise ValueError(f"Missing required field in system prompt template: {e}")
        
        # Add user prompt
        if self.parsed_prompt["user"]:
            try:
                user_content = self.parsed_prompt["user"].format(**format_kwargs)
                messages.append(ChatMessage(role=MessageRole.user, content=user_content))
            except KeyError as e:
                raise ValueError(f"Missing required field in user prompt template: {e}")
        else:
            raise ValueError("No user prompt found in grader prompt file")
        
        # Create prompt
        prompt = Prompt(messages=messages)
        
        # Get grading
        responses = await api(model_id=self.model, prompt=prompt, n=1)
        grading_response = responses[0].completion
        
        # Extract and parse grading
        return self._extract_grading(grading_response)
    
    def _extract_grading(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse grading from <grading>...</grading> tags.
        
        Expected format:
        <grading>
        Grade: 8       # or "8/10"
        Feedback: Detailed feedback text
        </grading>
        
        Returns:
            Dict with 'grade' (int) and 'feedback' (str)
        """
        match = re.search(r'<grading>(.*?)</grading>', response, re.DOTALL)
        if not match:
            raise ValueError(f"No <grading> tags found in response from {self.name} grader")
        
        content = match.group(1).strip()
        
        # Extract grade
        grade_match = re.search(r'Grade:\s*(\d+)(?:/\d+)?', content)
        if not grade_match:
            raise ValueError(f"No valid Grade found in {self.name} grader response")
        grade = int(grade_match.group(1))
        
        # Extract feedback
        feedback_match = re.search(r'Feedback:\s*(.+)', content, re.DOTALL)
        if not feedback_match:
            raise ValueError(f"No Feedback found in {self.name} grader response")
        feedback = feedback_match.group(1).strip()
        
        return {
            "grade": grade,
            "feedback": feedback
        }


def format_feedback(
    response_grades: Dict[str, Dict[str, Dict[str, Any]]], 
    transcript_grades: Dict[str, Dict[str, Any]]
) -> str:
    """
    Format all grading feedback into a single string for iteration.
    
    Args:
        response_grades: Dict of grader_name -> model_alias -> {grade, feedback}
        transcript_grades: Dict of grader_name -> {grade, feedback}
        
    Returns:
        Formatted feedback string
    """
    feedback_parts = []
    
    # Format transcript feedback
    if transcript_grades:
        feedback_parts.append("=== Transcript Feedback ===")
        for grader_name, grade_data in transcript_grades.items():
            grade = grade_data["grade"]
            feedback = grade_data["feedback"]
            feedback_parts.append(f"\n[{grader_name}] Grade: {grade}/10\n{feedback}")
    
    # Format response feedback
    if response_grades:
        feedback_parts.append("\n\n=== Model Response Feedback ===")
        for grader_name, model_feedbacks in response_grades.items():
            for model_alias, grade_data in model_feedbacks.items():
                grade = grade_data["grade"]
                feedback = grade_data["feedback"]
                feedback_parts.append(f"\n[{grader_name} on {model_alias}] Grade: {grade}/10\n{feedback}")
    
    feedback_parts.append("\n\n=== End of Feedback ===")
    
    return '\n'.join(feedback_parts)