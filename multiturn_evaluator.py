"""
Multi-turn evaluator that uses auto-auditors' ConversationRunner for prefill evaluations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Add auto-auditors paths
easy_orchestrator_path = Path(__file__).parent / "external" / "auto-auditors" / "easy-orchestrator"
auto_eval_gen_path = Path(__file__).parent / "external" / "auto-auditors" / "auto-eval-gen"
sys.path.insert(0, str(easy_orchestrator_path))
sys.path.insert(0, str(auto_eval_gen_path))

from prefill_evals.evaluator import Evaluator, EvalResult, ResponseGrader
from prefill_evals.models import (
    ScenarioEval, ModelSpec, AgentMessage, TextMessage, ToolCall, ToolResult
)

try:
    # Import from auto-auditors
    from conversation_runner import ConversationRunner
    from prefill_utils import get_last_turn_number
except ImportError as e:
    logger.error(f"Failed to import auto-auditors modules: {e}")
    logger.error(f"Make sure the auto-auditors submodule is properly initialized")
    raise


class MultiturnEvaluator(Evaluator):
    """
    Multi-turn evaluator using auto-auditors' ConversationRunner.
    
    This evaluator uses the run_from_prefill functionality to continue
    conversations from existing scenarios.
    """
    
    def __init__(
        self,
        eval: ScenarioEval,
        runs_per_model: int = 1,
        graders: List[ResponseGrader] = None,
        user_simulator_model: Optional[ModelSpec] = None,
        max_turns: int = 10,
        temperature: float = 0.7,
        evaluator_system_prompt: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        dotenv_path: Optional[Path] = None,
    ):
        super().__init__(graders=graders, cache_dir=cache_dir, dotenv_path=dotenv_path)
        
        self.eval = eval
        self.runs_per_model = runs_per_model
        self.user_simulator_model = user_simulator_model
        self.max_turns = max_turns
        self.temperature = temperature
        self.evaluator_system_prompt = evaluator_system_prompt
    
    def _convert_to_easy_orchestrator_format(self) -> Dict[str, Any]:
        """
        Convert ScenarioEval to easy-orchestrator transcript format.
        
        Returns:
            Dictionary with 'events' list in easy-orchestrator format
        """
        events = []
        turn_number = 0
        
        # Add system prompt if present
        if self.eval.system:
            events.append({
                "event": "target_system_prompt",
                "content": self.eval.system,
                "turn": turn_number
            })
        
        # Convert messages to events
        for msg in self.eval.messages:
            if isinstance(msg, TextMessage):
                if msg.role == "user":
                    turn_number += 1
                    events.append({
                        "event": "evaluator_message",
                        "content": msg.content,
                        "turn": turn_number
                    })
                elif msg.role == "assistant":
                    events.append({
                        "event": "target_message", 
                        "content": msg.content,
                        "turn": turn_number
                    })
                elif msg.role == "system" and not self.eval.system:
                    # Handle system messages in the message list
                    events.append({
                        "event": "target_system_prompt",
                        "content": msg.content,
                        "turn": 0
                    })
            elif isinstance(msg, ToolCall):
                # Convert to easy-orchestrator tool call format
                events.append({
                    "event": "tool_call",
                    "content": {
                        "tool_name": msg.name,
                        "arguments": msg.params
                    },
                    "turn": turn_number
                })
            elif isinstance(msg, ToolResult):
                # Convert to easy-orchestrator tool response format
                # Need to find the corresponding tool call to get the tool name
                tool_name = "unknown"
                for i in range(len(self.eval.messages) - 1, -1, -1):
                    if isinstance(self.eval.messages[i], ToolCall) and self.eval.messages[i].id == msg.tool_call_id:
                        tool_name = self.eval.messages[i].name
                        break
                
                events.append({
                    "event": "tool_response",
                    "content": {
                        "tool_name": tool_name,
                        "result": msg.content
                    },
                    "turn": turn_number
                })
        
        return {
            "events": events,
            "variation_id": "prefill_eval"
        }
    
    def _convert_from_easy_orchestrator_format(self, events: List[Dict[str, Any]]) -> List[AgentMessage]:
        """
        Convert easy-orchestrator events back to AgentMessage format.
        
        Args:
            events: List of event dictionaries from ConversationRunner
            
        Returns:
            List of AgentMessage objects
        """
        messages = []
        tool_call_map = {}  # Map tool calls by some identifier
        
        for event in events:
            event_type = event.get('event')
            content = event.get('content', '')
            
            if event_type == 'evaluator_message':
                messages.append(TextMessage(role="user", content=content))
            elif event_type == 'target_message':
                messages.append(TextMessage(role="assistant", content=content))
            elif event_type == 'tool_call':
                # Convert tool call format
                if isinstance(content, dict):
                    tool_call = ToolCall(
                        name=content.get('tool_name', 'unknown'),
                        params=content.get('arguments', {}),
                        id=f"call_{len(messages)}"  # Generate ID based on position
                    )
                    tool_call_map[content.get('tool_name')] = tool_call.id
                    messages.append(tool_call)
                else:
                    # Fallback for non-dict content
                    messages.append(TextMessage(role="assistant", content=str(content)))
            elif event_type == 'tool_response':
                # Convert tool response format
                if isinstance(content, dict):
                    tool_name = content.get('tool_name', 'unknown')
                    tool_call_id = tool_call_map.get(tool_name, f"call_unknown")
                    messages.append(ToolResult(
                        tool_call_id=tool_call_id,
                        content=content.get('result', '')
                    ))
                else:
                    # Fallback for non-dict content
                    messages.append(TextMessage(role="user", content=str(content)))
            # Skip system prompts as they're not part of the message flow
            
        return messages
    
    def _get_model_name(self, model: ModelSpec) -> str:
        """
        Convert ModelSpec to model name format expected by auto-auditors.
        
        Args:
            model: ModelSpec object
            
        Returns:
            Model name string
        """
        # Map common model names to auto-auditors format
        model_id = model.model_id
        
        # Handle Anthropic models
        if model.provider == "anthropic":
            if "claude-3-opus" in model_id:
                return "claude-opus-3"
            elif "claude-3-sonnet" in model_id:
                return "claude-sonnet-3.5"
            elif "claude-3-haiku" in model_id:
                return "claude-haiku-3"
            elif "claude-sonnet-4" in model_id:
                return "claude-sonnet-4"
            else:
                # Try to use as-is
                return model_id
        elif model.provider == "openai":
            # OpenAI models can typically be used as-is
            return model_id
        else:
            return model_id
    
    async def run_eval(self, model: ModelSpec, num_runs: int = 1) -> EvalResult:
        """
        Run multi-turn evaluation using ConversationRunner.
        
        Args:
            model: Model specification to evaluate
            num_runs: Number of runs to perform
            
        Returns:
            EvalResult containing generated messages
        """
        all_responses = []
        
        for run_idx in range(num_runs):
            try:
                # Convert scenario to easy-orchestrator format
                transcript = self._convert_to_easy_orchestrator_format()
                
                # Calculate how many more turns we can do
                last_turn = get_last_turn_number(transcript)
                absolute_max_turns = last_turn + self.max_turns
                
                # Create conversation runner
                evaluator_model_name = self._get_model_name(self.user_simulator_model) if self.user_simulator_model else "claude-haiku-3"
                target_model_name = self._get_model_name(model)
                
                # Use configured evaluator system prompt or default
                if self.evaluator_system_prompt:
                    evaluator_system_prompt = self.evaluator_system_prompt
                elif self.user_simulator_model:
                    evaluator_system_prompt = "You are simulating a user in a conversation. Generate natural follow-up messages based on the conversation so far."
                else:
                    # If no user simulator, we're not generating new user messages
                    evaluator_system_prompt = ""
                
                runner = ConversationRunner(
                    evaluator_model=evaluator_model_name,
                    target_model=target_model_name,
                    evaluator_system_prompt=evaluator_system_prompt,
                    target_system_prompt=self.eval.system or "",
                    max_turns=absolute_max_turns,
                    temperature=self.temperature,
                    max_tokens=model.max_response_tokens or 4000,
                    target_thinking_enabled=False,
                    evaluator_thinking_enabled=False
                )
                
                # Determine if we need a follow-up message
                follow_up_message = None
                if self.eval.messages and isinstance(self.eval.messages[-1], TextMessage) and self.eval.messages[-1].role == "user":
                    # The last message is already a user message, use it as follow-up
                    follow_up_message = self.eval.messages[-1].content
                    # Remove it from the transcript since it will be added by run_from_prefill
                    transcript['events'] = transcript['events'][:-1]
                
                # Run prefill evaluation
                result_transcript = await runner.run_from_prefill(
                    existing_transcript=transcript,
                    initial_message=follow_up_message,
                    truncate_to_last_target=False,  # We've already prepared the transcript
                    display_progress=False
                )
                
                # Extract only the new messages generated during this run
                original_event_count = len(transcript['events'])
                new_events = result_transcript['events'][original_event_count:]
                
                # Convert new events to messages
                generated_messages = self._convert_from_easy_orchestrator_format(new_events)
                all_responses.append(generated_messages)
                
            except Exception as e:
                logger.error(f"Failed to run multi-turn eval for {model.model_id} (run {run_idx + 1}/{num_runs}): {str(e)}")
                raise
        
        # Grade responses if graders are available
        if self.graders:
            grades = []
            for response_messages in all_responses:
                response_grades = []
                # Render messages to text for grading
                from prefill_evals.evaluator import render_messages
                response_text = render_messages(response_messages)
                for grader in self.graders:
                    try:
                        grade = await grader.grade(response_text, self.eval)
                        response_grades.append(grade)
                    except Exception as e:
                        logger.error(f"Grading error: {str(e)}")
                        response_grades.append(None)
                grades.append(response_grades)
        else:
            grades = [[None] for _ in all_responses]
        
        return EvalResult(
            model=model,
            num_runs=num_runs,
            responses=all_responses,
            grades=grades
        )
    
