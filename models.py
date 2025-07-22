#!/usr/bin/env python3
"""
Agent message datatype for structured transcript representation.

This module provides typed message classes to replace untyped dictionary format
for better type safety, validation, and manipulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
import json

class AgentMessage(ABC):
    """Abstract base class for all message types in a transcript."""
    
    @abstractmethod
    def render(self, max_length: int = 100) -> str:
        """Render the message for display with truncation if needed."""
        pass


@dataclass
class TextMessage(AgentMessage):
    """Regular text message from user, agent, system, or tool."""
    role: str  # "user" | "assistant" | "system" | "tool" 
    content: str
    
    def render(self, max_length: int = 100) -> str:
        """Render text message with truncation."""
        content = self.content.replace('\n', ' ').strip()
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return f"TextMessage(role={repr(self.role)}, content={repr(content)})"


@dataclass
class ToolCall(AgentMessage):
    """Tool call made by the agent."""
    name: str
    params: Dict[str, str]
    id: Optional[str] = None
    
    def __post_init__(self):
        # Generate ID if not provided
        if self.id is None:
            # This will be set by the parser based on context
            self.id = f"call_{id(self)}"
    
    def render(self, max_length: int = 100) -> str:
        """Render tool call with parameter truncation."""
        params_parts = []
        for k, v in self.params.items():
            v_str = str(v)
            if len(v_str) > 50:  # Truncate long parameter values
                v_str = v_str[:50] + "..."
            params_parts.append(f"{k}={repr(v_str)}")
        args_preview = ', '.join(params_parts)
        return f"{self.name}({args_preview})"


@dataclass
class ToolResult(AgentMessage):
    """Result from a tool call."""
    tool_call_id: str
    content: str
    
    def render(self, max_length: int = 100) -> str:
        """Render tool result with truncation."""
        content = self.content.replace('\n', ' ').strip()
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return f"ToolResult(content={repr(content)})"


def to_openai_format(messages: List[AgentMessage]) -> List[Dict[str, Any]]:
    """
    Convert AgentMessage list to OpenAI API format.
    
    OpenAI format:
    - System messages: {"role": "system", "content": "..."}
    - User messages: {"role": "user", "content": "..."}
    - Assistant with tools: {"role": "assistant", "content": "...", "tool_calls": [...]}
    - Tool results: {"role": "tool", "tool_call_id": "...", "content": "..."}
    
    Args:
        messages: List of AgentMessage objects
        system_prompt: Optional system prompt
        
    Returns:
        List of message dicts in OpenAI format
    """
    result = []
    
    for msg in messages:
        if isinstance(msg, TextMessage):
            result.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, ToolCall):
            if len(result) == 0 or result[-1]["role"] != "assistant":
                result.append({"role": "assistant", "content": ""})
            previous_msg = result.pop(-1)
            if "tool_calls" not in previous_msg:
                previous_msg["tool_calls"] = []
            previous_msg["tool_calls"].append({
                "id": msg.id,
                "type": "function",
                "function": {"name": msg.name, "arguments": json.dumps(msg.params)}
            })
            result.append(previous_msg)
        elif isinstance(msg, ToolResult):
            result.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
            
    return result

def to_anthropic_format(messages: List[AgentMessage]) -> List[Dict[str, Any]]:
    """
    Convert AgentMessage list to Anthropic API format.
    
    Anthropic doesn't support tool messages, so we convert them to text.
    
    Args:
        messages: List of AgentMessage objects
        system_prompt: Optional system prompt (returned separately for Anthropic)
        
    Returns:
        List of message dicts in Anthropic format
    """
    result = []
    
    for msg in messages:
        if isinstance(msg, TextMessage):
            result.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, ToolCall):
            if len(result) == 0 or result[-1]["role"] != "assistant":
                result.append({"role": "assistant", "content": ""})
            previous_msg = result.pop(-1)
            if isinstance(previous_msg["content"], str):
                previous_msg["content"] = [{
                    "type": "text",
                    "text": previous_msg["content"]
                }]
            previous_msg["content"].append({
                "id": msg.id,
                "type": "tool_use",
                "name": msg.name,
                "input": msg.params
            })
            result.append(previous_msg)
        elif isinstance(msg, ToolResult):
            result.append({"role": "user", "content": {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content
            }})
    
    return result

@dataclass
class ToolParameter:
    name: str
    description: str
    type: str
    optional: bool

@dataclass
class ToolDefinition:
    """Represents a tool definition."""
    name: str
    description: Optional[str] = None
    parameters: List[ToolParameter] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_name = param["name"]
            param_type = param.get("type", "string")
            
            # Map types to JSON Schema types
            type_mapping = {
                "string": "string",
                "integer": "integer",
                "number": "number",
                "boolean": "boolean",
                "array": "array",
                "object": "object"
            }
            
            properties[param_name] = {
                "type": type_mapping.get(param_type, "string"),
                "description": param.get("description", "")
            }
            
            if not param.get("optional", False):
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"Tool: {self.name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_name = param["name"]
            param_type = param.get("type", "string")
            
            # Map types to JSON Schema types
            type_mapping = {
                "string": "string",
                "integer": "integer",
                "number": "number",
                "boolean": "boolean",
                "array": "array",
                "object": "object"
            }
            
            properties[param_name] = {
                "type": type_mapping.get(param_type, "string"),
                "description": param.get("description", "")
            }
            
            if not param.get("optional", False):
                required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description or f"Tool: {self.name}",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

@dataclass
class ScenarioEval:
    messages: List[AgentMessage]
    tools: List[ToolDefinition]
    system: Optional[str] = None

    # extra items for e.g. graders
    extra_items: Optional[Dict[str, str]] = None

def validate_scenario(scenario: ScenarioEval) -> List[str]:
    """
    Validate a list of AgentMessage objects for structural correctness.
    
    Checks:
    1. Message sequence follows user->agent alternation
    2. Tool calls have matching tool results
    3. Tool calls reference defined tools (if tools provided)
    4. Transcript ends with a user message
    
    Args:
        messages: List of AgentMessage objects
        tools: Optional list of tool definitions to validate against
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    messages = scenario.messages
    tools = scenario.tools
    
    if not messages:
        errors.append("ERROR: The transcript is empty. You need to include messages in the conversation.")
        return errors
    
    # Helper function to get message preview
    def get_message_preview(msg: AgentMessage, max_length: int = 150) -> str:
        """Get a preview of message content for error reporting."""
        if msg is None:
            return "None"
        return msg.render(max_length)
    
    # Helper to describe message position
    def describe_position(i: int, messages: List[AgentMessage]) -> str:
        """Describe the position of a message in the transcript."""
        msg = messages[i]
        position_desc = f"Message {i+1}"
        
        # Add context about what type of message it is
        if isinstance(msg, TextMessage):
            if msg.role == "assistant":
                # Count which agent response this is
                agent_count = sum(1 for j in range(i+1) 
                                if isinstance(messages[j], TextMessage) and messages[j].role == "assistant")
                position_desc += f" (Agent response #{agent_count})"
            elif msg.role == 'user':
                # Count which user message this is
                user_count = sum(1 for j in range(i+1) 
                               if isinstance(messages[j], TextMessage) and messages[j].role == 'user')
                position_desc += f" (User message #{user_count})"
        elif isinstance(msg, ToolCall):
            position_desc += f" (Tool call: {msg.name})"
        elif isinstance(msg, ToolResult):
            position_desc += f" (Tool result for call ID: {msg.tool_call_id})"
            
        return position_desc
    
    # Check message sequence
    expected_role = 'user'  # First message should be user
    pending_tool_calls = {}  # Map tool_call_id to ToolCall object
    
    for i, msg in enumerate(messages):
        if isinstance(msg, TextMessage):
            actual_role = msg.role
            
            if actual_role == 'user':
                if expected_role != 'user':
                    prev_msg = messages[i-1] if i > 0 else None
                    error = f"ERROR: Unexpected user message at {describe_position(i, messages)}\n"
                    error += f"  Context: After \"{get_message_preview(prev_msg)}\"\n"
                    error += f"  Problem: Expected an agent response, but found another user message\n"
                    error += f"  Current: \"{get_message_preview(msg)}\"\n"
                    error += f"  Fix: Ensure the agent responds before the user speaks again"
                    errors.append(error)
                
                if pending_tool_calls:
                    # Find the first pending tool call
                    first_pending = next(iter(pending_tool_calls.values()))
                    error = f"ERROR: Unresolved tool call before {describe_position(i, messages)}\n"
                    error += f"  Problem: Tool call '{first_pending.name}' (ID: {first_pending.id}) was made but never received a result\n"
                    error += f"  Context: The agent cannot proceed to the next user message with pending tool calls\n"
                    error += f"  Fix: Add <agentml:tool_result>...</agentml:tool_result> after the tool call"
                    errors.append(error)
                
                expected_role = "assistant"
                
            elif actual_role == "assistant":
                if expected_role != "assistant":
                    prev_msg = messages[i-1] if i > 0 else None
                    error = f"ERROR: Unexpected agent message at {describe_position(i, messages)}\n"
                    error += f"  Context: After \"{get_message_preview(prev_msg)}\"\n"
                    error += f"  Problem: Expected a user message, but found another agent response\n"
                    error += f"  Current: \"{get_message_preview(msg)}\"\n"
                    error += f"  Fix: Add a user message before this agent response"
                    errors.append(error)
                
                # Agent messages don't change expected role until we see what follows
                
            elif actual_role == 'system':
                # System messages are allowed anywhere
                pass
                
        elif isinstance(msg, ToolCall):
            # Tool calls should follow agent text (or another tool call)
            pending_tool_calls[msg.id] = msg
            expected_role = 'tool'  # Expect tool result next
            
        elif isinstance(msg, ToolResult):
            if msg.tool_call_id not in pending_tool_calls:
                error = f"ERROR: Orphaned tool result at {describe_position(i, messages)}\n"
                error += f"  Problem: Tool result references call ID '{msg.tool_call_id}' which doesn't exist\n"
                error += f"  Content: \"{get_message_preview(msg)}\"\n"
                if pending_tool_calls:
                    existing_ids = list(pending_tool_calls.keys())
                    error += f"  Known tool call IDs: {', '.join(existing_ids)}\n"
                else:
                    error += f"  Note: No tool calls have been made yet\n"
                error += f"  Fix: Ensure tool results match their corresponding tool call IDs"
                errors.append(error)
            else:
                # Remove from pending
                del pending_tool_calls[msg.tool_call_id]
                
            # After tool result, we might have more agent text or user message
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, TextMessage) and next_msg.role == 'user':
                    expected_role = 'user'
                else:
                    expected_role = "assistant"
    
    # Check for pending tool calls
    if pending_tool_calls:
        num_pending = len(pending_tool_calls)
        tool_calls_str = ", ".join(f"'{tc.name}' (ID: {tc.id})" for tc in pending_tool_calls.values())
        error = f"ERROR: {num_pending} unresolved tool call{'s' if num_pending > 1 else ''} at end of transcript\n"
        error += f"  Pending: {tool_calls_str}\n"
        error += f"  Problem: All tool calls must receive results before the transcript ends\n"
        error += f"  Fix: Add <agentml:tool_result>...</agentml:tool_result> for each pending tool call"
        errors.append(error)
    
    # Check that transcript ends with user message
    if messages:
        last_msg = messages[-1]
        if not (isinstance(last_msg, TextMessage) and last_msg.role == 'user'):
            error = f"ERROR: Invalid transcript ending at {describe_position(len(messages)-1, messages)}\n"
            error += f"  Current ending: \"{get_message_preview(last_msg)}\"\n"
            error += f"  Problem: Transcripts must end with a user message\n"
            error += f"  Fix: Add a final 'User:' message after the last agent response"
            errors.append(error)
    
    # Validate tool calls against definitions if provided
    if tools:
        tool_names = {tool.name for tool in tools} if hasattr(tools[0], 'name') else set(tools)
        
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolCall):
                if msg.name not in tool_names:
                    error = f"ERROR: Undefined tool at {describe_position(i, messages)}\n"
                    error += f"  Tool name: '{msg.name}'\n"
                    # Show parameters with truncation
                    if msg.params:
                        params_str = []
                        for k, v in list(msg.params.items())[:3]:
                            v_str = str(v)
                            if len(v_str) > 50:
                                v_str = v_str[:50] + "..."
                            params_str.append(f"{k}={repr(v_str)}")
                        if len(msg.params) > 3:
                            params_str.append("...")
                        error += f"  Parameters: {', '.join(params_str)}\n"
                    error += f"  Available tools: {', '.join(sorted(tool_names))}\n"
                    # Suggest similar tool names if any
                    similar = [t for t in tool_names if msg.name.lower() in t.lower() or t.lower() in msg.name.lower()]
                    if similar:
                        error += f"  Did you mean: {', '.join(similar)}?\n"
                    error += f"  Fix: Use one of the available tools or define '{msg.name}' in the <tools> section"
                    errors.append(error)
    
    return errors