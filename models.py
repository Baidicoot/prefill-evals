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

@dataclass
class ModelSpec:
    provider: str
    model_id: str
    max_response_tokens: Optional[int] = None
    alias: Optional[str] = None
    
    def get_short_name(self, max_length: int = 12) -> str:
        """Get a short, readable version of model names.
        
        Returns alias if set, otherwise:
        - For claude-* models: extract middle parts (e.g. claude-3-7-sonnet-latest -> 3-7-sonnet)
        - For OpenAI finetunes (ft:*): extract meaningful suffix (e.g. ft:...:name:id -> name or name:ckpt-step-N)
        - Otherwise: truncate to first 16 characters
        """
        # If alias is set, use it
        if self.alias:
            return self.alias
            
        # Otherwise use the shortening logic
        model_name = f"{self.provider}/{self.model_id}"
        
        # Remove provider prefix if present (e.g. "openai/gpt-4" -> "gpt-4")
        if "/" in model_name:
            parts = model_name.split("/", 1)
            if len(parts) == 2:
                model_name = parts[1]
        
        # Handle Claude models
        if model_name.startswith("claude-"):
            parts = model_name.split("-")
            if len(parts) > 2:
                # Remove first (claude) and last (version/date) parts
                return "-".join(parts[1:-1])
        
        # Handle OpenAI finetunes
        elif model_name.startswith("ft:"):
            parts = model_name.split(":")
            if len(parts) >= 2:
                # Check if last part looks like ckpt-step-N
                last_part = parts[-1]
                if last_part.startswith("ckpt-step-") and len(parts) >= 4:
                    # Get the meaningful name and checkpoint
                    name_part = parts[-3]
                    # Shorten ckpt-step-N to step-N
                    checkpoint = last_part.replace("ckpt-", "")
                    suffix = f":{checkpoint}"
                    # Truncate name part to fit in max_length chars total
                    max_name_len = max_length - len(suffix)
                    if len(name_part) > max_name_len:
                        name_part = name_part[:max_name_len]
                    return f"{name_part}{suffix}"
                elif len(parts) >= 3:
                    # Return second-last part (meaningful name when no checkpoint)
                    name_part = parts[-2]
                    # Truncate if too long
                    if len(name_part) > max_length:
                        return name_part[:max_length]
                    return name_part
        
        # Default: truncate to first 16 characters
        if len(model_name) > 16:
            return model_name[:16]
        return model_name

class AgentMessage(ABC):
    """Abstract base class for all message types in a transcript."""
    
    @abstractmethod
    def render(self, max_length: int = 100) -> str:
        """Render the message for display with truncation if needed."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to JSON-serializable dictionary."""
        pass
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AgentMessage':
        """Create AgentMessage from dictionary."""
        msg_type = data.get('type')
        if msg_type == 'text':
            return TextMessage.from_dict(data)
        elif msg_type == 'tool_call':
            return ToolCall.from_dict(data)
        elif msg_type == 'tool_result':
            return ToolResult.from_dict(data)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'type': 'text',
            'role': self.role,
            'content': self.content
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TextMessage':
        """Create TextMessage from dictionary."""
        return TextMessage(
            role=data['role'],
            content=data['content']
        )


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'type': 'tool_call',
            'name': self.name,
            'params': self.params,
            'id': self.id
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ToolCall':
        """Create ToolCall from dictionary."""
        return ToolCall(
            name=data['name'],
            params=data['params'],
            id=data.get('id')
        )


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'type': 'tool_result',
            'tool_call_id': self.tool_call_id,
            'content': self.content
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ToolResult':
        """Create ToolResult from dictionary."""
        return ToolResult(
            tool_call_id=data['tool_call_id'],
            content=data['content']
        )


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
                result.append({"role": "assistant", "content": []})
            previous_msg = result.pop(-1)
            if isinstance(previous_msg["content"], str):
                if previous_msg["content"]:  # Only add text block if content is non-empty
                    previous_msg["content"] = [{
                        "type": "text",
                        "text": previous_msg["content"]
                    }]
                else:
                    previous_msg["content"] = []
            elif not isinstance(previous_msg["content"], list):
                # Ensure content is a list
                previous_msg["content"] = []
            previous_msg["content"].append({
                "id": msg.id,
                "type": "tool_use",
                "name": msg.name,
                "input": msg.params
            })
            result.append(previous_msg)
        elif isinstance(msg, ToolResult):
            result.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content
            }]})
    
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
        
        if self.parameters:
            for param in self.parameters:
                param_name = param.name
                param_type = param.type
                
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
                    "description": param.description
                }
                
                if not param.optional:
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
        
        if self.parameters:
            for param in self.parameters:
                param_name = param.name
                param_type = param.type
                
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
                    "description": param.description
                }
                
                if not param.optional:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ScenarioEval to JSON-serializable dictionary."""
        result = {
            'messages': [msg.to_dict() for msg in self.messages],
            'tools': [{'name': tool.name, 'description': tool.description, 
                      'parameters': [{'name': p.name, 'description': p.description, 
                                    'type': p.type, 'optional': p.optional} 
                                   for p in (tool.parameters or [])]} 
                     for tool in self.tools],
        }
        if self.system is not None:
            result['system'] = self.system
        if self.extra_items is not None:
            result['extra_items'] = self.extra_items
        return result
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ScenarioEval':
        """Create ScenarioEval from dictionary."""
        messages = [AgentMessage.from_dict(msg) for msg in data['messages']]
        
        tools = []
        for tool_data in data.get('tools', []):
            parameters = []
            for param_data in tool_data.get('parameters', []):
                parameters.append(ToolParameter(
                    name=param_data['name'],
                    description=param_data['description'],
                    type=param_data['type'],
                    optional=param_data['optional']
                ))
            tools.append(ToolDefinition(
                name=tool_data['name'],
                description=tool_data.get('description'),
                parameters=parameters if parameters else None
            ))
        
        return ScenarioEval(
            messages=messages,
            tools=tools,
            system=data.get('system'),
            extra_items=data.get('extra_items')
        )

def serialize_messages(messages: List[AgentMessage]) -> List[Dict[str, Any]]:
    """Convert a list of AgentMessage objects to JSON-serializable list."""
    return [msg.to_dict() for msg in messages]


def deserialize_messages(data: List[Dict[str, Any]]) -> List[AgentMessage]:
    """Create a list of AgentMessage objects from JSON data."""
    return [AgentMessage.from_dict(msg_data) for msg_data in data]


def messages_to_text(messages: List[AgentMessage]) -> str:
    """
    Convert a list of AgentMessage objects to text transcript format.
    
    Format:
    - TextMessage with role "user" -> "User: {content}"
    - TextMessage with role "assistant" -> "Agent: {content}"
    - TextMessage with role "system" -> "System: {content}"
    - ToolCall -> "Agent: <agentml:function_calls>..."
    - ToolResult -> "<agentml:tool_result>..."
    """
    parts = []
    current_agent_content = []
    
    for msg in messages:
        if isinstance(msg, TextMessage):
            if msg.role == "user":
                # Flush any pending agent content
                if current_agent_content:
                    parts.append(f"Agent: {''.join(current_agent_content)}")
                    current_agent_content = []
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                current_agent_content.append(msg.content)
            elif msg.role == "system":
                # Flush any pending agent content
                if current_agent_content:
                    parts.append(f"Agent: {''.join(current_agent_content)}")
                    current_agent_content = []
                parts.append(f"System: {msg.content}")
        elif isinstance(msg, ToolCall):
            # Add tool call to agent content
            tool_call_text = f"\n<agentml:function_calls>\n<agentml:invoke name=\"{msg.name}\">\n"
            for param_name, param_value in msg.params.items():
                tool_call_text += f"<agentml:parameter name=\"{param_name}\">{param_value}</agentml:parameter>\n"
            tool_call_text += "</agentml:invoke>\n</agentml:function_calls>"
            current_agent_content.append(tool_call_text)
        elif isinstance(msg, ToolResult):
            # Add tool result
            current_agent_content.append(f"\n<agentml:tool_result>\n{msg.content}\n</agentml:tool_result>")
    
    # Flush any remaining agent content
    if current_agent_content:
        parts.append(f"Agent: {''.join(current_agent_content)}")
    
    return "\n\n".join(parts)


def validate_scenario(scenario: ScenarioEval, require_user_ending: bool = False) -> List[str]:
    """
    Validate a list of AgentMessage objects for structural correctness.
    
    Checks:
    1. Message sequence follows user->agent alternation
    2. Tool calls have matching tool results
    3. Tool calls reference defined tools (if tools provided)
    4. Transcript ends with a user message (only if require_user_ending=True)
    
    Args:
        scenario: ScenarioEval object to validate
        require_user_ending: If True, requires transcript to end with user message (for generation)
        
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
    
    # State machine for valid transitions
    # Key: current state, Value: list of valid next states
    transitions = {
        'start': ['user', 'system'],
        'system': ['user'],
        'user': ['assistant', 'tool_call'],
        'assistant': ['user', 'tool_call'],
        'tool_call': ['tool_result'],
        'tool_result': ['assistant', 'user', 'tool_call']
    }
    
    current_state = 'start'
    pending_tool_calls = {}  # Map tool_call_id to ToolCall object
    
    for i, msg in enumerate(messages):
        # Determine the message type for state machine
        if isinstance(msg, TextMessage):
            msg_type = msg.role  # 'user', 'assistant', or 'system'
        elif isinstance(msg, ToolCall):
            msg_type = 'tool_call'
        elif isinstance(msg, ToolResult):
            msg_type = 'tool_result'
        else:
            continue  # Skip unknown message types
        
        # Check if this transition is valid
        valid_next_states = transitions.get(current_state, [])
        if msg_type not in valid_next_states:
            prev_msg = messages[i-1] if i > 0 else None
            error = f"ERROR: Invalid message sequence at {describe_position(i, messages)}\n"
            error += f"  Context: After \"{get_message_preview(prev_msg)}\" (state: {current_state})\n"
            error += f"  Problem: Found {msg_type} message, but expected one of: {', '.join(valid_next_states)}\n"
            error += f"  Current: \"{get_message_preview(msg)}\"\n"
            errors.append(error)
        
        # Handle specific message types
        if isinstance(msg, TextMessage):
            if msg.role == 'user' and pending_tool_calls:
                # User message with unresolved tool calls
                first_pending = next(iter(pending_tool_calls.values()))
                error = f"ERROR: Unresolved tool call before {describe_position(i, messages)}\n"
                error += f"  Problem: Tool call '{first_pending.name}' (ID: {first_pending.id}) was made but never received a result\n"
                error += f"  Context: The agent cannot proceed to the next user message with pending tool calls\n"
                error += f"  Fix: Add <agentml:tool_result>...</agentml:tool_result> after the tool call"
                errors.append(error)
            
            # Update state
            current_state = msg.role
                
        elif isinstance(msg, ToolCall):
            # Track pending tool calls
            pending_tool_calls[msg.id] = msg
            current_state = 'tool_call'
            
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
            
            # Update state
            current_state = 'tool_result'
    
    # Check for pending tool calls
    if pending_tool_calls:
        num_pending = len(pending_tool_calls)
        tool_calls_str = ", ".join(f"'{tc.name}' (ID: {tc.id})" for tc in pending_tool_calls.values())
        error = f"ERROR: {num_pending} unresolved tool call{'s' if num_pending > 1 else ''} at end of transcript\n"
        error += f"  Pending: {tool_calls_str}\n"
        error += f"  Problem: All tool calls must receive results before the transcript ends\n"
        error += f"  Fix: Add <agentml:tool_result>...</agentml:tool_result> for each pending tool call"
        errors.append(error)
    
    # Check that transcript ends with user message (only if required)
    if require_user_ending and messages:
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