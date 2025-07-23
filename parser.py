#!/usr/bin/env python3
"""
XML-based transcript parser with manual parsing approach.
"""

from typing import Dict, List, Optional, Tuple, Any

from prefill_evals.models import (
    TextMessage, ToolCall, ToolResult, validate_scenario, ToolDefinition, ToolParameter, ScenarioEval
)

from pathlib import Path

# Configurable tag names
SYSTEM_TAG = "system"
USER_TAG = "user"
AGENT_TAG = "agent"
TOOL_CALL_PREFIX = "tool_call"
TOOL_RESULT_TAG = "tool_result"
ARGUMENT_PREFIX = "argument"

# Tool definition tags (for tools.txt parsing)
TOOL_DEF_TAG = "tool"
TOOL_NAME_ATTR = "name"
TOOL_DESCRIPTION_TAG = "description"
TOOL_PARAMETER_TAG = "parameter"
TOOL_PARAM_NAME_ATTR = "name"
TOOL_PARAM_TYPE_ATTR = "type"
TOOL_PARAM_OPTIONAL_ATTR = "optional"


def find_next_tag(content: str, start_pos: int = 0) -> Optional[Tuple[int, int, str, Dict[str, str]]]:
    """
    Find the next XML tag starting from start_pos.
    
    Returns:
        Tuple of (tag_start, tag_end, tag_name, attributes) or None if no tag found
        For <tool_call:function_name>, tag_name will be "tool_call:function_name"
    """
    # Find next '<'
    tag_start = content.find('<', start_pos)
    if tag_start == -1:
        return None
    
    # Skip if it's a closing tag
    if content[tag_start + 1] == '/':
        return find_next_tag(content, tag_start + 1)
    
    # Find the end of the tag '>'
    tag_end = content.find('>', tag_start)
    if tag_end == -1:
        return None
    
    # Extract tag content
    tag_content = content[tag_start + 1:tag_end].strip()
    
    # Split tag name and attributes
    parts = tag_content.split(None, 1)
    tag_name = parts[0]
    attributes = {}
    
    # Parse attributes if any (simple attribute parsing)
    if len(parts) > 1:
        # Simple attribute parsing (assumes attributes are well-formed)
        # This is a basic implementation - could be enhanced if needed
        pass
        
    return (tag_start, tag_end + 1, tag_name, attributes)


def find_closing_tag(content: str, tag_name: str, start_pos: int) -> Optional[int]:
    """
    Find the closing tag for the given tag name.
    
    Returns:
        Position after the closing tag or None if not found
    """
    # For tags with colons, we need to match exactly
    closing_tag = f"</{tag_name}>"
    pos = content.find(closing_tag, start_pos)
    if pos == -1:
        return None
    return pos + len(closing_tag)


def extract_tag_content(content: str, tag_start_pos: int, tag_end_pos: int, tag_name: str, closing_pos: int) -> str:
    """
    Extract content between opening and closing tags.
    """
    closing_tag_len = len(f"</{tag_name}>")
    return content[tag_end_pos:closing_pos - closing_tag_len].strip()


def parse_tool_call_content(content: str) -> Dict[str, str]:
    """
    Parse the content of a tool call to extract arguments.
    
    Returns:
        Dictionary of argument names to values
    """
    arguments = {}
    pos = 0
    
    while pos < len(content):
        # Find next argument tag
        arg_start = content.find(f"<{ARGUMENT_PREFIX}:", pos)
        if arg_start == -1:
            break
        
        # Find the end of the opening tag
        arg_tag_end = content.find('>', arg_start)
        if arg_tag_end == -1:
            break
        
        # Extract argument name
        arg_tag_content = content[arg_start + len(f"<{ARGUMENT_PREFIX}:"):arg_tag_end]
        arg_name = arg_tag_content.strip()
        
        # Find closing tag
        closing_tag = f"</{ARGUMENT_PREFIX}:{arg_name}>"
        closing_pos = content.find(closing_tag, arg_tag_end)
        if closing_pos == -1:
            break
        
        # Extract argument value
        arg_value = content[arg_tag_end + 1:closing_pos].strip()
        arguments[arg_name] = arg_value
        
        pos = closing_pos + len(closing_tag)
    
    return arguments


def parse_transcript(transcript: str) -> Dict[str, Any]:
    """
    Parse a transcript with XML-style tags using manual parsing.
    
    Expected format:
    <system>...</system>
    <user>...</user>
    <agent>...</agent>
    <tool_call:function_name>
        <argument:arg_name>...</argument:arg_name>
    </tool_call:function_name>
    <tool_result>...</tool_result>
    
    Returns:
        Dict with:
        - 'system': System prompt content (or None)
        - 'messages': List of AgentMessage objects
    """
    system_prompt = None
    messages = []
    pos = 0
    tool_call_counter = 0
    pending_tool_calls = {}  # Map position to tool call info
    
    while pos < len(transcript):
        # Find next tag
        tag_info = find_next_tag(transcript, pos)
        if tag_info is None:
            break
        
        tag_start, tag_end, tag_name, _ = tag_info
        
        # Find closing tag
        closing_pos = find_closing_tag(transcript, tag_name, tag_end)
        if closing_pos is None:
            pos = tag_end
            continue
        
        # Extract content
        content = extract_tag_content(transcript, tag_start, tag_end, tag_name, closing_pos)
        
        # Process based on tag type
        if tag_name == SYSTEM_TAG:
            system_prompt = content
        
        elif tag_name == USER_TAG:
            messages.append(TextMessage(role='user', content=content))
        
        elif tag_name == AGENT_TAG:
            messages.append(TextMessage(role='assistant', content=content))
        
        elif tag_name.startswith(f"{TOOL_CALL_PREFIX}:"):
            # Extract function name
            function_name = tag_name[len(f"{TOOL_CALL_PREFIX}:"):]
            
            # Parse arguments
            arguments = parse_tool_call_content(content)
            
            # Create tool call
            tool_call_id = f"call_{tool_call_counter}"
            tool_call_counter += 1
            
            tool_call = ToolCall(
                name=function_name,
                params=arguments,
                id=tool_call_id
            )
            messages.append(tool_call)
            
            # Remember this tool call for linking with results
            pending_tool_calls[len(messages) - 1] = tool_call_id
        
        elif tag_name == TOOL_RESULT_TAG:
            # Find the most recent tool call
            recent_tool_id = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], ToolCall):
                    recent_tool_id = messages[i].id
                    break
            
            if recent_tool_id:
                messages.append(ToolResult(
                    tool_call_id=recent_tool_id,
                    content=content
                ))
        
        pos = closing_pos
    
    return {
        'system': system_prompt,
        'messages': messages
    }


def get_transcript_ast(parsed: Dict[str, Any], max_content_length: int = 40) -> str:
    """
    Generate an AST-like representation of the parsed transcript.
    
    Args:
        parsed: Parsed transcript from parse_transcript()
        max_content_length: Maximum length of content to show
        
    Returns:
        String representation of the transcript structure
    """
    ast_lines = []
    
    # Add system prompt if present
    if parsed.get('system'):
        system_preview = parsed['system'][:max_content_length]
        if len(parsed['system']) > max_content_length:
            system_preview += "..."
        ast_lines.append(f"System: \"{system_preview}\"")
        ast_lines.append("")
    
    # Add messages
    messages = parsed.get('messages', [])
    for i, msg in enumerate(messages):
        # Use the render method for consistent formatting
        rendered = msg.render(max_content_length)
        
        if isinstance(msg, TextMessage):
            ast_lines.append(f"Message {i+1}: {msg.role.title()}")
            ast_lines.append(f"  {rendered}")
                
        elif isinstance(msg, ToolCall):
            ast_lines.append(f"Message {i+1}: Tool Call (id: {msg.id})")
            ast_lines.append(f"  {rendered}")
            
        elif isinstance(msg, ToolResult):
            ast_lines.append(f"Message {i+1}: Tool Result (id: {msg.tool_call_id})")
            ast_lines.append(f"  {rendered}")
        
        ast_lines.append("")  # Empty line between messages
    
    return "\n".join(ast_lines)


def parse_tool_definitions(tools_content: str) -> List[ToolDefinition]:
    """
    Parse tool definitions from XML format.
    
    Expected format:
    <tool name="function_name">
        <description>Tool description</description>
        <parameter name="param_name" type="string" optional="true">Parameter description</parameter>
    </tool>
    """
    tools = []
    pos = 0
    
    while pos < len(tools_content):
        # Find next tool tag
        tool_start = tools_content.find(f"<{TOOL_DEF_TAG}", pos)
        if tool_start == -1:
            break
        
        # Find end of opening tag
        tool_tag_end = tools_content.find('>', tool_start)
        if tool_tag_end == -1:
            break
        
        # Extract tool name from tag
        tag_content = tools_content[tool_start + len(f"<{TOOL_DEF_TAG}"):tool_tag_end]
        # Simple extraction of name attribute
        name_start = tag_content.find(f'{TOOL_NAME_ATTR}="')
        if name_start == -1:
            pos = tool_tag_end
            continue
        name_start += len(f'{TOOL_NAME_ATTR}="')
        name_end = tag_content.find('"', name_start)
        tool_name = tag_content[name_start:name_end]
        
        # Find closing tool tag
        closing_tag = f"</{TOOL_DEF_TAG}>"
        tool_closing = tools_content.find(closing_tag, tool_tag_end)
        if tool_closing == -1:
            break
        
        # Extract tool content
        tool_content = tools_content[tool_tag_end + 1:tool_closing]
        
        # Parse description
        description = None
        desc_start = tool_content.find(f"<{TOOL_DESCRIPTION_TAG}>")
        if desc_start != -1:
            desc_start += len(f"<{TOOL_DESCRIPTION_TAG}>")
            desc_end = tool_content.find(f"</{TOOL_DESCRIPTION_TAG}>", desc_start)
            if desc_end != -1:
                description = tool_content[desc_start:desc_end].strip()
        
        # Parse parameters
        parameters = []
        param_pos = 0
        while param_pos < len(tool_content):
            param_start = tool_content.find(f"<{TOOL_PARAMETER_TAG}", param_pos)
            if param_start == -1:
                break
            
            param_tag_end = tool_content.find('>', param_start)
            if param_tag_end == -1:
                break
            
            # Extract parameter attributes
            param_tag_content = tool_content[param_start + len(f"<{TOOL_PARAMETER_TAG}"):param_tag_end]
            
            # Extract name
            param_name = None
            name_start = param_tag_content.find(f'{TOOL_PARAM_NAME_ATTR}="')
            if name_start != -1:
                name_start += len(f'{TOOL_PARAM_NAME_ATTR}="')
                name_end = param_tag_content.find('"', name_start)
                param_name = param_tag_content[name_start:name_end]
            
            # Extract type
            param_type = 'string'  # default
            type_start = param_tag_content.find(f'{TOOL_PARAM_TYPE_ATTR}="')
            if type_start != -1:
                type_start += len(f'{TOOL_PARAM_TYPE_ATTR}="')
                type_end = param_tag_content.find('"', type_start)
                param_type = param_tag_content[type_start:type_end]
            
            # Check if optional
            param_optional = f'{TOOL_PARAM_OPTIONAL_ATTR}="true"' in param_tag_content
            
            # Extract parameter description
            param_desc_end = tool_content.find(f"</{TOOL_PARAMETER_TAG}>", param_tag_end)
            param_description = ""
            if param_desc_end != -1:
                param_description = tool_content[param_tag_end + 1:param_desc_end].strip()
            
            if param_name:
                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    optional=param_optional,
                    description=param_description
                ))
            
            param_pos = param_desc_end + len(f"</{TOOL_PARAMETER_TAG}>") if param_desc_end != -1 else param_tag_end + 1
        
        tools.append(ToolDefinition(
            name=tool_name,
            description=description,
            parameters=parameters
        ))
        
        pos = tool_closing + len(closing_tag)
    
    return tools


def parse_and_validate_items(
    transcript: str,
    tools: Optional[str] = None,
    extra_items: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    debug: bool = False,
) -> ScenarioEval:
    """
    Given a list of items, parse and validate them.
    These can be loaded from separate files or parsed from an author model's output.
    """
    # Parse the transcript
    parsed = parse_transcript(transcript)
    
    # Parse tools if provided
    tool_definitions = []
    if tools:
        tool_definitions = parse_tool_definitions(tools)
    
    # Debug: print parsed transcript
    if debug:
        print("\n" + "="*80)
        print("DEBUG: Parsed Transcript")
        print("="*80)
        print(get_transcript_ast(parsed))
        print("="*80 + "\n")
    
    # Create ScenarioEval object
    scenario = ScenarioEval(
        system=parsed.get('system'),
        messages=parsed['messages'],
        tools=tool_definitions,
        extra_items=extra_items
    )
    
    # Validate if requested
    if validate:
        validation_errors = validate_scenario(scenario)
        if validation_errors:
            print(f"Validation errors found: {len(validation_errors)}")
            for error in validation_errors:
                print(f"  {error}")
            # Raise exception to allow caller to handle validation errors
            raise ValueError(f"Scenario validation failed with {len(validation_errors)} errors")
    
    return scenario


def load_scenario_from_dir(
    path: Path,
    extra_items: Optional[List[str]] = None,
    debug: bool = False
) -> ScenarioEval:
    """
    Load a scenario from a directory where each item is saved as a separate .txt file.
    
    Args:
        path: Directory path containing scenario files
        extra_items: Optional list of additional item names to load
        debug: Whether to print debug information
        
    Returns:
        ScenarioEval with parsed and validated scenario data
        
    Expected files:
        - transcript.txt (required)
        - tools.txt (optional)
        - Any files matching {item_name}.txt for items in extra_items list
    """
    if not path.exists():
        raise ValueError(f"Scenario directory does not exist: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Load transcript (required)
    transcript_file = path / "transcript.txt"
    if not transcript_file.exists():
        raise ValueError(f"Required transcript.txt file not found in {path}")
    
    transcript = transcript_file.read_text()
    
    # Load tools (optional)
    tools = None
    tools_file = path / "tools.txt"
    if tools_file.exists():
        tools = tools_file.read_text()
    
    # Load extra items if specified
    extra_items_dict = {}
    if extra_items:
        for item_name in extra_items:
            item_file = path / f"{item_name}.txt"
            if item_file.exists():
                extra_items_dict[item_name] = item_file.read_text()
    
    # Parse and validate
    return parse_and_validate_items(
        transcript=transcript,
        tools=tools,
        extra_items=extra_items_dict,
        validate=True,
        debug=debug
    )


def render_transcript(scenario: 'ScenarioEval') -> str:
    """
    Render a ScenarioEval as an XML-formatted transcript string.
    
    Args:
        scenario: ScenarioEval object containing messages and optional system prompt
        
    Returns:
        XML-formatted transcript string
    """
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


def parse_xml_tags(content: str, tags: List[str], required_tags: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Parse XML-style tags from content into a dictionary.
    
    Args:
        content: Content with XML tags
        tags: List of tag names to extract
        required_tags: List of tags that must be present (raises ValueError if missing)
        
    Returns:
        Dict mapping tag names to their content
        
    Raises:
        ValueError: If required tags are missing or tags are malformed
    """
    result = {}
    required_tags = required_tags or []
    
    for tag in tags:
        # Find opening tag
        opening_tag = f"<{tag}>"
        closing_tag = f"</{tag}>"
        
        start_pos = content.find(opening_tag)
        if start_pos != -1:
            # Found opening tag
            end_pos = content.find(closing_tag, start_pos)
            if end_pos == -1:
                raise ValueError(f"Found opening {opening_tag} but no closing {closing_tag}")
            
            # Extract content between tags
            tag_content = content[start_pos + len(opening_tag):end_pos].strip()
            result[tag] = tag_content
        else:
            # Tag not found
            if tag in required_tags:
                raise ValueError(f"Missing required <{tag}> tag")
            result[tag] = None
    
    return result


def parse_generated_output(content: str, expected_items: List[str]) -> Dict[str, str]:
    """
    Parse XML-wrapped generation output into tools, transcript, and extra items.
    
    This is a specialized function for parsing generator output which requires
    a transcript tag and optionally includes tools and extra items.
    
    Expected format:
    <tools>
    ... (tool definitions in standard format) ...
    </tools>
    <transcript>
    ... (transcript in standard format) ...
    </transcript>
    <extra_item_name>
    ... (plain text content) ...
    </extra_item_name>
    
    Args:
        content: Generated content with XML wrapper tags
        expected_items: List of expected item names (e.g., ['misdeed', 'target_behavior'])
        
    Returns:
        Dict with keys: 'tools', 'transcript', and any expected_items
        
    Raises:
        ValueError: If required tags are missing
    """
    # Parse all tags: tools (optional), transcript (required), and any expected items
    all_tags = ['tools', 'transcript'] + expected_items
    
    # Only transcript is required for generation output
    result = parse_xml_tags(content, all_tags, required_tags=['transcript'])
    
    # Additional validation for generation output
    if not result.get('transcript'):
        raise ValueError("Transcript content cannot be empty")
    
    # Ensure tools is None if not present (for consistency)
    if 'tools' not in result or result['tools'] is None:
        result['tools'] = None
    
    return result