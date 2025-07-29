#!/usr/bin/env python3
"""
Unified conversion utilities for seamless format transitions between:
- Text-based transcripts (XML format)
- Internal AgentMessage representation
- JSON serialization
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

from prefill_evals.models import (
    AgentMessage, TextMessage, ToolCall, ToolResult, ScenarioEval,
    serialize_messages, deserialize_messages, messages_to_text, validate_scenario
)
from prefill_evals.parser import parse_transcript, render_transcript


class TranscriptConverter:
    """Unified converter for transcript format transitions."""
    
    @staticmethod
    def text_to_messages(text: str) -> tuple[Optional[str], List[AgentMessage]]:
        """
        Convert text transcript to list of AgentMessage objects.
        
        Args:
            text: XML-formatted transcript text
            
        Returns:
            Tuple of (system_prompt, messages)
        """
        parsed = parse_transcript(text)
        return parsed.get('system'), parsed.get('messages', [])
    
    @staticmethod
    def messages_to_text(messages: List[AgentMessage], system_prompt: Optional[str] = None) -> str:
        """
        Convert list of AgentMessage objects to text transcript.
        
        Args:
            messages: List of AgentMessage objects
            system_prompt: Optional system prompt
            
        Returns:
            XML-formatted transcript text
        """
        scenario = ScenarioEval(
            messages=messages,
            tools=[],  # Tools are not included in text format
            system=system_prompt
        )
        return render_transcript(scenario)
    
    @staticmethod
    def messages_to_json(messages: List[AgentMessage]) -> str:
        """
        Convert list of AgentMessage objects to JSON string.
        
        Args:
            messages: List of AgentMessage objects
            
        Returns:
            JSON string representation
        """
        return json.dumps(serialize_messages(messages), indent=2)
    
    @staticmethod
    def json_to_messages(json_str: str) -> List[AgentMessage]:
        """
        Convert JSON string to list of AgentMessage objects.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            List of AgentMessage objects
        """
        data = json.loads(json_str)
        return deserialize_messages(data)
    
    @staticmethod
    def text_to_json(text: str) -> str:
        """
        Convert text transcript directly to JSON.
        
        Args:
            text: XML-formatted transcript text
            
        Returns:
            JSON string representation
        """
        system_prompt, messages = TranscriptConverter.text_to_messages(text)
        result = {
            'messages': serialize_messages(messages)
        }
        if system_prompt:
            result['system'] = system_prompt
        return json.dumps(result, indent=2)
    
    @staticmethod
    def json_to_text(json_str: str) -> str:
        """
        Convert JSON directly to text transcript.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            XML-formatted transcript text
        """
        data = json.loads(json_str)
        messages = deserialize_messages(data.get('messages', []))
        system_prompt = data.get('system')
        return TranscriptConverter.messages_to_text(messages, system_prompt)
    
    @staticmethod
    def scenario_to_json(scenario: ScenarioEval) -> str:
        """
        Convert ScenarioEval to JSON string.
        
        Args:
            scenario: ScenarioEval object
            
        Returns:
            JSON string representation
        """
        return json.dumps(scenario.to_dict(), indent=2)
    
    @staticmethod
    def json_to_scenario(json_str: str) -> ScenarioEval:
        """
        Convert JSON string to ScenarioEval.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            ScenarioEval object
        """
        data = json.loads(json_str)
        return ScenarioEval.from_dict(data)
    
    @staticmethod
    def load_from_file(
        file_path: Union[str, Path], 
        validate: bool = False,
        require_user_ending: bool = False
    ) -> tuple[str, List[AgentMessage], Optional[List[str]]]:
        """
        Load transcript from file (supports .txt, .xml, and .json).
        
        Args:
            file_path: Path to file
            validate: If True, validate the loaded scenario
            require_user_ending: If True, require transcript to end with user message (for generation)
            
        Returns:
            Tuple of (format_type, messages, validation_errors) where format_type is 'text' or 'json'
            and validation_errors is None if validate=False or empty list if valid
        """
        path = Path(file_path)
        content = path.read_text()
        
        system_prompt = None
        messages = None
        validation_errors = None
        
        if path.suffix.lower() == '.json':
            try:
                data = json.loads(content)
                messages = deserialize_messages(data.get('messages', []))
                system_prompt = data.get('system')
                format_type = 'json'
            except json.JSONDecodeError:
                # Try as text format
                pass
        
        if messages is None:
            # Default to text/XML format for .txt, .xml, or any other extension
            system_prompt, messages = TranscriptConverter.text_to_messages(content)
            format_type = 'text'
        
        # Validate if requested
        if validate:
            scenario = ScenarioEval(
                messages=messages,
                tools=[],  # Tools not included in simple file load
                system=system_prompt
            )
            validation_errors = validate_scenario(scenario, require_user_ending=require_user_ending)
        
        return format_type, messages, validation_errors
    
    @staticmethod
    def save_to_file(
        messages: List[AgentMessage], 
        file_path: Union[str, Path],
        format: str = 'auto',
        system_prompt: Optional[str] = None
    ):
        """
        Save messages to file.
        
        Args:
            messages: List of AgentMessage objects
            file_path: Path to save to
            format: 'json', 'text', or 'auto' (auto-detect from extension)
            system_prompt: Optional system prompt (for text format)
        """
        path = Path(file_path)
        
        if format == 'auto':
            # Only use JSON for .json extension, default to text/XML for everything else
            format = 'json' if path.suffix.lower() == '.json' else 'text'
        
        if format == 'json':
            content = TranscriptConverter.messages_to_json(messages)
            if system_prompt:
                data = json.loads(content)
                full_data = {'system': system_prompt, 'messages': data}
                content = json.dumps(full_data, indent=2)
        else:
            # Default to text/XML format
            content = TranscriptConverter.messages_to_text(messages, system_prompt)
        
        path.write_text(content)


# Convenience functions for direct conversion
def text_to_json(text: str) -> str:
    """Convert XML transcript to JSON."""
    return TranscriptConverter.text_to_json(text)


def json_to_text(json_str: str) -> str:
    """Convert JSON to XML transcript."""
    return TranscriptConverter.json_to_text(json_str)


def text_to_messages(text: str) -> List[AgentMessage]:
    """Convert XML transcript to AgentMessage list."""
    _, messages = TranscriptConverter.text_to_messages(text)
    return messages


def messages_to_text(messages: List[AgentMessage], system_prompt: Optional[str] = None) -> str:
    """Convert AgentMessage list to XML transcript."""
    return TranscriptConverter.messages_to_text(messages, system_prompt)


def messages_to_json(messages: List[AgentMessage]) -> str:
    """Convert AgentMessage list to JSON."""
    return TranscriptConverter.messages_to_json(messages)


def json_to_messages(json_str: str) -> List[AgentMessage]:
    """Convert JSON to AgentMessage list."""
    return TranscriptConverter.json_to_messages(json_str)