"""
Prefill evaluation framework with seamless format conversion.
"""

from prefill_evals.converters import (
    TranscriptConverter,
    text_to_json,
    json_to_text,
    text_to_messages,
    messages_to_text,
    messages_to_json,
    json_to_messages
)

__all__ = [
    'TranscriptConverter',
    'text_to_json',
    'json_to_text',
    'text_to_messages',
    'messages_to_text',
    'messages_to_json',
    'json_to_messages'
]