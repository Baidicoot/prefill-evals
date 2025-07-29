#!/usr/bin/env python3
"""
Show parsed messages from a transcript.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefill_evals.converters import TranscriptConverter
from prefill_evals.models import TextMessage, ToolCall, ToolResult

# Load a real transcript
real_transcript = Path("../evals/deceptiveness_24_07_filtered/coding_deceptiveness/10-system-documentation-generation/transcript.txt")

format_type, messages, errors = TranscriptConverter.load_from_file(
    real_transcript,
    validate=True,
    require_user_ending=False
)

print(f"Loaded {len(messages)} messages from {format_type} format\n")

# Show first 10 messages in detail
print("First 10 messages:")
print("=" * 80)

for i, msg in enumerate(messages[:10]):
    print(f"\nMessage {i+1}:")
    print(f"  Type: {type(msg).__name__}")
    
    if isinstance(msg, TextMessage):
        print(f"  Role: {msg.role}")
        print(f"  Content: {msg.content[:200]}..." if len(msg.content) > 200 else f"  Content: {msg.content}")
    elif isinstance(msg, ToolCall):
        print(f"  Tool: {msg.name}")
        print(f"  ID: {msg.id}")
        print(f"  Parameters:")
        for key, value in msg.params.items():
            print(f"    {key}: {value[:100]}..." if len(str(value)) > 100 else f"    {key}: {value}")
    elif isinstance(msg, ToolResult):
        print(f"  Tool Call ID: {msg.tool_call_id}")
        print(f"  Result: {msg.content[:200]}..." if len(msg.content) > 200 else f"  Result: {msg.content}")

# Show message type summary
print("\n\nMessage Type Summary:")
print("=" * 80)
from collections import Counter

# Count by type
type_counts = Counter(type(msg).__name__ for msg in messages)
for msg_type, count in type_counts.items():
    print(f"  {msg_type}: {count}")

# Count TextMessages by role
text_messages = [msg for msg in messages if isinstance(msg, TextMessage)]
role_counts = Counter(msg.role for msg in text_messages)
print("\nTextMessage roles:")
for role, count in role_counts.items():
    print(f"  {role}: {count}")

# Show last 3 messages
print("\n\nLast 3 messages:")
print("=" * 80)

for i, msg in enumerate(messages[-3:], len(messages)-2):
    print(f"\nMessage {i}:")
    print(f"  Type: {type(msg).__name__}")
    
    if isinstance(msg, TextMessage):
        print(f"  Role: {msg.role}")
        print(f"  Content: {msg.content[:300]}..." if len(msg.content) > 300 else f"  Content: {msg.content}")

# Test JSON serialization of a few messages
print("\n\nJSON representation of first 3 messages:")
print("=" * 80)
import json
for i, msg in enumerate(messages[:3]):
    print(f"\nMessage {i+1} as JSON:")
    print(json.dumps(msg.to_dict(), indent=2))