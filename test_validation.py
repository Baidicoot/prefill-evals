#!/usr/bin/env python3
"""
Test validation behavior with different transcript endings.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefill_evals.converters import TranscriptConverter

# Test 1: Transcript that doesn't end with user message
print("Test 1: Transcript ending with agent message")
print("-" * 50)

format_type, messages, errors = TranscriptConverter.load_from_file(
    "test_transcript_no_user_end.txt",
    validate=True,
    require_user_ending=False  # Should pass
)

print(f"Format: {format_type}")
print(f"Number of messages: {len(messages)}")
print(f"Last message type: {type(messages[-1]).__name__}")
print(f"Last message role: {getattr(messages[-1], 'role', 'N/A')}")
print(f"Validation errors: {len(errors) if errors else 'None'}")

# Test 2: Same transcript but require user ending
print("\n\nTest 2: Same transcript but require user ending")
print("-" * 50)

format_type, messages, errors = TranscriptConverter.load_from_file(
    "test_transcript_no_user_end.txt",
    validate=True,
    require_user_ending=True  # Should fail
)

print(f"Format: {format_type}")
print(f"Number of messages: {len(messages)}")
print(f"Validation errors: {len(errors) if errors else 'None'}")
if errors:
    print("\nValidation errors:")
    for error in errors:
        print(f"  - {error[:100]}..." if len(error) > 100 else f"  - {error}")

# Test 3: Real transcript from evals (should end with user)
print("\n\nTest 3: Real transcript from evals")
print("-" * 50)

real_transcript = Path("../evals/deceptiveness_24_07_filtered/coding_deceptiveness/10-system-documentation-generation/transcript.txt")
if real_transcript.exists():
    format_type, messages, errors = TranscriptConverter.load_from_file(
        real_transcript,
        validate=True,
        require_user_ending=True  # Should pass
    )
    
    print(f"Format: {format_type}")
    print(f"Number of messages: {len(messages)}")
    print(f"Last message type: {type(messages[-1]).__name__}")
    print(f"Last message role: {getattr(messages[-1], 'role', 'N/A')}")
    print(f"Validation errors: {len(errors) if errors else 'None'}")