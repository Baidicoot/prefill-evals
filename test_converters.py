#!/usr/bin/env python3
"""
Test script for converter functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefill_evals.converters import TranscriptConverter

# Test with a sample transcript
test_transcript_path = Path("../evals/deceptiveness_24_07_filtered/coding_deceptiveness/10-system-documentation-generation/transcript.txt")

if test_transcript_path.exists():
    print(f"Testing with: {test_transcript_path}")
    
    # Load with validation
    format_type, messages, errors = TranscriptConverter.load_from_file(
        test_transcript_path, 
        validate=True,
        require_user_ending=False  # Don't require user ending for evaluation transcripts
    )
    
    print(f"\nFormat: {format_type}")
    print(f"Number of messages: {len(messages)}")
    
    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nNo validation errors!")
    
    # Show message types
    print("\nMessage types:")
    from collections import Counter
    msg_types = Counter(type(msg).__name__ for msg in messages)
    for msg_type, count in msg_types.items():
        print(f"  - {msg_type}: {count}")
    
    # Test conversion to JSON
    json_str = TranscriptConverter.messages_to_json(messages)
    print(f"\nJSON representation (first 200 chars):")
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
    
    # Test round-trip conversion
    messages_from_json = TranscriptConverter.json_to_messages(json_str)
    print(f"\nRound-trip successful: {len(messages) == len(messages_from_json)}")
    
    # Save as JSON
    json_output = Path("test_output.json")
    TranscriptConverter.save_to_file(messages, json_output)
    print(f"\nSaved to {json_output}")
    
    # Save as XML  
    xml_output = Path("test_output.xml")
    TranscriptConverter.save_to_file(messages, xml_output)
    print(f"Saved to {xml_output}")
else:
    print(f"Test file not found: {test_transcript_path}")
    print("Looking for other transcript files...")
    
    # Find some transcript files
    import glob
    transcripts = list(Path("../evals").glob("**/transcript.txt"))[:5]
    print(f"\nFound {len(transcripts)} transcript files. First few:")
    for t in transcripts:
        print(f"  - {t}")