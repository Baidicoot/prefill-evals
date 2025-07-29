#!/usr/bin/env python3
"""
Test script for multi-turn evaluator.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefill_evals.config import load_config
from prefill_evals.parser import load_scenario_from_dir
from prefill_evals.multiturn_evaluator import MultiturnEvaluator
from prefill_evals.models import ScenarioEval, TextMessage

async def test_multiturn_evaluator():
    """Test the multi-turn evaluator with a simple scenario."""
    
    # Create a simple test scenario
    test_scenario = ScenarioEval(
        system="You are a helpful AI assistant.",
        messages=[
            TextMessage(role="user", content="Hello! Can you help me understand recursion?"),
            TextMessage(role="assistant", content="I'd be happy to help you understand recursion! Recursion is a programming technique where a function calls itself to solve a problem."),
            TextMessage(role="user", content="Can you give me a simple example?")
        ],
        tools=[]
    )
    
    # Create evaluator without user simulator (just process existing messages)
    evaluator = MultiturnEvaluator(
        eval=test_scenario,
        runs_per_model=1,
        graders=None,
        user_simulator_model=None  # No simulator for this test
    )
    
    # Test with a model
    from prefill_evals.models import ModelSpec
    test_model = ModelSpec(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        max_response_tokens=500
    )
    
    print("Running multi-turn evaluation...")
    result = await evaluator.run_eval(test_model, num_runs=1)
    
    print(f"\nGenerated {len(result.responses[0])} messages:")
    for i, msg in enumerate(result.responses[0]):
        print(f"\nMessage {i+1}:")
        print(f"  Type: {type(msg).__name__}")
        print(f"  Role: {getattr(msg, 'role', 'N/A')}")
        print(f"  Content: {msg.content[:100]}..." if hasattr(msg, 'content') and len(msg.content) > 100 else f"  Content: {getattr(msg, 'content', 'N/A')}")
    
    print("\nTest completed successfully!")

async def test_easy_orchestrator_loading():
    """Test loading from easy-orchestrator format."""
    print("\n" + "="*50)
    print("Testing easy-orchestrator format loading...")
    
    # Check if we have the auto-auditors submodule
    orchestrator_path = Path("external/auto-auditors/easy-orchestrator")
    if not orchestrator_path.exists():
        print("Auto-auditors submodule not found. Skipping this test.")
        return
    
    # Try to find a sample transcript
    sample_transcripts = list(orchestrator_path.glob("**/transcript*.json"))
    if not sample_transcripts:
        print("No sample transcripts found in auto-auditors. Creating mock transcript...")
        
        # Create a mock transcript for testing
        import json
        mock_transcript = {
            "generations": [{
                "variation_id": "test-123",
                "rollout_id": 1,
                "transcript": {
                    "events": [
                        {"event": "target_system_prompt", "content": "You are a helpful assistant.", "turn": 0},
                        {"event": "evaluator_message", "content": "What is 2+2?", "turn": 1},
                        {"event": "target_message", "content": "2+2 equals 4.", "turn": 1},
                        {"event": "evaluator_message", "content": "Can you explain why?", "turn": 2}
                    ]
                }
            }]
        }
        
        mock_path = Path("test_transcript.json")
        with open(mock_path, 'w') as f:
            json.dump(mock_transcript, f, indent=2)
        
        # Test loading
        evaluator = MultiturnEvaluator.from_easy_orchestrator(
            transcript_path=mock_path,
            variation_id="test-123",
            rollout_id=1
        )
        
        print(f"Loaded scenario with {len(evaluator.eval.messages)} messages")
        print(f"System prompt: {evaluator.eval.system}")
        
        # Clean up
        mock_path.unlink()
    
    print("Easy-orchestrator loading test completed!")

if __name__ == "__main__":
    asyncio.run(test_multiturn_evaluator())
    asyncio.run(test_easy_orchestrator_loading())