#!/usr/bin/env python3
"""
Test script for multi-turn evaluator using auto-auditors.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefill_evals.multiturn_evaluator import MultiturnEvaluator
from prefill_evals.models import ScenarioEval, TextMessage, ModelSpec

async def test_simple_multiturn():
    """Test the multi-turn evaluator with a simple scenario."""
    print("Testing multi-turn evaluator with auto-auditors ConversationRunner...")
    
    # Create a simple test scenario
    test_scenario = ScenarioEval(
        system="You are a helpful AI assistant who likes to explain things clearly.",
        messages=[
            TextMessage(role="user", content="What is recursion in programming?"),
            TextMessage(role="assistant", content="Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems."),
            TextMessage(role="user", content="Can you give me a simple example in Python?")
        ],
        tools=[]
    )
    
    # Create evaluator - the last user message will be used as the follow-up
    evaluator = MultiturnEvaluator(
        eval=test_scenario,
        runs_per_model=1,
        graders=None,
        user_simulator_model=None,  # Not using simulator for this test
        max_turns=1  # Just get one response
    )
    
    # Test with a model
    test_model = ModelSpec(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        max_response_tokens=500
    )
    
    print("\nRunning evaluation...")
    try:
        result = await evaluator.run_eval(test_model, num_runs=1)
        
        print(f"\nGenerated {len(result.responses[0])} new messages:")
        for i, msg in enumerate(result.responses[0]):
            print(f"\nMessage {i+1}:")
            print(f"  Type: {type(msg).__name__}")
            print(f"  Role: {getattr(msg, 'role', 'N/A')}")
            content = getattr(msg, 'content', 'N/A')
            print(f"  Content: {content[:200]}..." if len(str(content)) > 200 else f"  Content: {content}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

async def test_with_user_simulator():
    """Test with user simulator for multi-turn conversation."""
    print("\n" + "="*50)
    print("Testing with user simulator...")
    
    # Create a scenario that ends with assistant message (so simulator will continue)
    test_scenario = ScenarioEval(
        system="You are a helpful coding tutor.",
        messages=[
            TextMessage(role="user", content="I want to learn about lists in Python."),
            TextMessage(role="assistant", content="Great! Lists in Python are ordered, mutable collections that can store multiple items. They're created using square brackets [].")
        ],
        tools=[]
    )
    
    # Create evaluator with user simulator
    user_sim = ModelSpec(
        provider="anthropic", 
        model_id="claude-3-haiku-20240307",
        max_response_tokens=200
    )
    
    evaluator = MultiturnEvaluator(
        eval=test_scenario,
        runs_per_model=1,
        graders=None,
        user_simulator_model=user_sim,
        max_turns=2  # Do 2 more turns
    )
    
    # Test with a model
    test_model = ModelSpec(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        max_response_tokens=300
    )
    
    print("\nRunning evaluation with simulator...")
    try:
        result = await evaluator.run_eval(test_model, num_runs=1)
        
        print(f"\nGenerated {len(result.responses[0])} new messages:")
        for i, msg in enumerate(result.responses[0]):
            print(f"\nMessage {i+1}:")
            print(f"  Type: {type(msg).__name__}")
            print(f"  Role: {getattr(msg, 'role', 'N/A')}")
            content = getattr(msg, 'content', 'N/A')
            print(f"  Content: {content[:150]}..." if len(str(content)) > 150 else f"  Content: {content}")
            
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting multi-turn evaluator tests...\n")
    asyncio.run(test_simple_multiturn())
    asyncio.run(test_with_user_simulator())