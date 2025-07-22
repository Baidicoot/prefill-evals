#!/usr/bin/env python3
"""
Scenario generation pipeline for creating prefill-based evaluations.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import os
import sys

# Add safety-tooling to path if it's a sibling directory
safety_tooling_path = Path(__file__).parent.parent / "safety-tooling"
if safety_tooling_path.exists():
    sys.path.insert(0, str(safety_tooling_path))

from prefill_evals.config import GenerationConfig, ScenarioFeedbackConfig
from prefill_evals.parser import parse_generated_output, parse_and_validate_items
from prefill_evals.models import ScenarioEval, validate_scenario
from prefill_evals.feedback import ModelBasedScenarioFeedback, format_feedback_for_revision

# Import safety-tooling components
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils


class ScenarioGenerator:
    """Generate scenarios with iterative refinement based on feedback."""
    
    def __init__(self, config: GenerationConfig):
        """
        Initialize the generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config
        self.conversation_history = []
        
        # Setup environment and initialize InferenceAPI
        utils.setup_environment()
        cache_dir = Path(os.environ.get("CACHE_DIR", ".cache"))
        self.inference_api = InferenceAPI(
            cache_dir=cache_dir,
            anthropic_num_threads=80,
            openai_num_threads=100,
        )
        
        # Load prompt templates
        self.prompts = self._load_prompts()
        
        # Load seed items
        self.seed_items = self._load_seed_items()
        
        # Initialize feedback providers
        self.feedback_providers = self._init_feedback_providers()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from the prompts directory."""
        prompts = {}
        prompt_files = {
            'system': 'system.txt',
            'initial': 'initial.txt',
            'revision': 'revision.txt',
            'error': 'error.txt'
        }
        
        for prompt_name, filename in prompt_files.items():
            prompt_path = self.config.prompts_dir / filename
            if not prompt_path.exists():
                raise FileNotFoundError(f"Required prompt file not found: {prompt_path}")
            with open(prompt_path, 'r') as f:
                prompts[prompt_name] = f.read()
        
        return prompts
    
    def _load_seed_items(self) -> Dict[str, str]:
        """Load seed items from files."""
        seed_items = {}
        for item_name, item_path in self.config.seed_items.items():
            if not item_path.exists():
                raise FileNotFoundError(f"Seed item file not found: {item_path}")
            with open(item_path, 'r') as f:
                seed_items[item_name] = f.read()
        return seed_items
    
    def _init_feedback_providers(self) -> List[ModelBasedScenarioFeedback]:
        """Initialize feedback providers from config."""
        providers = []
        for fb_config in self.config.feedback_providers:
            provider = ModelBasedScenarioFeedback(
                name=fb_config.name,
                model_config={
                    'provider': fb_config.model.provider,
                    'model_id': fb_config.model.model_id
                },
                template_file=fb_config.template_file,
                inference_api=self.inference_api
            )
            providers.append(provider)
        return providers
    
    async def generate_scenario(self) -> Tuple[ScenarioEval, Dict[str, Any]]:
        """
        Generate a scenario with iterative refinement.
        
        Returns:
            Tuple of (final_scenario, generation_metadata)
        """
        metadata = {
            'iterations': [],
            'final_iteration': 0,
            'success': False
        }
        
        # Initial generation
        print("Generating initial scenario...")
        initial_response = await self._initial_generation()
        
        scenario = None
        for iteration in range(self.config.max_iterations):
            iteration_data = {
                'iteration': iteration + 1,
                'type': 'initial' if iteration == 0 else 'revision'
            }
            
            # Parse and validate the generation
            try:
                if iteration == 0:
                    response = initial_response
                else:
                    # Get revision based on feedback
                    response = await self._get_revision(feedback_text)
                
                iteration_data['raw_response'] = response
                
                # Parse the generated output
                parsed = parse_generated_output(response, self.config.extra_items_to_generate)
                iteration_data['parsed'] = True
                
                # Create scenario from parsed content
                extra_items = {k: v for k, v in parsed.items() 
                              if k not in ['tools', 'transcript'] and v is not None}
                
                scenario = parse_and_validate_items(
                    transcript=parsed['transcript'],
                    tools=parsed['tools'],
                    extra_items=extra_items,
                    validate=True,
                    debug=False
                )
                
                # Check for validation errors
                validation_errors = validate_scenario(scenario)
                if validation_errors:
                    iteration_data['validation_errors'] = validation_errors
                    # Send errors back to model
                    error_text = "\n".join(validation_errors)
                    print(f"Validation errors found, requesting fix...")
                    response = await self._handle_error(error_text)
                    continue
                
                iteration_data['valid'] = True
                
                # Get feedback on the scenario
                print(f"Getting feedback on iteration {iteration + 1}...")
                feedback_results = await self._get_feedback(scenario)
                iteration_data['feedback'] = feedback_results
                
                # Check if we should continue
                if self._should_continue(feedback_results):
                    feedback_text = format_feedback_for_revision(feedback_results)
                    print(f"Feedback suggests improvements, continuing...")
                else:
                    print(f"Scenario accepted after {iteration + 1} iterations")
                    metadata['success'] = True
                    metadata['final_iteration'] = iteration + 1
                    break
                
            except Exception as e:
                iteration_data['error'] = str(e)
                print(f"Error in iteration {iteration + 1}: {e}")
                
                # Try to recover from parsing errors
                if iteration < self.config.max_iterations - 1:
                    response = await self._handle_error(str(e))
                    continue
                else:
                    raise
            
            finally:
                metadata['iterations'].append(iteration_data)
        
        if scenario is None:
            raise RuntimeError("Failed to generate valid scenario after all iterations")
        
        # Save the scenario
        await self._save_scenario(scenario, metadata)
        
        return scenario, metadata
    
    async def _initial_generation(self) -> str:
        """Generate the initial scenario."""
        # Format the initial prompt with seed items
        format_kwargs = self.seed_items.copy()
        
        initial_prompt = self.prompts['initial'].format(**format_kwargs)
        
        # Set up conversation with system prompt
        self.conversation_history = [
            {"role": "system", "content": self.prompts['system']},
            {"role": "user", "content": initial_prompt}
        ]
        
        # Get response from model
        response = await self._call_generator_model()
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _get_revision(self, feedback: str) -> str:
        """Get a revision based on feedback."""
        # Format revision prompt
        revision_prompt = self.prompts['revision'].format(feedback=feedback)
        
        # Add to conversation
        self.conversation_history.append({"role": "user", "content": revision_prompt})
        
        # Get response
        response = await self._call_generator_model()
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _handle_error(self, error: str) -> str:
        """Handle parsing/validation errors."""
        # Format error prompt
        error_prompt = self.prompts['error'].format(error=error)
        
        # Add to conversation
        self.conversation_history.append({"role": "user", "content": error_prompt})
        
        # Get response
        response = await self._call_generator_model()
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _call_generator_model(self) -> str:
        """Call the generator model with current conversation."""
        provider = self.config.generator_model.provider
        model_id = self.config.generator_model.model_id
        
        # Convert conversation history to ChatMessage format
        messages = []
        for msg in self.conversation_history:
            role = MessageRole[msg['role']] if msg['role'] in ['user', 'assistant', 'system'] else MessageRole.user
            messages.append(ChatMessage(
                content=msg['content'],
                role=role
            ))
        
        # Create prompt
        prompt = Prompt(messages=messages)
        
        # Call the model using InferenceAPI
        responses = await self.inference_api(
            model_id=model_id,
            prompt=prompt,
            max_tokens=16384,
            force_provider=provider if provider in ['anthropic', 'openai'] else None,
        )
        
        # Return the first response's completion
        return responses[0].completion
    
    async def _get_feedback(self, scenario: ScenarioEval) -> List[Dict[str, Any]]:
        """Get feedback from all providers."""
        results = []
        
        for provider in self.feedback_providers:
            try:
                feedback = await provider.provide_feedback(scenario)
                feedback['provider_name'] = provider.name
                results.append(feedback)
            except Exception as e:
                print(f"Error getting feedback from {provider.name}: {e}")
                results.append({
                    'provider_name': provider.name,
                    'feedback': f"Error: {e}",
                    'error': True
                })
        
        return results
    
    def _should_continue(self, feedback_results: List[Dict[str, Any]]) -> bool:
        """
        Determine if we should continue iterating based on feedback.
        
        Simple heuristic: continue if average score < 8.0 or any provider
        explicitly suggests improvements.
        """
        scores = []
        for result in feedback_results:
            if 'score' in result and not result.get('error', False):
                scores.append(result['score'])
        
        # If we have scores, use them
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score < 8.0:
                return True
        
        # Check for explicit improvement suggestions
        improvement_keywords = ['improve', 'better', 'fix', 'issue', 'problem', 'unclear']
        for result in feedback_results:
            if result.get('error', False):
                continue
            feedback_lower = result['feedback'].lower()
            if any(keyword in feedback_lower for keyword in improvement_keywords):
                return True
        
        return False
    
    async def _save_scenario(self, scenario: ScenarioEval, metadata: Dict[str, Any]):
        """Save the generated scenario to the output directory."""
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a subdirectory for this scenario
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_dir = self.config.output_dir / f"scenario_{timestamp}"
        scenario_dir.mkdir()
        
        # Save transcript
        with open(scenario_dir / "transcript.txt", 'w') as f:
            # Render in the expected format
            parts = []
            if scenario.system:
                parts.append(f"<system>\n{scenario.system}\n</system>")
            
            for msg in scenario.messages:
                # Render based on message type
                from prefill_evals.models import TextMessage, ToolCall, ToolResult
                
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
            
            f.write("\n\n".join(parts))
        
        # Save tools if present
        if scenario.tools:
            with open(scenario_dir / "tools.txt", 'w') as f:
                # Render tools in expected format
                parts = []
                for tool in scenario.tools:
                    parts.append(f'<tool name="{tool.name}">')
                    if tool.description:
                        parts.append(f'  <description>{tool.description}</description>')
                    for param in tool.parameters:
                        opt_attr = ' optional="true"' if param.optional else ''
                        parts.append(f'  <parameter name="{param.name}" type="{param.type}"{opt_attr}>')
                        parts.append(f'    {param.description}')
                        parts.append('  </parameter>')
                    parts.append('</tool>')
                f.write("\n".join(parts))
        
        # Save extra items
        if scenario.extra_items:
            for item_name, item_content in scenario.extra_items.items():
                with open(scenario_dir / f"{item_name}.txt", 'w') as f:
                    f.write(item_content)
        
        # Save metadata
        with open(scenario_dir / "generation_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save conversation history
        with open(scenario_dir / "conversation_history.json", 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"Scenario saved to: {scenario_dir}")


async def generate_scenario_from_config(config_path: Path):
    """
    Generate a scenario from a configuration file.
    
    Args:
        config_path: Path to generation config YAML/JSON
    """
    from prefill_evals.config import load_generation_config
    
    # Load config
    config = load_generation_config(config_path)
    
    # Create generator
    generator = ScenarioGenerator(config)
    
    # Generate scenario
    scenario, metadata = await generator.generate_scenario()
    
    return scenario, metadata


async def generate_sweep_from_config(config_path: Path, max_concurrent: int = 5):
    """
    Generate multiple scenarios from a sweep configuration file.
    
    Args:
        config_path: Path to sweep config YAML/JSON
        max_concurrent: Maximum number of concurrent generation tasks
    """
    from prefill_evals.config import load_sweep_config
    import asyncio
    
    # Load sweep configs
    configs = load_sweep_config(config_path)
    
    if not configs:
        print("No configurations found in sweep config")
        return []
    
    total = len(configs)
    print(f"Starting sweep with {total} configurations (max {max_concurrent} concurrent)...")
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_semaphore(config, index):
        async with semaphore:
            print(f"\n[{index}/{total}] Starting generation for: {config.output_dir}")
            
            # Create generator
            generator = ScenarioGenerator(config)
            
            # Generate scenario
            try:
                scenario, metadata = await generator.generate_scenario()
                print(f"✓ [{index}/{total}] Successfully generated: {config.output_dir}")
                return (scenario, metadata)
            except Exception as e:
                print(f"✗ [{index}/{total}] Failed to generate {config.output_dir}: {e}")
                return None
    
    # Create tasks for all configs
    tasks = [
        generate_with_semaphore(config, i) 
        for i, config in enumerate(configs, 1)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (failed generations)
    successful_results = [r for r in results if r is not None]
    
    print(f"\n{'='*60}")
    print(f"Sweep complete! Generated {len(successful_results)}/{total} scenarios")
    print(f"{'='*60}")
    
    return successful_results