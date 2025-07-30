"""
CLI tool for running prefill-style evals.
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from collections import defaultdict
import json
from datetime import datetime
import os
import logging
import traceback
import uuid
import tempfile
import time

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=True, **kwargs):
            self.total = total
            self.n = 0
            print(desc or "Processing...")
        
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"Progress: {self.n}/{self.total}")
        
        def set_description(self, desc):
            print(f"\r{desc}", end='', flush=True)
        
        def close(self):
            print()  # New line at the end

from prefill_evals.config import load_config, EvalConfig, ModelBasedResponseGraderConfig, StringMatchGraderConfig, MultiturnEvaluatorConfig
from prefill_evals.parser import load_scenario_from_dir
from prefill_evals.evaluator import ScenarioEvaluator, EvalResult, ResponseGrading, render_messages
from prefill_evals.models import ScenarioEval, ModelSpec, AgentMessage, TextMessage
from prefill_evals.autograders import ModelBasedResponseGrader, StringMatchGrader
from prefill_evals.multiturn_evaluator import MultiturnEvaluator

def serialize_agent_message(message: AgentMessage) -> Dict[str, Any]:
    """Convert AgentMessage to JSON-serializable dict."""
    return message.to_dict()

def serialize_response_grading(grading: Optional[ResponseGrading]) -> Optional[Dict[str, Any]]:
    """Convert ResponseGrading to JSON-serializable dict."""
    if grading is None:
        return None
    return {
        "score": grading.score,
        "data": grading.data or {}
    }

def serialize_model_spec(model: ModelSpec) -> Dict[str, Any]:
    """Convert ModelSpec to JSON-serializable dict."""
    data = {
        "provider": model.provider,
        "model_id": model.model_id
    }

    if model.max_response_tokens is not None:
        data["max_response_tokens"] = model.max_response_tokens
    
    if model.alias is not None:
        data["alias"] = model.alias
    
    return data

def serialize_eval_result(result: EvalResult, scenario_path: Path) -> Dict[str, Any]:
    """Convert EvalResult to JSON-serializable dict with scenario info."""
    # Serialize responses - each response is a list of messages
    serialized_responses = []
    for response_messages in result.responses:
        serialized_messages = [serialize_agent_message(msg) for msg in response_messages]
        serialized_responses.append({
            "messages": serialized_messages,
            "text": render_messages(response_messages)  # Include rendered text for backward compatibility
        })
    
    return {
        "scenario_path": str(scenario_path),
        "model": serialize_model_spec(result.model),
        "num_runs": result.num_runs,   
        "responses": serialized_responses,
        "grades": [
            [serialize_response_grading(grade) for grade in response_grades]
            for response_grades in result.grades
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def create_error_result(model: ModelSpec, error: Exception, scenario_path: Path) -> Dict[str, Any]:
    """Create an error result for failed evaluations."""
    return {
        "scenario_path": str(scenario_path),
        "model": serialize_model_spec(model),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


class ProgressiveResultsSaver:
    """Save evaluation results progressively as they complete with batched writes."""
    
    def __init__(self, output_path: Path, config: Optional[EvalConfig] = None, 
                 batch_size: int = 10, flush_interval: float = 5.0):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Pending results buffer
        self.pending_results = []
        self.last_flush_time = time.time()
        
        # Background flush task
        self.flush_task = None
        self.should_stop = False
        
        self.data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "config": {
                    "models": [serialize_model_spec(m) for m in config.models] if config else [],
                    "runs_per_model": config.runs_per_model if config else 1,
                    "num_scenarios": len(config.scenarios) if config else 0,
                    "autograders": [g.name for g in config.autograders] if config else []
                }
            },
            "results": []
        }
        
        # If file exists, load existing results
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "results" in existing_data:
                        self.data = existing_data
                        print(f"Loaded {len(self.data['results'])} existing results from {self.output_path}")
                    elif isinstance(existing_data, list):
                        # Old format - convert to new format
                        self.data["results"] = existing_data
                        print(f"Converted {len(existing_data)} existing results to new format")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self._background_flush())
    
    async def _background_flush(self):
        """Background task that periodically flushes pending results to disk."""
        while not self.should_stop:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_if_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")
    
    async def _flush_if_needed(self):
        """Check if flush is needed based on batch size or time."""
        async with self.lock:
            current_time = time.time()
            time_since_flush = current_time - self.last_flush_time
            
            if (len(self.pending_results) >= self.batch_size or 
                (len(self.pending_results) > 0 and time_since_flush >= self.flush_interval)):
                await self._flush_to_disk()
    
    async def _flush_to_disk(self):
        """Flush pending results to disk. Must be called with lock held."""
        if not self.pending_results:
            return
        
        # Add pending results to data
        self.data["results"].extend(self.pending_results)
        self.data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Generate unique temp file name
        temp_dir = self.output_path.parent
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir, text=True)
        
        try:
            # Write to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename
            Path(temp_path).replace(self.output_path)
            
            # Clear pending results
            self.pending_results = []
            self.last_flush_time = time.time()
            
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e
    
    async def save_result(self, result: EvalResult, scenario_path: Path):
        """Add a result to the pending buffer."""
        async with self.lock:
            serialized = serialize_eval_result(result, scenario_path)
            self.pending_results.append(serialized)
            
            # Check if we should flush
            if len(self.pending_results) >= self.batch_size:
                await self._flush_to_disk()
    
    async def save_error(self, error_result: Dict[str, Any]):
        """Add an error result to the pending buffer."""
        async with self.lock:
            self.pending_results.append(error_result)
            
            # Check if we should flush
            if len(self.pending_results) >= self.batch_size:
                await self._flush_to_disk()
    
    async def save_batch(self, results_with_paths: List[Tuple[EvalResult, Path]]):
        """Save multiple results at once."""
        async with self.lock:
            for result, scenario_path in results_with_paths:
                serialized = serialize_eval_result(result, scenario_path)
                self.pending_results.append(serialized)
            
            # Always flush after batch save
            await self._flush_to_disk()
    
    async def finalize(self):
        """Ensure all pending results are written and clean up."""
        self.should_stop = True
        
        # Cancel background task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        async with self.lock:
            await self._flush_to_disk()


def truncate_model_name(model_name: str, max_length: int = 20) -> str:
    """Truncate model name to last N characters if needed."""
    if len(model_name) <= max_length:
        return model_name
    return "..." + model_name[-(max_length - 3):]



class EvaluationProgress:
    """Track evaluation progress and statistics."""
    
    def __init__(self):
        self.completed_evaluations = 0
        self.error_count = 0  # Track individual evaluation errors
        self.model_scores = defaultdict(list)  # model_key -> list of average scores
        self.model_specs = {}  # model_key -> ModelSpec object
        self.errors_by_model = defaultdict(int)  # Track errors per model
        self.lock = asyncio.Lock()
    
    async def update(self, scenario_results: List[EvalResult], scenario_path: Path):
        """Update progress statistics."""
        async with self.lock:
            # Update scores
            for result in scenario_results:
                self.completed_evaluations += 1
                model_key = f"{result.model.provider}/{result.model.model_id}"
                
                # Store the ModelSpec for later reference
                self.model_specs[model_key] = result.model
                
                if result.grades and any(result.grades):
                    # Calculate average score across all graders and runs
                    all_scores = []
                    for response_grades in result.grades:
                        for grade in response_grades:
                            if grade is not None and hasattr(grade, 'score'):
                                all_scores.append(grade.score)
                    
                    if all_scores:
                        avg_score = sum(all_scores) / len(all_scores)
                        self.model_scores[model_key].append(avg_score)
    
    def get_summary(self) -> str:
        """Get current progress summary."""
        summary_parts = [f"✓ {self.completed_evaluations}"]
        
        if self.error_count > 0:
            summary_parts.append(f"⚠ {self.error_count} errors")
        
        # Add average scores
        if self.model_scores:
            avg_scores = []
            for model_key, scores in sorted(self.model_scores.items()):
                if scores:
                    avg = sum(scores) / len(scores)
                    # Get short model name for display using ModelSpec method
                    model_spec = self.model_specs.get(model_key)
                    if model_spec:
                        display_name = model_spec.get_short_name()
                    else:
                        # Fallback if ModelSpec not found (shouldn't happen)
                        display_name = model_key.split("/")[-1][:12]
                    # Add error count for this model if any
                    errors = self.errors_by_model.get(model_key, 0)
                    if errors > 0:
                        avg_scores.append(f"{display_name}: {avg:.1f} (⚠{errors})")
                    else:
                        avg_scores.append(f"{display_name}: {avg:.1f}")
            
            if avg_scores:
                summary_parts.append("│ Avg: " + ", ".join(avg_scores))
        
        return " │ ".join(summary_parts)


def save_eval_results(results: List[EvalResult], output_path: Path):
    """Save evaluation results to a file."""
    # For now, just print the results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\nModel: {result.model.provider}/{result.model.model_id} ({result.model.get_short_name()})")
        print(f"Number of runs: {result.num_runs}")
        
        # Print full responses
        print("\nResponses:")
        for i, response in enumerate(result.responses):
            print(f"\n  --- Run {i+1} ---")
            print(f"  {response}")
            print(f"  --- End Run {i+1} ---")
        
        # Print grades if available
        if result.grades:
            print("\nGrades:")
            for i, response_grades in enumerate(result.grades):
                print(f"\n  Run {i+1}:")
                for grade in response_grades:
                    print(f"    Score: {grade.score}")
                    if grade.data and 'feedback' in grade.data:
                        print(f"    Feedback: {grade.data['feedback']}")
        
        print("\n" + "="*80)
    
    print(f"\n[Results would be saved to: {output_path}]")


async def evaluate_model_on_scenario(
    model_config: Any,
    scenario_path: Path,
    evaluator: ScenarioEvaluator,
    config: EvalConfig,
    semaphore: asyncio.Semaphore,
    progress: EvaluationProgress,
    results_saver: Optional[ProgressiveResultsSaver] = None
) -> Optional[EvalResult]:
    """Evaluate a single model on a single scenario."""
    try:
        # Only hold semaphore for the actual API calls
        async with semaphore:
            result = await evaluator.run_eval(
                model=model_config,
                num_runs=config.runs_per_model
            )
        
        # Save successful result progressively if saver is provided
        if results_saver:
            await results_saver.save_result(result, scenario_path)
        
        return result
        
    except Exception as e:
        # Log the error with full context
        error_msg = f"Error evaluating {model_config.provider}/{model_config.model_id} ({model_config.get_short_name()}) on {scenario_path}: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        # Create and save error result
        if results_saver:
            error_result = create_error_result(
                model_config,
                e,
                scenario_path
            )
            await results_saver.save_error(error_result)
        
        # Track error in progress
        async with progress.lock:
            progress.error_count += 1
            model_key = f"{model_config.provider}/{model_config.model_id}"
            progress.errors_by_model[model_key] += 1
            # Store the ModelSpec for later reference
            progress.model_specs[model_key] = model_config
        
        return None


async def evaluate_model_scenario_pair(
    model_config: Any,
    scenario_path: Path,
    config: EvalConfig,
    semaphore: asyncio.Semaphore,
    progress: EvaluationProgress,
    pbar: Optional['tqdm'] = None,
    results_saver: Optional[ProgressiveResultsSaver] = None
) -> Optional[EvalResult]:
    """Evaluate a single (model, scenario) pair."""
    try:
        # Load and validate scenario
        scenario = load_scenario_from_dir(scenario_path, config.extra_items)
    except Exception as e:
        logger.error(f"Error loading scenario {scenario_path}: {str(e)}")
        if pbar:
            pbar.update(1)
        return None
    
    # Create grader instances from configs
    graders = []
    for grader_config in config.autograders:
        if isinstance(grader_config, ModelBasedResponseGraderConfig):
            graders.append(ModelBasedResponseGrader(grader_config))
        elif isinstance(grader_config, StringMatchGraderConfig):
            graders.append(StringMatchGrader(grader_config))
        else:
            raise ValueError(f"Unknown grader config type: {type(grader_config)}")
    
    # Create appropriate evaluator based on config
    if isinstance(config.evaluator, MultiturnEvaluatorConfig):
        evaluator = MultiturnEvaluator(
            eval=scenario,
            runs_per_model=config.runs_per_model,
            graders=graders,
            user_simulator_model=config.evaluator.user_simulator_model,
            max_turns=config.evaluator.max_turns,
            temperature=config.evaluator.temperature,
            evaluator_system_prompt=config.evaluator.evaluator_system_prompt
        )
    else:
        # Default to ScenarioEvaluator
        evaluator = ScenarioEvaluator(
            eval=scenario,
            runs_per_model=config.runs_per_model,
            graders=graders
        )
    
    # Evaluate this model on this scenario
    result = await evaluate_model_on_scenario(
        model_config, scenario_path, evaluator,
        config, semaphore, progress, results_saver
    )
    
    # Update progress bar
    if pbar:
        await progress.update([result] if result else [], scenario_path)
        pbar.set_description(progress.get_summary())
        pbar.update(1)
    
    return result


async def run_evaluation(config: EvalConfig, max_concurrent: int = 5, output_path: Optional[Path] = None) -> List[EvalResult]:
    """
    Run evaluation given a configuration object.
    
    Args:
        config: EvalConfig object with models, scenarios, and settings
        max_concurrent: Maximum number of model evaluations to run concurrently across all scenarios
        output_path: Optional path to save progressive results
        
    Returns:
        List of EvalResult objects
    """
    # Scenarios is always a list of Path objects now (from glob expansion)
    scenario_paths = config.scenarios
    
    # Create all (model, scenario) pairs
    model_scenario_pairs = [
        (model, scenario_path)
        for scenario_path in scenario_paths
        for model in config.models
    ]
    
    print(f"Found {len(scenario_paths)} scenarios to evaluate")
    print(f"Models: {[f'{m.provider}/{m.model_id} ({m.get_short_name()})' for m in config.models]}")
    print(f"Running with max {max_concurrent} concurrent model evaluations")
    print(f"Total evaluations to run: {len(model_scenario_pairs)}")
    
    # Create progressive results saver if output path provided
    results_saver = None
    if output_path:
        results_saver = ProgressiveResultsSaver(output_path, config)
        print(f"Saving results progressively to: {output_path}")
    
    print()
    
    # Create progress tracker
    progress = EvaluationProgress()
    
    # Create global semaphore to limit concurrent model evaluations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create progress bar for (model, scenario) pairs
    pbar = tqdm(total=len(model_scenario_pairs), desc="Starting evaluations...", 
                unit="eval", dynamic_ncols=True)
    
    # Create tasks for all (model, scenario) pairs
    tasks = [
        evaluate_model_scenario_pair(model_config, scenario_path, config, semaphore, progress, pbar, results_saver)
        for model_config, scenario_path in model_scenario_pairs
    ]
    
    # Run all tasks concurrently
    try:
        results = await asyncio.gather(*tasks)
    finally:
        pbar.close()
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Finalize results saver if it exists
    if results_saver:
        await results_saver.finalize()
        print(f"\nAll results saved to: {output_path}")
    
    # Print final summary
    print(f"\n\nEvaluation Complete!")
    print(f"Total evaluations completed: {len(results)}/{len(model_scenario_pairs)}")
    if progress.error_count > 0:
        print(f"Evaluation errors: {progress.error_count}")
    
    # Print average scores by model
    if progress.model_scores:
        print("\nAverage Scores by Model:")
        for model, scores in sorted(progress.model_scores.items()):
            if scores:
                avg = sum(scores) / len(scores)
                errors = progress.errors_by_model.get(model, 0)
                if errors > 0:
                    print(f"  {model}: {avg:.2f} (n={len(scores)}, errors={errors})")
                else:
                    print(f"  {model}: {avg:.2f} (n={len(scores)})")
    
    return results


async def run_and_save_evaluation(config_path: Path, output_dir: Optional[Path] = None, max_concurrent: int = 5) -> List[EvalResult]:
    """
    Run evaluation from a configuration file and save results.
    
    Args:
        config_path: Path to the evaluation config file (YAML or JSON)
        output_dir: Optional directory to save results (defaults to config directory)
        
    Returns:
        List of EvalResult objects
    """
    # Load configuration
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = config_path.parent / "results"
    
    # Create output file path with timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_results_{config_path.stem}_{timestamp}.json"
    
    # Run evaluation with progressive saving
    results = await run_evaluation(config, max_concurrent, output_path=output_file)
    
    print(f"\nEvaluation results saved to: {output_file}")
    
    # Also print summary
    if results:
        print(f"\nTotal scenarios evaluated: {len(set(r.scenario_path for r in results if hasattr(r, 'scenario_path')))}")
        print(f"Total model evaluations: {len(results)}")
    
    print("\nEvaluation complete!")
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prefill-based evaluation tools"
    )
    
    # Add global argument for env file
    parser.add_argument(
        "--env",
        type=Path,
        default=Path("/Users/cofibration/Documents/fellows-projects/rl-character-science-workspace/.env"),
        help="Path to .env file (default: /Users/cofibration/Documents/fellows-projects/rl-character-science-workspace/.env)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluations on language models')
    eval_parser.add_argument(
        "config",
        type=Path,
        help="Path to evaluation config file (YAML or JSON)"
    )
    eval_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Directory to save results (defaults to config_dir/results)"
    )
    eval_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent model evaluations across all scenarios (default: 5)"
    )
    
    # Generate subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate scenario evaluations')
    gen_parser.add_argument(
        "config",
        type=Path,
        help="Path to generation config file (YAML or JSON)"
    )
    
    # Generate sweep subcommand
    sweep_parser = subparsers.add_parser('generate-sweep', help='Generate multiple scenarios from a sweep config')
    sweep_parser.add_argument(
        "config",
        type=Path,
        help="Path to sweep generation config file (YAML or JSON)"
    )
    sweep_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent generation tasks (default: 5)"
    )
    
    # View results subcommand
    view_parser = subparsers.add_parser('view-results', help='Generate HTML viewer for evaluation results')
    view_parser.add_argument(
        "results",
        type=Path,
        help="Path to evaluation results JSON file"
    )
    view_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results_viewer.html"),
        help="Output HTML file path (default: results_viewer.html)"
    )
    view_parser.add_argument(
        "--scenario-filter",
        type=str,
        help="Glob pattern to filter scenarios (e.g., '*/problem_*')"
    )
    view_parser.add_argument(
        "--model-filter",
        type=str,
        help="Glob pattern to filter models (e.g., 'gpt-4*')"
    )
    view_parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include results with errors (default: exclude)"
    )
    view_parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum grade score to include"
    )
    view_parser.add_argument(
        "--max-score",
        type=float,
        help="Maximum grade score to include"
    )
    view_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of results per scenario"
    )
    
    # View scenario subcommand
    scenario_parser = subparsers.add_parser('view-scenario', help='Generate HTML viewer for a single scenario')
    scenario_parser.add_argument(
        "scenario",
        type=Path,
        help="Path to scenario directory"
    )
    scenario_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("scenario_viewer.html"),
        help="Output HTML file path (default: scenario_viewer.html)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from specified .env file
    if args.env and args.env.exists():
        load_dotenv(args.env)
    
    if args.command == 'evaluate':
        # Run the async evaluation
        asyncio.run(run_and_save_evaluation(args.config, args.output_dir, args.max_concurrent))
    elif args.command == 'generate':
        # Run the async generation
        from prefill_evals.generator import generate_scenario_from_config
        asyncio.run(generate_scenario_from_config(args.config))
    elif args.command == 'generate-sweep':
        # Run the async sweep generation
        from prefill_evals.generator import generate_sweep_from_config
        asyncio.run(generate_sweep_from_config(args.config, max_concurrent=args.max_concurrent))
    elif args.command == 'view-results':
        # Generate HTML viewer for results
        from prefill_evals.results_viewer import generate_results_html
        generate_results_html(
            results_path=args.results,
            output_path=args.output,
            scenario_filter=args.scenario_filter,
            model_filter=args.model_filter,
            include_errors=args.include_errors,
            min_score=args.min_score,
            max_score=args.max_score,
            limit=args.limit
        )
    elif args.command == 'view-scenario':
        # Generate HTML viewer for a single scenario
        from prefill_evals.results_viewer import generate_scenario_html
        generate_scenario_html(
            scenario_path=args.scenario,
            output_path=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
