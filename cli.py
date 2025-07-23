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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

from prefill_evals.config import load_config, EvalConfig
from prefill_evals.parser import load_scenario_from_dir
from prefill_evals.evaluator import ScenarioEvaluator, EvalResult, ResponseGrading
from prefill_evals.autograders import ModelBasedResponseGrader


def serialize_response_grading(grading: Optional[ResponseGrading]) -> Optional[Dict[str, Any]]:
    """Convert ResponseGrading to JSON-serializable dict."""
    if grading is None:
        return None
    return {
        "score": grading.score,
        "data": grading.data or {}
    }


def serialize_eval_result(result: EvalResult, scenario_path: Path) -> Dict[str, Any]:
    """Convert EvalResult to JSON-serializable dict with scenario info."""
    return {
        "scenario_path": str(scenario_path),
        "provider": result.provider,
        "model_id": result.model_id,
        "num_runs": result.num_runs,
        "responses": result.responses,
        "grades": [
            [serialize_response_grading(grade) for grade in response_grades]
            for response_grades in result.grades
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def create_error_result(provider: str, model_id: str, error: Exception, scenario_path: Path) -> Dict[str, Any]:
    """Create an error result for failed evaluations."""
    return {
        "scenario_path": str(scenario_path),
        "provider": provider,
        "model_id": model_id,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


class ProgressiveResultsSaver:
    """Save evaluation results progressively as they complete."""
    
    def __init__(self, output_path: Path, config: Optional[EvalConfig] = None):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "config": {
                    "models": [{"provider": m.provider, "model_id": m.model_id} for m in config.models] if config else [],
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
    
    async def save_result(self, result: EvalResult, scenario_path: Path):
        """Save a single evaluation result to the file."""
        async with self.lock:
            serialized = serialize_eval_result(result, scenario_path)
            self.data["results"].append(serialized)
            self.data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Write the entire data structure atomically
            temp_path = self.output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.output_path)
    
    async def save_error(self, error_result: Dict[str, Any]):
        """Save an error result to the file."""
        async with self.lock:
            self.data["results"].append(error_result)
            self.data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Write the entire data structure atomically
            temp_path = self.output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.output_path)
    
    async def save_batch(self, results_with_paths: List[Tuple[EvalResult, Path]]):
        """Save multiple results at once."""
        async with self.lock:
            for result, scenario_path in results_with_paths:
                serialized = serialize_eval_result(result, scenario_path)
                self.data["results"].append(serialized)
            
            self.data["metadata"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Write the entire data structure atomically
            temp_path = self.output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.output_path)


class EvaluationProgress:
    """Track evaluation progress and statistics."""
    
    def __init__(self):
        self.completed_scenarios = 0
        self.failed_scenarios = 0
        self.error_count = 0  # Track individual evaluation errors
        self.model_scores = defaultdict(list)  # model_id -> list of scores
        self.grader_scores = defaultdict(list)  # grader_name -> list of scores
        self.errors_by_model = defaultdict(int)  # Track errors per model
        self.lock = asyncio.Lock()
    
    async def update(self, scenario_results: List[EvalResult], scenario_path: Path, failed: bool = False):
        """Update progress statistics."""
        async with self.lock:
            if failed:
                self.failed_scenarios += 1
            else:
                self.completed_scenarios += 1
                
                # Update scores
                for result in scenario_results:
                    model_key = f"{result.provider}/{result.model_id}"
                    
                    if result.grades and result.grades[0]:
                        # Average scores across runs for this scenario
                        for grader_idx in range(len(result.grades[0])):
                            scores = []
                            for response_grades in result.grades:
                                if grader_idx < len(response_grades) and response_grades[grader_idx] is not None:
                                    scores.append(response_grades[grader_idx].score)
                            
                            if scores:
                                avg_score = sum(scores) / len(scores)
                                self.model_scores[model_key].append(avg_score)
                                # Assuming single grader for now
                                self.grader_scores["deceptiveness"].append(avg_score)
    
    def get_summary(self) -> str:
        """Get current progress summary."""
        summary_parts = [f"✓ {self.completed_scenarios}"]
        
        if self.failed_scenarios > 0:
            summary_parts.append(f"✗ {self.failed_scenarios}")
        
        if self.error_count > 0:
            summary_parts.append(f"⚠ {self.error_count} errors")
        
        # Add average scores
        if self.model_scores:
            avg_scores = []
            for model, scores in sorted(self.model_scores.items()):
                if scores:
                    avg = sum(scores) / len(scores)
                    # Add error count for this model if any
                    errors = self.errors_by_model.get(model, 0)
                    if errors > 0:
                        avg_scores.append(f"{model}: {avg:.1f} (⚠{errors})")
                    else:
                        avg_scores.append(f"{model}: {avg:.1f}")
            
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
        print(f"\nModel: {result.provider}/{result.model_id}")
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


async def evaluate_single_scenario(
    scenario_path: Path,
    config: EvalConfig,
    semaphore: asyncio.Semaphore,
    progress: EvaluationProgress,
    pbar: Optional['tqdm'] = None,
    results_saver: Optional[ProgressiveResultsSaver] = None
) -> List[EvalResult]:
    """Evaluate a single scenario with all models."""
    async with semaphore:
        try:
            # Load and validate scenario
            scenario = load_scenario_from_dir(scenario_path, config.extra_items)
            
        except Exception as e:
            await progress.update([], scenario_path, failed=True)
            if pbar:
                pbar.set_description(progress.get_summary())
                pbar.update(1)
            return []
        
        # Create grader instances from configs
        graders = []
        for grader_config in config.autograders:
            graders.append(ModelBasedResponseGrader(grader_config))
        
        # Create evaluator
        evaluator = ScenarioEvaluator(
            eval=scenario,
            runs_per_model=config.runs_per_model,
            graders=graders
        )
        
        scenario_results = []
        
        # Run each model with error handling
        for model_config in config.models:
            try:
                result = await evaluator.run_eval(
                    provider=model_config.provider,
                    model_id=model_config.model_id,
                    num_runs=config.runs_per_model
                )
                scenario_results.append(result)
                
                # Save successful result progressively if saver is provided
                if results_saver:
                    await results_saver.save_result(result, scenario_path)
                    
            except Exception as e:
                # Log the error with full context
                error_msg = f"Error evaluating {model_config.provider}/{model_config.model_id} on {scenario_path}: {str(e)}"
                logger.error(error_msg)
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                
                # Create and save error result
                if results_saver:
                    error_result = create_error_result(
                        model_config.provider,
                        model_config.model_id,
                        e,
                        scenario_path
                    )
                    await results_saver.save_error(error_result)
                
                # Track error in progress
                async with progress.lock:
                    progress.error_count += 1
                    progress.errors_by_model[f"{model_config.provider}/{model_config.model_id}"] += 1
        
        # Update progress
        await progress.update(scenario_results, scenario_path)
        if pbar:
            pbar.set_description(progress.get_summary())
            pbar.update(1)
        
        return scenario_results


async def run_evaluation(config: EvalConfig, max_concurrent: int = 5, output_path: Optional[Path] = None) -> List[EvalResult]:
    """
    Run evaluation given a configuration object.
    
    Args:
        config: EvalConfig object with models, scenarios, and settings
        max_concurrent: Maximum number of scenarios to evaluate concurrently
        output_path: Optional path to save progressive results
        
    Returns:
        List of EvalResult objects
    """
    # Scenarios is always a list of Path objects now (from glob expansion)
    scenario_paths = config.scenarios
    
    print(f"Found {len(scenario_paths)} scenarios to evaluate")
    print(f"Running with max {max_concurrent} concurrent evaluations")
    print(f"Models: {[f'{m.provider}/{m.model_id}' for m in config.models]}")
    
    # Create progressive results saver if output path provided
    results_saver = None
    if output_path:
        results_saver = ProgressiveResultsSaver(output_path, config)
        print(f"Saving results progressively to: {output_path}")
    
    print()
    
    # Create progress tracker
    progress = EvaluationProgress()
    
    # Create semaphore to limit concurrent evaluations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create progress bar
    pbar = tqdm(total=len(scenario_paths), desc="Starting evaluations...", 
                unit="scenario", dynamic_ncols=True)
    
    # Create tasks for all scenarios
    tasks = [
        evaluate_single_scenario(scenario_path, config, semaphore, progress, pbar, results_saver)
        for scenario_path in scenario_paths
    ]
    
    # Run all tasks concurrently
    try:
        results_lists = await asyncio.gather(*tasks)
    finally:
        pbar.close()
    
    # Print final summary
    print(f"\n\nEvaluation Complete!")
    print(f"Scenarios evaluated: {progress.completed_scenarios}")
    if progress.failed_scenarios > 0:
        print(f"Scenarios failed: {progress.failed_scenarios}")
    if progress.error_count > 0:
        print(f"Individual evaluation errors: {progress.error_count}")
    
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
    
    # Flatten results
    all_results = []
    for results in results_lists:
        all_results.extend(results)
    
    return all_results


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
        help="Maximum number of concurrent scenario evaluations (default: 5)"
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
