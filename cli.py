"""
CLI tool for running prefill-style evals.
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path("/Users/cofibration/Documents/fellows-projects/rl-character-science-workspace/.env")
if env_path.exists():
    load_dotenv(env_path)

from prefill_evals.config import load_config, EvalConfig
from prefill_evals.parser import load_scenario_from_dir
from prefill_evals.evaluator import ScenarioEvaluator, EvalResult


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


async def run_evaluation(config: EvalConfig) -> List[EvalResult]:
    """
    Run evaluation given a configuration object.
    
    Args:
        config: EvalConfig object with models, scenarios, and settings
        
    Returns:
        List of EvalResult objects
    """
    # Handle scenarios - can be a single directory or list of directories
    scenario_paths = []
    if isinstance(config.scenarios, Path):
        if config.scenarios.is_file():
            # If it's a file, assume it contains a list of scenario directories
            raise NotImplementedError("Loading scenarios from list file not yet implemented")
        else:
            # Single scenario directory
            scenario_paths = [config.scenarios]
    else:
        # List of scenario paths
        scenario_paths = config.scenarios
    
    all_results = []
    
    # Run evaluation for each scenario
    for scenario_path in scenario_paths:
        print(f"\nLoading scenario from {scenario_path}")
        scenario = load_scenario_from_dir(scenario_path, config.extra_items)
        
        # Create evaluator
        evaluator = ScenarioEvaluator(
            eval=scenario,
            runs_per_model=config.runs_per_model,
            graders=config.autograders
        )
        
        # Run each model
        for model_config in config.models:
            print(f"\nRunning {model_config.provider} model: {model_config.model_id}")
            result = await evaluator.run_eval(
                provider=model_config.provider,
                model_id=model_config.model_id,
                num_runs=config.runs_per_model
            )
            all_results.append(result)
            
            # Print summary
            print(f"  Completed {len(result.responses)} runs")
            if result.grades and result.grades[0]:
                avg_scores = {}
                for grader_idx, grader in enumerate(config.autograders):
                    scores = [response_grades[grader_idx].score for response_grades in result.grades]
                    avg_score = sum(scores) / len(scores)
                    avg_scores[grader.name] = avg_score
                    print(f"  Average {grader.name} score: {avg_score:.2f}")
    
    return all_results


async def run_and_save_evaluation(config_path: Path, output_dir: Optional[Path] = None) -> List[EvalResult]:
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
    
    # Run evaluation
    results = await run_evaluation(config)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_results_{config_path.stem}.json"
    print(f"\nSaving results to {output_file}")
    save_eval_results(results, output_file)
    
    print("\nEvaluation complete!")
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prefill-based evaluation tools"
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
    
    if args.command == 'evaluate':
        # Run the async evaluation
        asyncio.run(run_and_save_evaluation(args.config, args.output_dir))
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
