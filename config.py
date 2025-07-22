"""
Configuration data models and loading for conversation pre-fill eval generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import json

@dataclass
class ModelConfig:
    provider: str
    model_id: str

@dataclass
class ModelBasedResponseGraderConfig:
    name: str
    grader: ModelConfig
    template_file: Path
    extra_items: Optional[List[str]] = None

@dataclass
class EvalConfig:
    models: List[ModelConfig]
    runs_per_model: int
    autograders: List[ModelBasedResponseGraderConfig]
    scenarios: Union[Path, List[Path]]
    extra_items: Optional[List[str]] = None


@dataclass
class ScenarioFeedbackConfig:
    name: str
    model: ModelConfig
    template_file: Path


@dataclass
class GenerationConfig:
    generator_model: ModelConfig
    prompts_dir: Path  # Directory containing standard prompt files
    seed_items: Dict[str, Path]  # item_name -> file_path
    extra_items_to_generate: List[str]
    feedback_providers: List[ScenarioFeedbackConfig]
    max_iterations: int = 3
    output_dir: Path

def load_config(config_path: Path) -> EvalConfig:
    """
    Load evaluation configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        EvalConfig object with loaded configuration
        
    Raises:
        ValueError: If file doesn't exist or has unsupported format
    """
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")
    
    # Load the config file based on extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Parse models
    models = []
    for model_data in config_data.get('models', []):
        models.append(ModelConfig(
            provider=model_data['provider'],
            model_id=model_data['model_id']
        ))
    
    # Parse autograders
    autograders = []
    for grader_data in config_data.get('autograders', []):
        grader_config = ModelConfig(
            provider=grader_data['grader']['provider'],
            model_id=grader_data['grader']['model_id']
        )
        autograders.append(ModelBasedResponseGraderConfig(
            name=grader_data['name'],
            grader=grader_config,
            template_file=Path(grader_data['template_file']),
            extra_items=grader_data.get('extra_items')
        ))
    
    # Parse scenarios - can be a single path or list of paths
    scenarios_data = config_data.get('scenarios')
    if isinstance(scenarios_data, str):
        scenarios = Path(scenarios_data)
    elif isinstance(scenarios_data, list):
        scenarios = [Path(s) for s in scenarios_data]
    else:
        raise ValueError("scenarios must be a path string or list of path strings")
    
    return EvalConfig(
        models=models,
        runs_per_model=config_data.get('runs_per_model', 1),
        autograders=autograders,
        scenarios=scenarios,
        extra_items=config_data.get('extra_items')
    )


def load_generation_config(config_path: Path) -> GenerationConfig:
    """
    Load generation configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        GenerationConfig object with loaded configuration
        
    Raises:
        ValueError: If file doesn't exist or has unsupported format
    """
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")
    
    # Load the config file based on extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Parse generator model
    gen_model_data = config_data.get('generator_model', {})
    generator_model = ModelConfig(
        provider=gen_model_data['provider'],
        model_id=gen_model_data['model_id']
    )
    
    # Parse seed items
    seed_items = {}
    for item_name, item_path in config_data.get('seed_items', {}).items():
        seed_items[item_name] = Path(item_path)
    
    # Parse feedback providers
    feedback_providers = []
    for feedback_data in config_data.get('feedback_providers', []):
        feedback_model = ModelConfig(
            provider=feedback_data['model']['provider'],
            model_id=feedback_data['model']['model_id']
        )
        feedback_providers.append(ScenarioFeedbackConfig(
            name=feedback_data['name'],
            model=feedback_model,
            template_file=Path(feedback_data['template_file'])
        ))
    
    return GenerationConfig(
        generator_model=generator_model,
        prompts_dir=Path(config_data['prompts_dir']),
        seed_items=seed_items,
        extra_items_to_generate=config_data.get('extra_items_to_generate', []),
        feedback_providers=feedback_providers,
        max_iterations=config_data.get('max_iterations', 3),
        output_dir=Path(config_data['output_dir'])
    )