"""
Configuration data models and loading for conversation pre-fill eval generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import json
import glob

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
    max_concurrent: Optional[int] = None  # None means no limit

@dataclass
class StringMatchGraderConfig:
    name: str
    expected_strings: List[str]  # List of strings to match against
    match_type: str = "exact"  # "exact", "contains", "fuzzy"
    case_sensitive: bool = False
    fuzzy_threshold: float = 0.8  # For fuzzy matching (0.0 to 1.0)
    score_on_match: float = 1.0  # Score to assign when matched
    score_on_no_match: float = 0.0  # Score to assign when not matched
    match_all: bool = False  # If True, all expected_strings must match

@dataclass
class EvalConfig:
    models: List[ModelConfig]
    runs_per_model: int
    autograders: List[Union[ModelBasedResponseGraderConfig, StringMatchGraderConfig]]
    scenarios: List[Path]  # Always a list now, populated via glob expansion
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
    output_dir: Path
    max_iterations: int = 3

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
        grader_type = grader_data.get('type', 'model')  # Default to model-based for backward compatibility
        
        if grader_type == 'string_match':
            # String match grader
            autograders.append(StringMatchGraderConfig(
                name=grader_data['name'],
                expected_strings=grader_data['expected_strings'],
                match_type=grader_data.get('match_type', 'exact'),
                case_sensitive=grader_data.get('case_sensitive', False),
                fuzzy_threshold=grader_data.get('fuzzy_threshold', 0.8),
                score_on_match=grader_data.get('score_on_match', 1.0),
                score_on_no_match=grader_data.get('score_on_no_match', 0.0),
                match_all=grader_data.get('match_all', False)
            ))
        elif grader_type == 'model':
            # Model-based grader
            grader_config = ModelConfig(
                provider=grader_data['grader']['provider'],
                model_id=grader_data['grader']['model_id']
            )
            autograders.append(ModelBasedResponseGraderConfig(
                name=grader_data['name'],
                grader=grader_config,
                template_file=Path(grader_data['template_file']),
                extra_items=grader_data.get('extra_items'),
                max_concurrent=grader_data.get('max_concurrent')
            ))
        else:
            raise ValueError(f"Unknown autograder type: {grader_type}. Supported types: 'model', 'string_match'")
    
    # Parse scenarios - always treat as glob patterns (direct paths work too)
    scenarios_data = config_data.get('scenarios')
    if isinstance(scenarios_data, str):
        # Single glob pattern
        matched_paths = glob.glob(scenarios_data, recursive=True)
        scenarios = [Path(p) for p in matched_paths if Path(p).is_dir()]
    elif isinstance(scenarios_data, list):
        # Multiple glob patterns
        scenarios = []
        for pattern in scenarios_data:
            matched_paths = glob.glob(pattern, recursive=True)
            scenarios.extend([Path(p) for p in matched_paths if Path(p).is_dir()])
    else:
        raise ValueError("scenarios must be a glob pattern string or list of glob patterns")
    
    if not scenarios:
        raise ValueError(f"No directories found matching pattern(s): {scenarios_data}")
    
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
    
    # Handle as single config
    return create_generation_config(config_data)


def create_generation_config(config_data: Dict[str, Any]) -> GenerationConfig:
    """
    Create a GenerationConfig from a configuration dictionary.
    
    Args:
        config_data: Dictionary containing configuration data
        
    Returns:
        GenerationConfig object
    """
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


def expand_sweep_config(config_data: Dict[str, Any], base_path: Path) -> List[GenerationConfig]:
    """
    Expand a sweep configuration into multiple GenerationConfig objects.
    
    Args:
        config_data: Dictionary containing base and sweeps sections
        base_path: Base path for resolving relative paths
        
    Returns:
        List of GenerationConfig objects
    """
    configs = []
    base = config_data.get('base', {})
    sweeps = config_data.get('sweeps', {})
    
    # Process each input directory pattern
    for pattern in sweeps.get('input_dirs', []):
        # Handle both absolute and relative paths
        if Path(pattern).is_absolute():
            search_pattern = pattern
        else:
            search_pattern = str(base_path / pattern)
        
        # Find all directories matching the pattern
        for dir_path_str in glob.glob(search_pattern):
            dir_path = Path(dir_path_str)
            if dir_path.is_dir():
                # Create config for this directory
                cfg = base.copy()
                cfg['seed_items'] = {}
                
                # Find seed items based on pattern
                for item_name, filename in sweeps.get('seed_items_pattern', {}).items():
                    item_path = dir_path / filename
                    if item_path.exists():
                        cfg['seed_items'][item_name] = str(item_path)
                
                # Skip if no seed items found
                if not cfg['seed_items']:
                    print(f"Warning: No seed items found in {dir_path}, skipping")
                    continue
                
                # Set output directory
                output_template = sweeps.get('output_dir_template', 'outputs/{folder_name}')
                
                # Calculate relative path from base_path for {folder_path}
                try:
                    relative_path = dir_path.relative_to(base_path)
                except ValueError:
                    # If dir_path is not relative to base_path, use absolute path
                    relative_path = dir_path
                
                # Replace both {folder_name} and {folder_path} placeholders
                output_dir = output_template.replace('{folder_name}', dir_path.name)
                output_dir = output_dir.replace('{folder_path}', str(relative_path))
                cfg['output_dir'] = output_dir
                
                # Create and add config
                configs.append(create_generation_config(cfg))
    
    if not configs:
        raise ValueError("No valid configurations generated from sweep patterns")
    
    return configs


def load_sweep_config(config_path: Path) -> List[GenerationConfig]:
    """
    Load a sweep configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the sweep configuration file
        
    Returns:
        List of GenerationConfig objects
        
    Raises:
        ValueError: If file doesn't exist, has unsupported format, or isn't a sweep config
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
    
    # Ensure this is a sweep config
    if 'base' not in config_data or 'sweeps' not in config_data:
        raise ValueError("Sweep config must contain 'base' and 'sweeps' sections")
    
    return expand_sweep_config(config_data, config_path.parent)