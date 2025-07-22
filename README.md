# Prefill-based evaluations

A framework for evaluating language models using conversation prefills with tool use support.

## Overview

This framework allows you to:
- Define conversation scenarios with user/agent interactions and tool calls
- Run multiple language models on these scenarios
- Grade model responses using configurable autograders
- Compare performance across different models

## Installation

```bash
pip install -r requirements.txt
```

Set up API keys in your environment or `.env` file:
```bash
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## Quick Start

1. Create a scenario directory with conversation files:
```
scenarios/my_scenario/
├── transcript.txt    # Required: The conversation
├── tools.txt        # Optional: Tool definitions
└── context.txt      # Optional: Additional context
```

2. Create an evaluation config:
```yaml
# eval_config.yaml
models:
  - provider: anthropic
    model_id: claude-3-5-sonnet-20241022
  - provider: openai
    model_id: gpt-4

runs_per_model: 3

scenarios: scenarios/my_scenario/

autograders:
  - name: quality_grader
    grader:
      provider: anthropic
      model_id: claude-3-5-sonnet-20241022
    template_file: graders/quality_template.txt
```

3. Run the evaluation:
```bash
python -m prefill_evals.cli evaluate eval_config.yaml
```

## Scenario Format

### Transcript Format (transcript.txt)

Conversations use XML tags for structure:

```xml
<system>
You are a helpful assistant.
</system>

<user>
What's the weather like?
</user>

<agent>
I'll check the weather for you.
</agent>

<tool_call:get_weather>
  <argument:location>San Francisco</argument:location>
</tool_call:get_weather>

<tool_result>
The weather in San Francisco is 68°F and sunny.
</tool_result>

<agent>
The weather in San Francisco is currently 68°F and sunny. It's a beautiful day!
</agent>

<user>
Thanks!
</user>
```

### Tool Definitions (tools.txt)

Define available tools in XML format:

```xml
<tool name="get_weather">
  <description>Get current weather for a location</description>
  <parameter name="location" type="string">City name or coordinates</parameter>
  <parameter name="units" type="string" optional="true">Temperature units (celsius/fahrenheit)</parameter>
</tool>
```

## Configuration

### Evaluation Config Structure

```yaml
models:                    # List of models to evaluate
  - provider: anthropic    # Provider: "anthropic" or "openai"
    model_id: model-name   # Model identifier

runs_per_model: 1         # Number of times to run each model

scenarios:                # Path to scenario directory or list of paths
  - scenarios/scenario1/
  - scenarios/scenario2/

autograders:              # Optional: Model-based response graders
  - name: grader_name
    grader:
      provider: anthropic
      model_id: model-name
    template_file: path/to/template.txt
    extra_items:          # Optional: Additional files to load
      - context

extra_items:              # Optional: Additional files to load for all scenarios
  - context
```

## API Usage

For programmatic use:

```python
import asyncio
from prefill_evals.config import load_config
from prefill_evals.cli import run_evaluation

# Load configuration
config = load_config("eval_config.yaml")

# Run evaluation
results = asyncio.run(run_evaluation(config))

# Process results
for result in results:
    print(f"{result.model_id}: {result.responses}")
```

## File Structure

```
prefill_evals/
├── __init__.py
├── cli.py          # Command-line interface
├── config.py       # Configuration data models
├── evaluator.py    # Model evaluation logic
├── models.py       # Message and tool data models
└── parser.py       # Transcript parsing utilities
```

## Extending the Framework

### Adding a New Model Provider

1. Add the provider handling in `evaluator.py`:
```python
async def run_custom_model(self, model_id: str, num_runs: int = 1) -> List[str]:
    # Implement your model API calls
    pass
```

2. Update the `run_model` method to route to your provider.

### Custom Response Graders

Create a grader by extending `ResponseGrader`:

```python
from prefill_evals.evaluator import ResponseGrader, ResponseGrading

class CustomGrader(ResponseGrader):
    async def grade(self, response: str, eval: ScenarioEval) -> ResponseGrading:
        # Implement grading logic
        return ResponseGrading(score=0.8, data={"feedback": "Good"})
```

## Scenario Generation

The framework includes a powerful generation pipeline for creating new evaluation scenarios using language models.

### Generation Overview

1. **Seed Items**: Start with high-level descriptions and requirements
2. **Initial Generation**: Model creates scenario based on seed items
3. **Validation**: Automatic parsing and validation of generated content
4. **Feedback**: Model-based feedback providers evaluate the scenario
5. **Revision**: Iterative improvement based on feedback
6. **Output**: Final scenario saved in standard format

### Running Generation

```bash
python -m prefill_evals.cli generate generation_config.yaml
```

### Generation Configuration

```yaml
generator_model:
  provider: anthropic
  model_id: claude-3-5-sonnet-20241022

prompts_dir: generation_prompts/

seed_items:
  scenario_description: seeds/description.txt
  context: seeds/context.txt
  requirements: seeds/requirements.txt

extra_items_to_generate:
  - target_behavior
  - misdeed

feedback_providers:
  - name: coherence
    model:
      provider: anthropic
      model_id: claude-3-5-sonnet-20241022
    template_file: feedback_templates/coherence.txt

max_iterations: 3
output_dir: generated_scenarios/
```

### Prompt Templates

Create a prompts directory with these standard files:

```
generation_prompts/
├── system.txt      # System prompt for generator
├── initial.txt     # Initial generation ({seed_item} variables)
├── revision.txt    # Revision based on feedback ({feedback})
└── error.txt       # Error correction ({error})
```

### Feedback Templates

Feedback providers evaluate generated scenarios:

```
feedback_templates/
├── coherence.txt   # Check logical flow
├── realism.txt     # Evaluate plausibility
└── coverage.txt    # Verify requirements met
```

Templates receive `{transcript}`, `{tools}`, and any `{extra_item}` variables.

### Example Workflow

1. Create seed items describing your scenario
2. Set up feedback templates for quality checks
3. Configure generation parameters
4. Run generation - the system will iterate until quality thresholds are met
5. Find your generated scenario in the output directory

## Notes

- Tool calls must have matching tool results in the transcript
- Transcripts must end with a user message
- The framework validates scenarios before running evaluations
- Results are saved in JSON format for further analysis
- Generation supports multiple iterations with model-based feedback