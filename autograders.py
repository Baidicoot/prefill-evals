from typing import Optional, Dict, Any, List, Tuple

from prefill_evals.evaluator import ResponseGrader, ResponseGrading
from prefill_evals.config import ModelBasedResponseGraderConfig
from prefill_evals.models import ScenarioEval
from prefill_evals.parser import parse_xml_tags, render_transcript

from safetytooling.apis import InferenceAPI

from pathlib import Path

def parse_autograder_prompt(path: Path) -> Tuple[str, Optional[str]]:
    with open(path, "r") as f:
        prompt_str = f.read()
    
    system_prompt = None
    if "System:\n" in prompt_str:
        system_prompt = prompt_str.split("System:\n")[-1].split("User:\n")[0]
    prompt = prompt_str.split("User:\n")[-1]

    return prompt, system_prompt

class ModelBasedResponseGrader(ResponseGrader):
    """
    Grade a model response with another model.
    """

    def __init__(self, config: ModelBasedResponseGraderConfig):
        self.config = config
        self.api = InferenceAPI()

        self.prompt, self.system_prompt = parse_autograder_prompt(self.config.template_file)

    async def grade(self, response: str, eval: ScenarioEval) -> Optional[ResponseGrading]:
        input_items = eval.extra_items if eval.extra_items is not None else {}
        input_items["response"] = response
        input_items["transcript"] = render_transcript(eval)
        
        formated_prompt = self.prompt.format(**input_items)
        response = await self.api.ask_single_question(
            question=formated_prompt,
            system_prompt=self.system_prompt,
            model_id=self.config.grader.model_id,
            force_provider=self.config.grader.provider
        )
        output_items = self.config.extra_items if self.config.extra_items is not None else []
        output_items.append("grade")
        # Use the general XML parser for autograder output
        parsed_output = parse_xml_tags(response[0], output_items, required_tags=['grade'])

        if "grade" not in parsed_output:
            return None

        try:
            grade = float(parsed_output["grade"])
        except:
            return None
        
        parsed_output.pop("grade")
        
        return ResponseGrading(score=grade, data=parsed_output)