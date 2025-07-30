from typing import Optional, Dict, Any, List, Tuple
import asyncio
import re
from difflib import SequenceMatcher

from prefill_evals.evaluator import ResponseGrader, ResponseGrading, render_messages
from prefill_evals.config import ModelBasedResponseGraderConfig, StringMatchGraderConfig, BinaryChoiceGraderConfig
from prefill_evals.models import ScenarioEval, AgentMessage
from prefill_evals.parser import parse_xml_tags, render_transcript

from safetytooling.apis import InferenceAPI

from pathlib import Path

# Global InferenceAPI instance - shared across all graders
_inference_api = None

def get_inference_api() -> InferenceAPI:
    """Get or create the global InferenceAPI instance."""
    global _inference_api
    if _inference_api is None:
        _inference_api = InferenceAPI()
    return _inference_api

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
        self.name = config.name
        self.config = config
        self.api = get_inference_api()  # Use shared global instance
        self.prompt, self.system_prompt = parse_autograder_prompt(self.config.template_file)
        
        # Create semaphore for this grader if max_concurrent is specified
        self.semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None

    async def grade(self, response: str, eval: ScenarioEval) -> Optional[ResponseGrading]:
        # Acquire semaphore if configured
        if self.semaphore:
            async with self.semaphore:
                return await self._do_grade(response, eval)
        else:
            return await self._do_grade(response, eval)
    
    async def _do_grade(self, response: str, eval: ScenarioEval) -> Optional[ResponseGrading]:
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


class StringMatchGrader(ResponseGrader):
    """
    Grade a model response based on string matching.
    """
    
    def __init__(self, config: StringMatchGraderConfig):
        self.name = config.name
        self.config = config
        
        # Precompile regex patterns for contains matching if needed
        if config.match_type in ["contains", "contains_word"] and not config.case_sensitive:
            if config.match_type == "contains_word":
                # Use word boundaries for word matching
                self.patterns = [re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE) for s in config.expected_strings]
            else:
                self.patterns = [re.compile(re.escape(s), re.IGNORECASE) for s in config.expected_strings]
        elif config.match_type == "contains_word" and config.case_sensitive:
            # Case-sensitive word boundary patterns
            self.patterns = [re.compile(r'\b' + re.escape(s) + r'\b') for s in config.expected_strings]
        else:
            self.patterns = None
    
    async def grade(self, response: str, eval: ScenarioEval) -> Optional[ResponseGrading]:
        """Grade response based on string matching criteria."""
        matches = []
        
        for expected in self.config.expected_strings:
            if self.config.match_type == "exact":
                if self.config.case_sensitive:
                    matched = response == expected
                else:
                    matched = response.lower() == expected.lower()
            
            elif self.config.match_type == "contains":
                if self.config.case_sensitive:
                    matched = expected in response
                else:
                    # Use precompiled patterns for case-insensitive contains
                    idx = self.config.expected_strings.index(expected)
                    matched = bool(self.patterns[idx].search(response))
            
            elif self.config.match_type == "contains_word":
                # Use word boundary matching to match whole words only
                idx = self.config.expected_strings.index(expected)
                matched = bool(self.patterns[idx].search(response))
            
            elif self.config.match_type == "fuzzy":
                # Use SequenceMatcher for fuzzy matching
                if not self.config.case_sensitive:
                    response_cmp = response.lower()
                    expected_cmp = expected.lower()
                else:
                    response_cmp = response
                    expected_cmp = expected
                
                similarity = SequenceMatcher(None, response_cmp, expected_cmp).ratio()
                matched = similarity >= self.config.fuzzy_threshold
            
            else:
                raise ValueError(f"Unknown match_type: {self.config.match_type}")
            
            matches.append(matched)
        
        # Determine final match based on match_all setting
        if self.config.match_all:
            final_match = all(matches)
        else:
            final_match = any(matches)
        
        # Assign score based on match result
        score = self.config.score_on_match if final_match else self.config.score_on_no_match
        
        # Include match details in data
        data = {
            "matched": final_match,
            "match_details": [
                {"expected": exp, "matched": m} 
                for exp, m in zip(self.config.expected_strings, matches)
            ],
            "match_type": self.config.match_type,
            "match_all": self.config.match_all
        }
        
        return ResponseGrading(score=score, data=data)


class BinaryChoiceGrader(ResponseGrader):
    """
    Grade a model response based on binary choice detection.
    Returns None if both positive and negative strings are detected, or if neither are detected.
    """
    
    def __init__(self, config: BinaryChoiceGraderConfig):
        self.name = config.name
        self.config = config
        
        # Precompile regex patterns if needed
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        if self.config.match_type in ["contains", "contains_word"] and not self.config.case_sensitive:
            if self.config.match_type == "contains_word":
                # Use word boundaries for word matching
                self.positive_patterns = [re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE) for s in self.config.positive_strings]
                self.negative_patterns = [re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE) for s in self.config.negative_strings]
            else:
                self.positive_patterns = [re.compile(re.escape(s), re.IGNORECASE) for s in self.config.positive_strings]
                self.negative_patterns = [re.compile(re.escape(s), re.IGNORECASE) for s in self.config.negative_strings]
        elif self.config.match_type == "contains_word" and self.config.case_sensitive:
            # Case-sensitive word boundary patterns
            self.positive_patterns = [re.compile(r'\b' + re.escape(s) + r'\b') for s in self.config.positive_strings]
            self.negative_patterns = [re.compile(r'\b' + re.escape(s) + r'\b') for s in self.config.negative_strings]
        else:
            self.positive_patterns = None
            self.negative_patterns = None
    
    def _check_match(self, response: str, expected: str, patterns: Optional[List] = None, idx: int = 0) -> bool:
        """Check if expected string matches in response according to match_type."""
        if self.config.match_type == "exact":
            if self.config.case_sensitive:
                return response == expected
            else:
                return response.lower() == expected.lower()
        
        elif self.config.match_type == "contains":
            if self.config.case_sensitive:
                return expected in response
            else:
                # Use precompiled patterns for case-insensitive contains
                return bool(patterns[idx].search(response))
        
        elif self.config.match_type == "contains_word":
            # Use word boundary matching to match whole words only
            return bool(patterns[idx].search(response))
        
        elif self.config.match_type == "fuzzy":
            # Use SequenceMatcher for fuzzy matching
            if not self.config.case_sensitive:
                response_cmp = response.lower()
                expected_cmp = expected.lower()
            else:
                response_cmp = response
                expected_cmp = expected
            
            similarity = SequenceMatcher(None, response_cmp, expected_cmp).ratio()
            return similarity >= self.config.fuzzy_threshold
        
        else:
            raise ValueError(f"Unknown match_type: {self.config.match_type}")
    
    async def grade(self, response: str, eval: ScenarioEval) -> Optional[ResponseGrading]:
        """Grade response based on binary choice criteria."""
        # Check for positive matches
        positive_matches = []
        for i, positive_str in enumerate(self.config.positive_strings):
            matched = self._check_match(response, positive_str, self.positive_patterns, i)
            positive_matches.append(matched)
        
        # Check for negative matches
        negative_matches = []
        for i, negative_str in enumerate(self.config.negative_strings):
            matched = self._check_match(response, negative_str, self.negative_patterns, i)
            negative_matches.append(matched)
        
        # Determine if we found any positive or negative matches
        has_positive = any(positive_matches)
        has_negative = any(negative_matches)
        
        # Determine score based on matches
        if has_positive and has_negative:
            # Both detected
            if self.config.score_both is None:
                return None  # Skip grading
            score = self.config.score_both
            choice = "both"
        elif has_positive and not has_negative:
            # Only positive detected
            score = self.config.score_positive
            choice = "positive"
        elif has_negative and not has_positive:
            # Only negative detected
            score = self.config.score_negative
            choice = "negative"
        else:
            # Neither detected
            if self.config.score_neither is None:
                return None  # Skip grading
            score = self.config.score_neither
            choice = "neither"
        
        # Include match details in data
        data = {
            "choice": choice,
            "positive_matches": [
                {"string": s, "matched": m} 
                for s, m in zip(self.config.positive_strings, positive_matches)
            ],
            "negative_matches": [
                {"string": s, "matched": m} 
                for s, m in zip(self.config.negative_strings, negative_matches)
            ],
            "match_type": self.config.match_type
        }
        
        return ResponseGrading(score=score, data=data)