"""
Explanation Generator and Refiner (The Actor in Actor-Critic Architecture).
Generates initial candidate explanations and iteratively refines them using feedback from LLMJudge.
"""

import logging
import time
from typing import Optional

from src.baselines.schemas import JudgeResult
from src.refinement.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    INITIAL_GENERATION_USER_PROMPT,
    FORMAL_GENERATION_USER_PROMPT,
    REFINEMENT_USER_PROMPT,
)
from src.utils.config import get_gemini_api_key, get_openai_api_key

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generator (Actor) capable of synthesizing initial code explanations and
    revising them given counterexamples and critique from LLMJudge.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        provider: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        mock: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.mock = mock

        if provider is None:
            if "gemini" in model.lower():
                self.provider = "gemini"
            elif "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
                self.provider = "openai"
            else:
                self.provider = "gemini"
        else:
            self.provider = provider.lower()

        self.api_key = api_key
        if not self.mock:
            self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            from google import genai
            key = self.api_key or get_gemini_api_key()
            self.client = genai.Client(api_key=key) if key else None
        elif self.provider == "openai":
            from openai import OpenAI
            key = self.api_key or get_openai_api_key()
            self.client = OpenAI(api_key=key) if key else None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_initial(self, code: str, formal: bool = False) -> str:
        """
        Generate an initial candidate explanation E_0 for program P using the LLM.
        :param code: Source code of program P.
        :param formal: If True, instructs LLM with rigorous formal verification requirements.
                       If False, asks for a standard natural explanation (reflecting typical LLM outputs).
        """
        if self.mock:
            return self._mock_initial(code)

        template = FORMAL_GENERATION_USER_PROMPT if formal else INITIAL_GENERATION_USER_PROMPT
        prompt = template.format(code=code.strip())
        return self._generate(prompt)

    def refine(self, code: str, previous_explanation: str, judge_result: JudgeResult) -> str:
        """
        Refine an existing explanation E_t into E_{t+1} using the Critic's feedback.
        """
        if self.mock:
            return self._mock_refine(code, previous_explanation, judge_result)

        # Build structured feedback sections
        unsound_claims = [c for c in judge_result.claims if not c.is_sound]
        if unsound_claims:
            claims_text = "SPECIFIC UNSOUND CLAIMS IDENTIFIED:\n" + "\n".join(
                f"- Claim: \"{c.claim}\"\n  Why it is wrong: {c.rationale}"
                for c in unsound_claims
            )
        else:
            claims_text = "No individual claims flagged as strictly unsound, but overall score is suboptimal."

        if judge_result.counterexample and judge_result.counterexample.found:
            ce = judge_result.counterexample
            ce_text = (
                f"CONCRETE REFUTING COUNTEREXAMPLE:\n"
                f"- Input: {ce.input_value}\n"
                f"- Actual Program Output: {ce.actual_output}\n"
                f"- Output Implied by Your Explanation: {ce.claimed_output}\n"
                f"- Flaw Detail: {ce.explanation}"
            )
        else:
            ce_text = "No concrete counterexample was produced, but overall correctness requirements were violated."

        prompt = REFINEMENT_USER_PROMPT.format(
            code=code.strip(),
            previous_explanation=previous_explanation.strip(),
            verdict=judge_result.verdict.value,
            soundness_score=judge_result.soundness_score,
            completeness_score=judge_result.completeness_score,
            summary=judge_result.summary,
            unsound_claims_section=claims_text,
            counterexample_section=ce_text,
        )

        return self._generate(prompt)

    def _generate(self, user_prompt: str) -> str:
        """Execute model call with automatic retry."""
        if self.client is None:
            raise RuntimeError(
                f"Generator client for '{self.provider}' is not initialized. Check your API key or pass mock=True."
            )

        if self.provider == "gemini":
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
                temperature=self.temperature,
            )
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=config,
                    )
                    return response.text.strip()
                except Exception as e:
                    logger.warning(f"Generator Gemini API attempt {attempt + 1} failed ({e}).")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        raise

        elif self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content.strip()

        raise ValueError(f"Unknown provider: {self.provider}")

    def _mock_initial(self, code: str) -> str:
        lower = code.lower()
        if "compute_average" in lower:
            return "This function calculates the average of a list of numbers. If the list is empty, it returns 0.0 without errors."
        elif "search" in lower:
            return "This function performs binary search. If the target is not in the list, it raises a ValueError."
        return "This function executes the program logic and returns the computed result."

    def _mock_refine(self, code: str, previous_explanation: str, judge_result: JudgeResult) -> str:
        lower_c = code.lower()
        if "compute_average" in lower_c:
            return (
                "This function calculates the arithmetic mean of a list of numbers by dividing sum(nums) by len(nums). "
                "Notice that on an empty list (nums = []), evaluating len(nums) produces 0, which raises an unhandled ZeroDivisionError."
            )
        elif "search" in lower_c:
            return (
                "This function performs binary search on a sorted list. If the target element is present, "
                "it returns its index. If the target element is absent or not found, it returns -1."
            )
        elif "verify_gauss_sum" in lower_c:
            return (
                "This function simulates 32-bit unsigned arithmetic to check Gauss's triangular summation formula. "
                "While the identity holds for values below 65536, at n = 65536 the intermediate product n*(n+1) overflows "
                "the 32-bit boundary, causing the closed-form calculation to diverge from the accumulated sum and returning False."
            )
        return (
            previous_explanation + " (Revised: Corrected edge case handling based on verifier feedback.)"
        )

