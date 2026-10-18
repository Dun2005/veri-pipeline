"""
Refinement Loop Orchestrator (Actor-Critic Feedback Controller).
Implements the interactive loop between Generator (Actor) and Verifier (LLMJudge / Critic).
"""

import logging
from typing import Optional, List

from src.baselines.llm_judge import LLMJudge
from src.baselines.schemas import Verdict
from src.refinement.generator import ExplanationGenerator
from src.refinement.schemas import RefinementResult, RefinementStep

logger = logging.getLogger(__name__)


class RefinementLoop:
    """
    Orchestrates the Actor-Critic feedback loop:
    P -> [Generator] -> E_t -> [LLMJudge] -> Feedback -> [Generator] -> E_{t+1}
    Continues until judge verifies correctness or max_iterations is reached.
    """

    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        generator: Optional[ExplanationGenerator] = None,
        max_iterations: int = 3,
        mock: bool = False,
    ):
        """
        :param judge: An instance of LLMJudge acting as the Critic.
        :param generator: An instance of ExplanationGenerator acting as the Actor.
        :param max_iterations: Maximum refinement rounds before halting (default: 3).
        :param mock: If True, uses offline mock logic for both judge and generator.
        """
        self.mock = mock
        self.max_iterations = max_iterations
        self.judge = judge or LLMJudge(mock=mock)
        self.generator = generator or ExplanationGenerator(mock=mock)

    def run(self, code: str, initial_explanation: Optional[str] = None) -> RefinementResult:
        """
        Execute the iterative refinement loop.

        :param code: Source code of program P.
        :param initial_explanation: Optional initial candidate explanation E_0.
                                     If omitted, generator synthesizes an initial draft.
        :return: RefinementResult containing final explanation and trajectory trace.
        """
        code = code.strip()

        # Step 0: Ensure starting explanation exists
        if not initial_explanation or not initial_explanation.strip():
            print(f"==> [1/3 Actor] Synthesizing initial candidate explanation E_0 with LLM ({self.generator.model})...")
            current_explanation = self.generator.generate_initial(code)
            print(f"    Initial Explanation E_0 generated ({len(current_explanation.split())} words).")
        else:
            current_explanation = initial_explanation.strip()

        starting_explanation = current_explanation
        steps: List[RefinementStep] = []
        converged = False

        print(f"\n==> [2/3 Starting Feedback Loop] Max iterations: {self.max_iterations}")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            print(f"==> [Critic] Evaluating explanation with LLM Judge ({self.judge.model})...")
            
            # Critic evaluates current explanation
            judge_res = self.judge.evaluate(code, current_explanation)
            print(f"    Verdict: {judge_res.verdict.value} | Soundness: {judge_res.soundness_score:.2f} | Completeness: {judge_res.completeness_score:.2f}")
            
            step = RefinementStep(
                iteration=iteration,
                explanation=current_explanation,
                judge_result=judge_res,
            )
            steps.append(step)

            # Check convergence: CORRECT verdict and no refuting counterexample
            has_no_counterexample = (
                not judge_res.counterexample or not judge_res.counterexample.found
            )
            is_correct = (judge_res.verdict == Verdict.CORRECT) and has_no_counterexample

            if is_correct:
                print(f"==> [Convergence] Formal correctness verified at iteration {iteration}!")
                converged = True
                break

            # If not converged and iterations remain, prompt generator to refine
            if iteration < self.max_iterations:
                if judge_res.counterexample and judge_res.counterexample.found:
                    print(f"    [!] Counterexample found: {judge_res.counterexample.input_value} -> P(x) = {judge_res.counterexample.actual_output}")
                print(f"==> [Actor] Refining explanation E_{iteration} with LLM based on Critic feedback...")
                current_explanation = self.generator.refine(code, current_explanation, judge_res)

        # Build trajectory summary
        initial_soundness = steps[0].judge_result.soundness_score
        final_soundness = steps[-1].judge_result.soundness_score
        final_verdict = steps[-1].judge_result.verdict

        if converged:
            evolution_summary = (
                f"Successfully converged in {len(steps)} iteration(s). "
                f"Soundness improved from {initial_soundness:.2f} to {final_soundness:.2f} ({final_verdict.value})."
            )
        else:
            evolution_summary = (
                f"Reached max iterations ({self.max_iterations}) without full convergence. "
                f"Soundness evolved from {initial_soundness:.2f} to {final_soundness:.2f} ({final_verdict.value})."
            )

        return RefinementResult(
            code=code,
            initial_explanation=starting_explanation,
            final_explanation=steps[-1].explanation,
            converged=converged,
            total_iterations=len(steps),
            final_verdict=final_verdict,
            initial_soundness=initial_soundness,
            final_soundness=final_soundness,
            steps=steps,
            evolution_summary=evolution_summary,
        )

