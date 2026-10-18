"""
Unit tests for Actor-Critic Explanation Refinement Feedback Loop.
Tests convergence, trajectory tracking, and mock refinement steps.
"""

import unittest

from src.baselines.llm_judge import LLMJudge
from src.baselines.schemas import Verdict
from src.refinement.generator import ExplanationGenerator
from src.refinement.loop import RefinementLoop
from src.refinement.schemas import RefinementResult


class TestRefinementLoop(unittest.TestCase):

    def setUp(self):
        # Use mock mode for fast, deterministic unit test execution
        self.judge = LLMJudge(mock=True)
        self.generator = ExplanationGenerator(mock=True)
        self.loop = RefinementLoop(judge=self.judge, generator=self.generator, max_iterations=3, mock=True)

    def test_already_correct_explanation_converges_immediately(self):
        code = """def sum_evens(nums: list[int]) -> int:
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += n
    return total
"""
        initial_expl = "Computes sum of even numbers. Returns 0 on empty list."

        res: RefinementResult = self.loop.run(code, initial_expl)

        self.assertTrue(res.converged)
        self.assertEqual(res.total_iterations, 1)
        self.assertEqual(res.final_verdict, Verdict.CORRECT)
        self.assertEqual(len(res.steps), 1)
        self.assertAlmostEqual(res.final_soundness, 1.0, places=2)

    def test_flawed_explanation_refines_and_converges(self):
        code = """def compute_average(nums: list[float]) -> float:
    return sum(nums) / len(nums)
"""
        flawed_expl = "This function calculates arithmetic mean. If empty, it returns 0.0 safely."

        res: RefinementResult = self.loop.run(code, flawed_expl)

        # Iteration 1: INCORRECT (ZeroDivisionError on empty)
        # Iteration 2: Refined explanation explicitly addresses ZeroDivisionError -> CORRECT
        self.assertTrue(res.converged)
        self.assertEqual(res.total_iterations, 2)
        self.assertEqual(res.final_verdict, Verdict.CORRECT)
        self.assertEqual(len(res.steps), 2)
        self.assertLess(res.initial_soundness, 0.5)
        self.assertGreater(res.final_soundness, 0.9)
        self.assertIn("ZeroDivisionError", res.final_explanation)

    def test_max_iterations_cap_respected(self):
        loop_capped = RefinementLoop(judge=self.judge, generator=self.generator, max_iterations=1, mock=True)
        code = """def compute_average(nums: list[float]) -> float:
    return sum(nums) / len(nums)
"""
        flawed_expl = "This function calculates arithmetic mean. If empty, it returns 0.0 safely."

        res = loop_capped.run(code, flawed_expl)

        self.assertFalse(res.converged)
        self.assertEqual(res.total_iterations, 1)
        self.assertEqual(res.final_verdict, Verdict.INCORRECT)

    def test_synthesizes_initial_explanation_when_omitted(self):
        code = """def compute_average(nums: list[float]) -> float:
    return sum(nums) / len(nums)
"""
        res = self.loop.run(code, initial_explanation=None)

        self.assertIsNotNone(res.initial_explanation)
        self.assertGreater(len(res.initial_explanation), 10)
        self.assertTrue(res.converged)


if __name__ == "__main__":
    unittest.main()

