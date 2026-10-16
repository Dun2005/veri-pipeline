"""
Unit tests for LLM-as-a-Judge Baseline.
Tests schema validation, JSON deserialization, and mock evaluation.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.baselines.schemas import JudgeResult, Verdict, ClaimAnalysis, Counterexample
from src.baselines.llm_judge import LLMJudge


class TestLLMJudgeSchemas(unittest.TestCase):
    def test_schema_valid_correct(self):
        result = JudgeResult(
            verdict=Verdict.CORRECT,
            soundness_score=1.0,
            completeness_score=0.95,
            chain_of_thought="Step 1: Examined code...",
            claims=[
                ClaimAnalysis(
                    claim="Returns sum of evens",
                    is_sound=True,
                    rationale="Loop adds numbers where n % 2 == 0",
                )
            ],
            counterexample=Counterexample(found=False),
            summary="Explanation accurately describes program semantics.",
        )
        self.assertEqual(result.verdict, Verdict.CORRECT)
        self.assertEqual(len(result.claims), 1)
        self.assertFalse(result.counterexample.found)

    def test_schema_valid_incorrect_with_counterexample(self):
        result = JudgeResult(
            verdict=Verdict.INCORRECT,
            soundness_score=0.2,
            completeness_score=0.5,
            chain_of_thought="Step 1: Found empty list causes ZeroDivisionError.",
            claims=[
                ClaimAnalysis(
                    claim="Safely handles empty list returning 0.0",
                    is_sound=False,
                    rationale="Raises ZeroDivisionError when len(nums) == 0",
                )
            ],
            counterexample=Counterexample(
                found=True,
                input_value="nums = []",
                actual_output="ZeroDivisionError",
                claimed_output="0.0",
                explanation="The code does not check for empty list before division.",
            ),
            summary="Explanation falsely claims graceful empty list handling.",
        )
        self.assertEqual(result.verdict, Verdict.INCORRECT)
        self.assertTrue(result.counterexample.found)
        self.assertEqual(result.counterexample.input_value, "nums = []")

    def test_serialization_deserialization(self):
        data = {
            "verdict": "INCORRECT",
            "soundness_score": 0.0,
            "completeness_score": 0.5,
            "chain_of_thought": "Tracing execution...",
            "claims": [],
            "counterexample": {"found": False},
            "summary": "Hallucinated behavior.",
        }
        res = JudgeResult.model_validate(data)
        self.assertEqual(res.verdict, Verdict.INCORRECT)
        json_str = res.model_dump_json()
        res_back = JudgeResult.model_validate_json(json_str)
        self.assertEqual(res_back.verdict, Verdict.INCORRECT)


class TestMockJudgeEvaluation(unittest.TestCase):
    @patch("src.baselines.llm_judge.get_gemini_api_key", return_value="dummy_key")
    @patch("google.genai.Client")
    def test_evaluate_gemini_mock(self, mock_client_cls, mock_get_key):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = """{
            "verdict": "CORRECT",
            "soundness_score": 1.0,
            "completeness_score": 1.0,
            "chain_of_thought": "Code computes evens accurately.",
            "claims": [
                {"claim": "Sums even numbers", "is_sound": true, "rationale": "Matches loop condition"}
            ],
            "counterexample": {"found": false},
            "summary": "Valid."
        }"""
        mock_client.models.generate_content.return_value = mock_response

        judge = LLMJudge(model="gemini-2.5-flash", provider="gemini")
        result = judge.evaluate(
            code="def f(x): return x * 2",
            explanation="Multiplies input by 2.",
        )
        self.assertEqual(result.verdict, Verdict.CORRECT)
        self.assertEqual(result.soundness_score, 1.0)


if __name__ == "__main__":
    unittest.main()

