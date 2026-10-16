"""Baselines package for program explanation verification."""
from src.baselines.schemas import JudgeResult, Verdict, ClaimAnalysis, Counterexample
from src.baselines.llm_judge import LLMJudge
from src.baselines.prompts import LLM_JUDGE_SYSTEM_PROMPT, LLM_JUDGE_USER_PROMPT_TEMPLATE

__all__ = [
    "LLMJudge",
    "JudgeResult",
    "Verdict",
    "ClaimAnalysis",
    "Counterexample",
    "LLM_JUDGE_SYSTEM_PROMPT",
    "LLM_JUDGE_USER_PROMPT_TEMPLATE",
]

