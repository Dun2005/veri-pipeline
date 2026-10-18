"""
Schemas for Explanation Refinement Loop (Actor-Critic Feedback Architecture).
Based on Self-Refine (Madaan et al., NeurIPS 2023) and Reflexion (Shinn et al., NeurIPS 2023).
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.baselines.schemas import JudgeResult, Verdict


class RefinementStep(BaseModel):
    """
    A single iteration step within the Actor-Critic feedback loop.
    """
    iteration: int = Field(..., description="1-indexed iteration number.")
    explanation: str = Field(..., description="The explanation evaluated in this iteration.")
    judge_result: JudgeResult = Field(..., description="The judge's comprehensive evaluation result.")


class RefinementResult(BaseModel):
    """
    Final outcome of the iterative refinement process.
    """
    code: str = Field(..., description="The target Python program.")
    initial_explanation: str = Field(..., description="Starting explanation (E_0).")
    final_explanation: str = Field(..., description="Final revised explanation (E*).")
    converged: bool = Field(..., description="True if the judge verified the explanation as CORRECT.")
    total_iterations: int = Field(..., description="Total number of refinement iterations executed.")
    final_verdict: Verdict = Field(..., description="Final verdict returned by the judge.")
    initial_soundness: float = Field(..., description="Soundness score at iteration 1.")
    final_soundness: float = Field(..., description="Soundness score at final iteration.")
    steps: List[RefinementStep] = Field(default_factory=list, description="Trajectory of all iteration steps.")
    evolution_summary: str = Field(..., description="Summary of how the explanation improved across iterations.")

