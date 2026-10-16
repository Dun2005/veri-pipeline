"""
Schemas for LLM-as-a-Judge Evaluation.
Based on G-Eval (Liu et al., TACL 2023) and MT-Bench (Zheng et al., NeurIPS 2023).
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    CORRECT = "CORRECT"              # Sound: Every claim is mathematically true for P
    INCORRECT = "INCORRECT"          # Unsound: Contradicts actual code behavior or hallucinated logic
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"  # Mostly true but misses critical edge conditions or has minor imprecision


class ClaimAnalysis(BaseModel):
    claim: str = Field(
        ...,
        description="A distinct atomic claim made by the explanation about the program's behavior or invariants."
    )
    is_sound: bool = Field(
        ...,
        description="True if the claim holds strictly under the operational semantics of the program."
    )
    rationale: str = Field(
        ...,
        description="Mathematical or semantic justification for why the claim is sound or unsound."
    )


class Counterexample(BaseModel):
    found: bool = Field(
        ...,
        description="True if a concrete input falsifies any claim in the explanation."
    )
    input_value: Optional[str] = Field(
        default=None,
        description="Concrete input value (e.g. x = -1, nums = []) that exposes the flaw in the explanation."
    )
    actual_output: Optional[str] = Field(
        default=None,
        description="The actual execution output or behavior produced by the program on this input."
    )
    claimed_output: Optional[str] = Field(
        default=None,
        description="The output or behavior implied by the explanation on this input."
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Detailed description of why this input constitutes a counterexample."
    )


class JudgeResult(BaseModel):
    """
    Standard evaluation output schema for LLM-as-a-Judge baseline.
    Conforms to G-Eval rubric structure with quantitative scores and qualitative verification.
    """
    verdict: Verdict = Field(
        ...,
        description="Final correctness verdict: CORRECT, INCORRECT, or PARTIALLY_CORRECT."
    )
    soundness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 and 1.0 indicating degree of mathematical soundness (freedom from false claims)."
    )
    completeness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score between 0.0 and 1.0 indicating how comprehensively the explanation covers edge cases and main logic."
    )
    chain_of_thought: str = Field(
        ...,
        description="Step-by-step semantic analysis tracing program execution and comparing with claims (Wei et al. 2022)."
    )
    claims: List[ClaimAnalysis] = Field(
        default_factory=list,
        description="Decomposed atomic claims extracted from the explanation and their individual verifications."
    )
    counterexample: Optional[Counterexample] = Field(
        default=None,
        description="A concrete counterexample if an unsound claim was refuted."
    )
    summary: str = Field(
        ...,
        description="A concise executive summary of the evaluation finding."
    )
    is_execution_verified: Optional[bool] = Field(
        default=None,
        description="True if the counterexample was confirmed by running the target code in Python sandbox."
    )
    runtime_output: Optional[str] = Field(
        default=None,
        description="The actual runtime output or exception caught when executing the counterexample."
    )
    audit_correction_applied: bool = Field(
        default=False,
        description="True if the self-consistency auditor corrected a label inversion or verbosity bias bug."
    )
    audit_note: Optional[str] = Field(
        default=None,
        description="Notes from the self-consistency audit and runtime sandbox verification."
    )

