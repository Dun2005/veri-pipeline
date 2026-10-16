"""
Formatting utility for JudgeResult output.
"""

from typing import Optional
from src.baselines.schemas import JudgeResult, Verdict


def print_result(result: JudgeResult, expected: Optional[Verdict] = None) -> None:
    """Pretty prints JudgeResult to console."""
    verdict_color = {
        Verdict.CORRECT: "\033[92m",          # Green
        Verdict.INCORRECT: "\033[91m",        # Red
        Verdict.PARTIALLY_CORRECT: "\033[93m",# Yellow
    }
    reset = "\033[0m"
    bold = "\033[1m"

    v_str = result.verdict.value
    color = verdict_color.get(result.verdict, "")

    match_str = ""
    if expected is not None:
        if result.verdict == expected:
            match_str = f" [MATCHES EXPECTED: {expected.value}]"
        else:
            match_str = f" [DISAGREES WITH EXPECTED: {expected.value}]"

    print(f"\n{bold}EVALUATION VERDICT:{reset} {color}{bold}{v_str}{reset}{match_str}")
    print(f"Soundness Score:    {result.soundness_score:.2f} / 1.00")
    print(f"Completeness Score: {result.completeness_score:.2f} / 1.00")
    print(f"Summary:            {result.summary}")

    print(f"\n{bold}Reasoning (Chain of Thought):{reset}")
    print(result.chain_of_thought)

    if result.claims:
        print(f"\n{bold}Claim-by-Claim Verification ({len(result.claims)} claims):{reset}")
        for i, c in enumerate(result.claims, 1):
            mark = "\033[92m[VALID]\033[0m" if c.is_sound else "\033[91m[INVALID]\033[0m"
            print(f"  {i}. {mark} {c.claim}")
            print(f"     Rationale: {c.rationale}")

    if result.counterexample and result.counterexample.found:
        ce = result.counterexample
        print(f"\n{bold}\033[91mRefuting Counterexample Found:\033[0m{reset}")
        print(f"  Input (x):       {ce.input_value}")
        print(f"  Actual P(x):     {ce.actual_output}")
        print(f"  Claimed E(x):    {ce.claimed_output}")
        if ce.explanation:
            print(f"  Explanation:     {ce.explanation}")

    if result.is_execution_verified:
        print(f"\n{bold}\033[92m[Sandbox Grounding]: Confirmed by Python runtime execution!\033[0m{reset}")
        print(f"  Runtime Output:  {result.runtime_output}")

    if result.audit_correction_applied:
        print(f"\n{bold}\033[93m[Self-Consistency Auditor]: Correction Applied\033[0m{reset}")
        print(f"  Details:         {result.audit_note}")

    print("-" * 50)

