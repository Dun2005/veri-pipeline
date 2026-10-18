"""
Formatting utility for JudgeResult and RefinementResult output.
"""

from typing import Optional
from src.baselines.schemas import JudgeResult, Verdict
from src.refinement.schemas import RefinementResult


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


def print_refinement_result(result: RefinementResult) -> None:
    """Pretty prints the full Actor-Critic refinement trajectory."""
    reset = "\033[0m"
    bold = "\033[1m"
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    cyan = "\033[96m"

    print("\n" + "=" * 75)
    print(f"{bold}{cyan}ACTOR-CRITIC REFINEMENT LOOP TRAJECTORY{reset}")
    print("=" * 75)
    print(f"{bold}Target Program:{reset}\n{result.code}")
    print("-" * 75)
    print(f"{bold}Initial Explanation (E_0):{reset}\n{result.initial_explanation}")

    for step in result.steps:
        v_color = green if step.judge_result.verdict == Verdict.CORRECT else red
        print("\n" + "-" * 75)
        print(
            f"{bold}Iteration {step.iteration}: {v_color}{step.judge_result.verdict.value}{reset} "
            f"(Soundness: {step.judge_result.soundness_score:.2f}, Completeness: {step.judge_result.completeness_score:.2f})"
        )
        print(f"Explanation E_{step.iteration - 1}:\n\"{step.explanation}\"")
        print(f"Judge Critique: {step.judge_result.summary}")

        if step.judge_result.counterexample and step.judge_result.counterexample.found:
            ce = step.judge_result.counterexample
            print(f"  {red}[!] Counterexample:{reset} {ce.input_value} -> P(x) = {ce.actual_output} (vs Claimed: {ce.claimed_output})")

    print("\n" + "=" * 75)
    status_str = f"{green}CONVERGED IN {result.total_iterations} ROUNDS{reset}" if result.converged else f"{yellow}STOPPED AT MAX ROUNDS{reset}"
    print(f"{bold}FINAL STATUS:{reset} {status_str}")
    print(f"Soundness Trajectory: {result.initial_soundness:.2f} -> {result.final_soundness:.2f}")
    print(f"Evolution Summary:    {result.evolution_summary}")
    print(f"\n{bold}{green}Verified Sound Explanation (E*):{reset}")
    print(f"\"\"\"\n{result.final_explanation}\n\"\"\"")
    print("=" * 75)
