"""
benchmark_comparison.py - Comprehensive Head-to-Head Comparison:
1. Formal SMT / AST-Anchored Logic Verifier (Formal-Program-Verification-via-LLM-Chain-of-Thought-and-SMT-Logic)
   vs
2. SOTA LLM Judge + Closed-Loop Actor-Critic Pipeline (veri-pipeline)

Runs both pipelines on identical programs from the SV-COMP dataset (E:\\BTL\\Dataset\\sv-benchmarks-loops-py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add AST-verifier project path
ASTVERIFIER_PATH = Path(r"E:\BTL\Formal-Program-Verification-via-LLM-Chain-of-Thought-and-SMT-Logic\src")
if str(ASTVERIFIER_PATH) not in sys.path:
    sys.path.insert(0, str(ASTVERIFIER_PATH))

from astverifier.pipeline import run_pipeline as run_smt_pipeline

# Add veri-pipeline path
VERI_ROOT = Path(__file__).resolve().parent
if str(VERI_ROOT) not in sys.path:
    sys.path.insert(0, str(VERI_ROOT))

from src.baselines.llm_judge import LLMJudge
from src.refinement.generator import ExplanationGenerator
from src.refinement.loop import RefinementLoop
from src.baselines.schemas import Verdict

DATASET_ROOT = Path(r"E:\BTL\Dataset\sv-benchmarks-loops-py")

# Benchmark Test Suite: Curated from SV-COMP to evaluate expressivity, soundness, and refinement
BENCHMARK_CASES = [
    {
        "id": "SV-COMP sumt2 (Parity Loop Input Violation)",
        "file": DATASET_ROOT / "loops-crafted-1" / "sumt2.py",
        "code": None,  # loaded from file
        "initial_explanation": (
            "The function iterates l from 0 up to n. In each iteration, if l is even, i is incremented; "
            "otherwise j is incremented. Total increments to i and j equal the number of iterations l. "
            "Therefore, (i + j) == l is an invariant and the assertion holds unconditionally."
        ),
        "ground_truth_code": "COUNTEREXAMPLE",  # i and j are inputs! If i=1, j=0, l=0, (i+j) != l
        "flaw_type": "Unsound assumption about uninitialized input parameters"
    },
    {
        "id": "SV-COMP eq2 (Dead Loop / Vacuous Truth)",
        "file": DATASET_ROOT / "loop-invariants" / "eq2.py",
        "code": None,
        "initial_explanation": (
            "The function enters a while loop that increments y and z synchronously. "
            "Because both variables increase at the same rate inside the loop, the property y == z "
            "holds as a post-condition invariant when the loop terminates."
        ),
        "ground_truth_code": "COUNTEREXAMPLE",  # while 0 never executes! y and z are arbitrary inputs
        "flaw_type": "Vacuous truth fallacy: loop condition is while 0, so loop never executes"
    },
    {
        "id": "SV-COMP loopv1 (Modulo Step Invariant)",
        "file": DATASET_ROOT / "loops-crafted-1" / "loopv1.py",
        "code": None,
        "initial_explanation": (
            "In this loop, i is incremented by 3 at each execution step. "
            "Because increments are strictly multiples of 3, (i % 3) == 0 is guaranteed to hold "
            "upon loop termination for any input."
        ),
        "ground_truth_code": "COUNTEREXAMPLE",  # If initial input i is not divisible by 3 (e.g. i = 1), (i % 3) != 0
        "flaw_type": "Unsound assumption on initial state (ignores non-zero initial modulo)"
    },
    {
        "id": "SV-COMP sum02-1 (Gauss 32-bit Integer Overflow)",
        "file": None,
        "code": """def verify_gauss_sum(n: int) -> bool:
    sn = 0
    for i in range(n + 1):
        sn = (sn + i) & 0xFFFFFFFF
    rhs = (((n * (n + 1)) & 0xFFFFFFFF) // 2)
    assert sn == rhs or sn == 0
    return sn == rhs or sn == 0
""",
        "initial_explanation": (
            "This function computes the sum of the first n integers in 32-bit unsigned arithmetic. "
            "By Gauss's identity, the sum of 0..n is mathematically identical to n*(n+1)//2. "
            "Since sn and rhs implement the same arithmetic progression, sn == rhs holds for all n >= 0."
        ),
        "ground_truth_code": "COUNTEREXAMPLE",  # n >= 65536 truncates before division by 2!
        "flaw_type": "Bitwise masking before integer division truncates overflow bits"
    }
]


def run_benchmark():
    print("=" * 80)
    print("HEAD-TO-HEAD BENCHMARK: FORMAL SMT (Z3) vs LLM JUDGE + ACTOR-CRITIC")
    print("=" * 80)
    print(f"Dataset root: {DATASET_ROOT}")

    # Initialize veri-pipeline agents
    judge = LLMJudge(model="gemini-3.8-flash", mock=False)
    generator = ExplanationGenerator(model="gemini-3.5-flash-lite", mock=False)
    refinement_loop = RefinementLoop(judge=judge, generator=generator, max_iterations=3, mock=False)

    summary_rows = []

    for idx, case in enumerate(BENCHMARK_CASES, 1):
        print(f"\n[{idx}/{len(BENCHMARK_CASES)}] Testing: {case['id']}")
        print("-" * 80)

        # 1. Load code
        if case["file"] and case["file"].exists():
            code = case["file"].read_text(encoding="utf-8")
        else:
            code = case["code"]

        # --- A. RUN FORMAL SMT PIPELINE ---
        print(">>> [Pipeline 1: Formal SMT / AST-Anchored Verifier]...")
        t0 = time.perf_counter()
        smt_res = run_smt_pipeline(
            code,
            baseline="AST_ANCHORED",
            program_id=case["id"],
            enable_feedback=False,
            enable_invariants=False
        )
        smt_time = time.perf_counter() - t0

        if smt_res.rejected_by_subset:
            smt_status = "REJECTED (Subset Gate)"
            smt_detail = f"Gate error: {smt_res.subset_rejection_reason}"
        else:
            first_status = smt_res.claim_results[0].status.value if smt_res.claim_results else "NO_CLAIMS"
            smt_status = first_status
            smt_detail = f"Claims: {smt_res.n_claims}, Verified: {smt_res.n_verified}, CE: {smt_res.n_counterexample}"

        print(f"    SMT Status: {smt_status} (Time: {smt_time:.3f}s)")
        print(f"    SMT Details: {smt_detail}")

        # --- B. RUN LLM JUDGE + ACTOR-CRITIC PIPELINE ---
        print("\n>>> [Pipeline 2: LLM Judge & Actor-Critic Refinement]...")
        t1 = time.perf_counter()
        refine_res = refinement_loop.run(code=code, initial_explanation=case["initial_explanation"])
        veri_time = time.perf_counter() - t1

        initial_soundness = refine_res.initial_soundness
        final_soundness = refine_res.final_soundness
        converged = refine_res.converged
        iterations = refine_res.total_iterations
        initial_verdict = refine_res.steps[0].judge_result.verdict.value if refine_res.steps else "UNKNOWN"

        print(f"    Judge Initial Verdict: {initial_verdict} (Soundness: {initial_soundness:.2f})")
        print(f"    Refinement: Converged={converged} in {iterations} iteration(s) -> Final Soundness: {final_soundness:.2f}")
        print(f"    Execution Time: {veri_time:.2f}s")

        summary_rows.append({
            "id": case["id"],
            "gt": case["ground_truth_code"],
            "smt_status": smt_status,
            "smt_time": smt_time,
            "judge_initial": initial_verdict,
            "judge_soundness_start": initial_soundness,
            "judge_soundness_end": final_soundness,
            "converged": converged,
            "iterations": iterations,
            "veri_time": veri_time,
        })

    # Print Final Summary Table
    print("\n" + "=" * 90)
    print("FINAL COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Benchmark Problem':<38} | {'Ground Truth':<12} | {'SMT Pipeline (Z3)':<20} | {'veri-pipeline (Actor-Critic)':<20}")
    print("-" * 96)
    for r in summary_rows:
        conv_str = f"Conv ({r['iterations']} iters, S:{r['judge_soundness_end']:.2f})" if r['converged'] else f"Fail ({r['iterations']} iters)"
        print(f"{r['id'][:38]:<38} | {r['gt']:<12} | {r['smt_status'][:20]:<20} | {conv_str:<20}")
    print("=" * 96)


if __name__ == "__main__":
    run_benchmark()

