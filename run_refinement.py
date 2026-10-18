"""
CLI Runner for Actor-Critic Explanation Refinement Feedback Loop.

Demonstrates how an initial hallucinated or flawed code explanation E_0
is iteratively refined through Critic (LLMJudge) counterexamples until reaching E*.

Usage:
    # Run built-in demo showing automatic error correction:
    python run_refinement.py --demo

    # Run on custom code and initial explanation:
    python run_refinement.py --code path/to/prog.py --explanation "This function..."

    # Run in mock mode:
    python run_refinement.py --demo --mock
"""

import argparse
import sys

from rich_or_plain import print_refinement_result
from src.refinement.loop import RefinementLoop
from src.baselines.llm_judge import LLMJudge
from src.refinement.generator import ExplanationGenerator

DEMO_REFINEMENT_CASES = [
    {
        "name": "Demo 1: Division by Zero Edge-Case Flaw (compute_average)",
        "code": """def compute_average(nums: list[float]) -> float:
    return sum(nums) / len(nums)
""",
        "flawed_initial_explanation": """This function safely calculates the arithmetic mean of a list of numbers.
If the list is empty, it returns 0.0 without raising an error.""",
        "description": "Initial explanation falsely claims 0.0 is returned on empty list. The feedback loop should catch ZeroDivisionError and correct the explanation."
    },
    {
        "name": "Demo 2: Binary Search Fallacious Exception (search)",
        "code": """def search(arr: list[int], target: int) -> int:
    low = 0
    high = len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1
""",
        "flawed_initial_explanation": """Performs binary search on a sorted list. If the element is found, it returns its index.
If the element is not found, or if the list is unsorted, it raises a ValueError.""",
        "description": "Initial explanation falsely claims ValueError is raised. The feedback loop should refute this and state -1 is returned."
    }
]


def main():
    parser = argparse.ArgumentParser(
        description="Actor-Critic Feedback Loop for Program Explanation Refinement"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run predefined refinement benchmark demo"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode without calling external APIs"
    )
    parser.add_argument("--code", type=str, help="Path to Python file or raw code snippet")
    parser.add_argument(
        "--explanation", type=str, default=None, help="Optional initial explanation E_0"
    )
    parser.add_argument(
        "--max-iter", type=int, default=3, help="Maximum refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="LLM model to use for both Actor and Critic",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "openai"],
        help="LLM provider (gemini or openai)",
    )

    args = parser.parse_args()

    judge = LLMJudge(model=args.model, provider=args.provider, mock=args.mock)
    generator = ExplanationGenerator(
        model=args.model, provider=args.provider, mock=args.mock
    )
    refiner = RefinementLoop(
        judge=judge, generator=generator, max_iterations=args.max_iter, mock=args.mock
    )

    if args.demo:
        print("=" * 75)
        print("RUNNING ACTOR-CRITIC EXPLANATION REFINEMENT DEMO")
        print("=" * 75)
        for case in DEMO_REFINEMENT_CASES:
            print(f"\n>>> Running Scenario: {case['name']}")
            print(f"Goal: {case['description']}")
            res = refiner.run(
                code=case["code"], initial_explanation=case["flawed_initial_explanation"]
            )
            print_refinement_result(res)
        return

    if not args.code:
        parser.print_help()
        print("\nTip: Run with --demo to see iterative refinement in action.")
        return

    # Load code
    try:
        with open(args.code, "r", encoding="utf-8") as f:
            code_str = f.read()
    except (FileNotFoundError, OSError):
        code_str = args.code

    # Load initial explanation if provided
    expl_str = None
    if args.explanation:
        try:
            with open(args.explanation, "r", encoding="utf-8") as f:
                expl_str = f.read()
        except (FileNotFoundError, OSError):
            expl_str = args.explanation

    res = refiner.run(code=code_str, initial_explanation=expl_str)
    print_refinement_result(res)


if __name__ == "__main__":
    main()

