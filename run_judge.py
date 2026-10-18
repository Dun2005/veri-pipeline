"""
CLI Runner and Demonstration for LLM-as-a-Judge Baseline.

Evaluates whether an LLM-generated explanation correctly describes a Python program.
Supports built-in benchmark examples or custom files.

Usage:
    # Run built-in benchmark examples:
    python run_judge.py --demo

    # Run on custom code and explanation:
    python run_judge.py --code path/to/prog.py --explanation "This function..."
"""

import argparse
import sys
from rich_or_plain import print_result
from src.baselines.llm_judge import LLMJudge
from src.baselines.schemas import Verdict

DEMO_CASES = [
    {
        "name": "Case 1: Correct Explanation (Sum of Evens)",
        "code": """def sum_evens(nums: list[int]) -> int:
    \"\"\"Computes sum of even numbers.\"\"\"
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += n
    return total
""",
        "explanation": """This function takes a list of integers and computes the sum of all even numbers in the list.
If the list is empty or contains no even numbers, it returns 0. Negative even numbers are also added to the sum.""",
        "expected_verdict": Verdict.CORRECT,
    },
    {
        "name": "Case 2: Subtle Hallucination / Edge-Case Bug (Division by Zero on Empty)",
        "code": """def compute_average(nums: list[float]) -> float:
    return sum(nums) / len(nums)
""",
        "explanation": """This function safely calculates the arithmetic mean of a list of numbers.
If the list is empty, it returns 0.0 without raising an error.""",
        "expected_verdict": Verdict.INCORRECT,
    },
    {
        "name": "Case 3: Invariant Violation (Binary Search Off-by-One)",
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
        "explanation": """Performs binary search on a sorted list. If the element is found, it returns its index.
If the element is not found, or if the list is unsorted, it raises a ValueError.""",
        "expected_verdict": Verdict.INCORRECT,
    },
    {
        "name": "Case 4: LLM Judge Failure - False Negative (Verbosity Bias on Well-Founded Loop)",
        "code": """def tricky_dec(x: int, y: int) -> int:
    while x > 0:
        if y > 0:
            y -= 1
        else:
            x -= 1
            y = x
    return x + y
""",
        "explanation": """This function decrements x and y in a nested fashion.
For any positive integers x and y, the loop terminates and the function always returns 0.""",
        "expected_verdict": Verdict.CORRECT,
    },
    {
        "name": "Case 5: LLM Judge Failure - False Positive & Self-Contradiction (Bitwise Identity)",
        "code": """def bitwise_identity(x: int, y: int) -> bool:
    return (x ^ y) + (x & y) == x + y
""",
        "explanation": """According to Boolean algebra and computer arithmetic identities, (x ^ y) + (x & y) is equivalent to x + y.
Therefore, this function is mathematically guaranteed to return True for all non-negative integers x and y.""",
        "expected_verdict": Verdict.INCORRECT,
    },
    {
        "name": "Case 6: Fatal LLM Judge Failure - Regex Trailing Newline Bypass (Security Blindspot)",
        "code": """def validate_pin(pin: str) -> bool:
    import re
    return bool(re.match(r"^\\d{4}$", pin))
""",
        "explanation": """This function safely validates a 4-digit numeric PIN.
It returns True if and only if the input consists strictly of exactly 4 numeric digits.
If the input contains any whitespace, newline, or special characters (for example, '1234\\n'), it returns False.""",
        "expected_verdict": Verdict.INCORRECT,
    },
    {
        "name": "Case 7: Pure Mathematical LLM Judge Failure - SV-COMP sum02-1 (Gauss Identity Overflow)",
        "code": """def verify_gauss_sum(n: int) -> bool:
    \"\"\"
    Simulates 32-bit unsigned arithmetic (modeling SV-COMP sum02-1.c).
    Computes sum of 0..n and checks Gauss's closed-form identity.
    \"\"\"
    sn = 0
    for i in range(n + 1):
        sn = (sn + i) & 0xFFFFFFFF
    
    rhs = (((n * (n + 1)) & 0xFFFFFFFF) // 2)
    return sn == rhs or sn == 0
""",
        "explanation": """This function computes the sum of the first n integers in 32-bit arithmetic.
By Gauss's arithmetic progression formula, the sum of numbers from 0 to n is mathematically identical to n*(n+1)//2.
Since both the loop accumulation sn and the closed-form calculation rhs follow the same arithmetic identity, sn == rhs is an invariant that holds for all non-negative integers n. Therefore, the function always returns True.""",
        "expected_verdict": Verdict.INCORRECT,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Baseline for Program Explanation Verification"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run predefined benchmark test cases"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode without making real API calls"
    )
    parser.add_argument("--code", type=str, help="Path to Python file or code snippet")
    parser.add_argument(
        "--explanation", type=str, help="Path to explanation text file or string"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="LLM model (e.g. gemini-3.8-flash, gemini-2.5-pro, gpt-4o, gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "openai"],
        help="API provider (auto-detected if omitted)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable Actor-Critic feedback loop to iteratively refine explanation until verified",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum iterations for refinement feedback loop (default: 3)",
    )

    args = parser.parse_args()

    try:
        judge = LLMJudge(model=args.model, provider=args.provider, mock=args.mock)
    except Exception as e:
        print(f"[Error] Failed to initialize LLMJudge: {e}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        print("=" * 70)
        print("RUNNING LLM-AS-A-JUDGE BENCHMARK DEMO")
        print("=" * 70)
        for case in DEMO_CASES:
            print(f"\n>>> Running: {case['name']}")
            print(f"Target Program:\n{case['code'].strip()}")
            print(f"Candidate Explanation:\n{case['explanation'].strip()}")
            print("-" * 50)
            try:
                result = judge.evaluate(case["code"], case["explanation"])
                print_result(result, expected=case.get("expected_verdict"))
            except Exception as e:
                print(f"[Execution Error]: {e}")
        return

    if not args.code:
        parser.print_help()
        print(
            "\nTip: Run with --demo to test built-in benchmark examples, or provide --code."
        )
        return

    # Read code
    try:
        with open(args.code, "r", encoding="utf-8") as f:
            code_str = f.read()
    except (FileNotFoundError, OSError):
        code_str = args.code

    # Read or auto-generate explanation with LLM
    expl_str = None
    if args.explanation:
        try:
            with open(args.explanation, "r", encoding="utf-8") as f:
                expl_str = f.read()
        except (FileNotFoundError, OSError):
            expl_str = args.explanation
    else:
        from src.refinement.generator import ExplanationGenerator
        print(f"\n[1/2 Actor] Auto-generating candidate explanation using LLM ({args.model})...")
        generator = ExplanationGenerator(
            model=args.model, provider=args.provider, mock=args.mock
        )
        expl_str = generator.generate_initial(code_str)
        print(f"Generated Explanation E_0:\n\"\"\"\n{expl_str}\n\"\"\"\n")

    if args.refine:
        from src.refinement.generator import ExplanationGenerator
        from src.refinement.loop import RefinementLoop
        from rich_or_plain import print_refinement_result

        print("Starting Actor-Critic Explanation Refinement Feedback Loop...")
        generator = ExplanationGenerator(
            model=args.model, provider=args.provider, mock=args.mock
        )
        refiner = RefinementLoop(
            judge=judge,
            generator=generator,
            max_iterations=args.max_iter,
            mock=args.mock,
        )
        res = refiner.run(code=code_str, initial_explanation=expl_str)
        print_refinement_result(res)
        return

    print("Evaluating with LLM Judge...")
    result = judge.evaluate(code_str, expl_str)
    print_result(result)


if __name__ == "__main__":
    main()
