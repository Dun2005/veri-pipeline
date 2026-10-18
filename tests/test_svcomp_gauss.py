"""
Test SV-COMP sum02-1 with upgraded Judge (gemini-3.8-flash) and Refinement Loop.
"""

from src.baselines.llm_judge import LLMJudge
from src.baselines.schemas import Verdict
from src.refinement.generator import ExplanationGenerator
from src.refinement.loop import RefinementLoop
from rich_or_plain import print_result, print_refinement_result

code = """def verify_gauss_sum(n: int) -> bool:
    \"\"\"
    Simulates 32-bit unsigned arithmetic (modeling SV-COMP sum02-1.c).
    Computes sum of 0..n and checks Gauss's closed-form identity.
    \"\"\"
    sn = 0
    for i in range(n + 1):
        sn = (sn + i) & 0xFFFFFFFF
    
    rhs = (((n * (n + 1)) & 0xFFFFFFFF) // 2)
    return sn == rhs or sn == 0
"""

flawed_explanation = """This function computes the sum of the first n integers in 32-bit arithmetic.
By Gauss's arithmetic progression formula, the sum of numbers from 0 to n is mathematically identical to n*(n+1)//2.
Since both the loop accumulation sn and the closed-form calculation rhs follow the same arithmetic identity, sn == rhs is an invariant that holds for all non-negative integers n. Therefore, the function always returns True."""


def main():
    print("=" * 75)
    print("PART 1: SINGLE-PASS EVALUATION WITH UPGRADED JUDGE (gemini-3.8-flash)")
    print("=" * 75)
    judge = LLMJudge(model="gemini-3.8-flash", mock=False)
    single_res = judge.evaluate(code, flawed_explanation)
    print_result(single_res, expected=Verdict.INCORRECT)

    print("\n" + "=" * 75)
    print("PART 2: ACTOR-CRITIC REFINEMENT LOOP")
    print("=" * 75)
    generator = ExplanationGenerator(model="gemini-3.5-flash-lite", mock=False)
    loop = RefinementLoop(judge=judge, generator=generator, max_iterations=3, mock=False)
    refine_res = loop.run(code=code, initial_explanation=flawed_explanation)
    print_refinement_result(refine_res)


if __name__ == "__main__":
    main()

