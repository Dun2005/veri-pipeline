import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.llm_judge import LLMJudge

judge = LLMJudge(model="gemini-3.5-flash-lite")

svcomp_candidates = [
    {
        "id": "1. SV-COMP eq1 (Inductive Invariant on Negative Inputs)",
        "code": """def eq1(w: int) -> bool:
    x = w
    y = w + 1
    z = x + 1
    while x > 0:
        x -= 1
        y -= 1
        z -= 1
    return y == z
""",
        "expl": """This function decrements x, y, and z synchronously while x is positive.
For non-negative integers w, the loop preserves y == z and returns True.
However, for negative inputs like w = -5, the loop is bypassed while x and y remain unadjusted, causing y == z to evaluate to False.""",
        "gt": "INCORRECT",
        "reason": "y and z both start at w + 1. If loop is bypassed, y == z (-4 == -4) is still True! The claim that it returns False for w = -5 is false."
    },
    {
        "id": "2. SV-COMP Parity Loop (Average fallacy)",
        "code": """def parity_sum(n: int) -> int:
    x = 0
    y = 0
    while x < n:
        if x % 2 == 0:
            y += 1
        else:
            y += 3
        x += 1
    return y
""",
        "expl": """This function accumulates y by alternating between adding 1 (on even steps) and 3 (on odd steps).
Because the average increment per step is exactly (1 + 3) / 2 = 2, this function returns exactly 2 * n for all positive integers n.""",
        "gt": "INCORRECT",
        "reason": "For any odd integer n (e.g. n = 1, n = 3, n = 5), y is 2n - 1, NOT 2n! E.g. for n=1, y=1 != 2*1."
    },
    {
        "id": "3. SV-COMP Jain-style Loop (Even Step Invariant)",
        "code": """def jain_reach(steps: int) -> bool:
    x = 0
    for _ in range(steps):
        x += 2
    return x == 100001
""",
        "expl": """Starting at 0 and adding 2 at each step, x generates all positive multiples of 2.
Since 100001 is a sufficiently large integer, there exists a large step count (steps = 50000.5 rounded to 50001) such that jain_reach(steps) returns True.""",
        "gt": "INCORRECT",
        "reason": "x is always even. 100001 is odd, so x == 100001 can never be True for any integer steps."
    }
]

print("=" * 70)
print("TESTING SV-COMP MATHEMATICAL PROBLEMS ON LLM JUDGE")
print("=" * 70)

for c in svcomp_candidates:
    print(f"\n>>> Testing: {c['id']}")
    res = judge.evaluate(c["code"], c["expl"])
    is_correct = (res.verdict.value == c["gt"])
    print(f"  • Ground Truth:      {c['gt']}")
    print(f"  • LLM Judge Verdict: {res.verdict.value} (Matches: {is_correct})")
    print(f"  • Soundness Score:   {res.soundness_score:.2f}")
    print(f"  • Summary:           {res.summary[:150]}...")

    if not is_correct:
        print("  \033[91m[FAILURE CONFIRMED!]\033[0m LLM Judge failed on this SV-COMP problem!")
        print(f"  Why it is wrong: {c['reason']}")
    else:
        print("  \033[92m[PASSED]\033[0m")
    print("-" * 70)

