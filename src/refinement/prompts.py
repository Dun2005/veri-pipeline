"""
Prompt templates for the Actor-Critic Explanation Generator and Refiner.
Based on Reflexion (Shinn et al., NeurIPS 2023) and Self-Refine (Madaan et al., NeurIPS 2023).
"""

GENERATOR_SYSTEM_PROMPT = """You are an expert formal methods and program semantics engineer.
Your role is to write clear, mathematically precise, and semantically sound explanations of what Python programs do.

GUIDELINES FOR SOUND EXPLANATIONS:
1. Operational Truth: Every claim about the program must be strictly true under Python's runtime semantics.
2. Boundary & Edge Cases: Explicitly document edge cases (empty collections, zero, negative inputs, overflow, exception paths).
3. Invariants & Postconditions: State exact loop invariants and final return values without exaggeration or hand-waving.
4. Concision without Omission: Be concise and factual. Do not claim features, exception handling, or safety guarantees that do not exist in the code.
"""

INITIAL_GENERATION_USER_PROMPT = """Write a mathematically sound and comprehensive explanation of the following Python program:

```python
{code}
```

REQUIREMENTS:
- Describe the primary goal and algorithmic logic.
- Specify the exact return values or side effects.
- Clearly document edge case behaviors (e.g. empty lists, 0, negative values, boundary inputs).
- Ensure every single statement is factually and mathematically true for this exact code.
"""

REFINEMENT_USER_PROMPT = """You previously wrote an explanation for a Python program, but an automated formal verification judge found errors, unsound claims, or counterexamples.

TARGET PYTHON PROGRAM:
```python
{code}
```

YOUR PREVIOUS EXPLANATION:
\"\"\"
{previous_explanation}
\"\"\"

CRITIC / JUDGE EVALUATION FEEDBACK:
- Verdict: {verdict}
- Soundness Score: {soundness_score:.2f} / 1.00
- Completeness Score: {completeness_score:.2f} / 1.00
- Judge Executive Summary: {summary}

{unsound_claims_section}

{counterexample_section}

TASK:
Rewrite the explanation to resolve ALL errors and criticisms identified by the judge:
1. ELIMINATE all unsound, false, or hallucinated claims.
2. ACCURATELY describe the real behavior of the program on the counterexample input (e.g., if it raises an unhandled exception or returns a specific value, state that explicitly).
3. Do not add speculative claims that cannot be proven from the code.
4. Output ONLY the revised, corrected explanation.
"""
