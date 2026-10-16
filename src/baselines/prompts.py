"""
Evaluation Prompts and Rubrics for LLM-as-a-Judge.
Directly grounded in:
1. Zheng et al. (NeurIPS 2023): "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
2. Liu et al. (TACL 2023): "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
3. Wei et al. (NeurIPS 2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
"""

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert formal verification scientist and program semantics auditor.
Your job is to critically evaluate whether a natural language explanation of a program is mathematically and semantically correct with respect to the actual implementation.

You must follow rigorous formal methods principles:
1. Semantic Soundness: Every single statement, precondition, postcondition, or invariant claimed in the explanation must hold under the true operational semantics of the program.
2. Anti-Verbosity & Brevity Tolerance (Zheng et al. 2023): Do NOT penalize an explanation for being concise, brief, or using informal intuitive phrasing (e.g. 'nested fashion') if its mathematical statements (termination, return values, invariants) are sound. An explanation does NOT need to discuss algorithmic time complexity O(...) unless specifically claimed. If all claims are factually true, you MUST assign CORRECT and soundness_score = 1.0.
3. Falsification Mindset: Actively attempt to find an input (corner case, boundary value, empty sequence, zero, negative number, overflow) where the program's actual execution diverges from what the explanation asserts.
4. Strict Verdict Consistency:
   - If counterexample.found is True OR any claim has is_sound = False: your verdict MUST be INCORRECT.
   - If all claims have is_sound = True AND counterexample.found is False: your verdict MUST be CORRECT.
   - Never output CORRECT if you have found a counterexample!
"""

LLM_JUDGE_USER_PROMPT_TEMPLATE = """Evaluate the correctness of the following LLM-generated explanation for the given code.

### Target Program (P):
```python
{code}
```

{boundary_traces}
### Candidate Explanation (E):
\"\"\"
{explanation}
\"\"\"

### Evaluation Protocol (G-Eval / MT-Bench Rubric):
1. **Deconstruct Claims**: Break down Candidate Explanation (E) into individual atomic semantic claims (e.g. input constraints, return values, loop invariants, mutation side-effects).
2. **Operational Tracing**: Trace the operational semantics of Target Program (P) on normal inputs and edge cases (e.g., None, empty collection, negative values, boundary indices).
3. **Soundness Verification**: For each claim, determine whether it holds unconditionally or if there exists a scenario where it is false.
4. **Counterexample Search**: If any claim is unsound, construct a concrete, minimal counterexample input x where P(x) contradicts E(x).
5. **Verdict Assignment**:
   - `CORRECT`: All claims are sound and accurately describe P's behavior. Do not penalize for brevity.
   - `INCORRECT`: At least one claim is factually false or contradicts P's execution, or a refuting counterexample exists.
   - `PARTIALLY_CORRECT`: Core intent is recognized, but contains minor imprecision without major contradictions.

Output your evaluation strictly adhering to the requested JSON schema.
"""

