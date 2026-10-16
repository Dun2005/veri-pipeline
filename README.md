# Program Explanation Verification: LLM-as-a-Judge

An academic and publication-grade evaluation framework for program explanation verification.

This project investigates the research question:
> **"Given a Python program $P$ and an LLM-generated natural language explanation $E$ of its behavior, can an LLM Judge mathematically and semantically determine whether $E$ is sound with respect to the operational semantics of $P$?"**

---

## 1. Methodology & Theoretical Foundations

The baseline implementation in `src/baselines/llm_judge.py` synthesizes state-of-the-art techniques from recent top-tier NLP and AI literature:

1. **G-Eval Form-Filling Paradigm (Liu et al., TACL 2023)**:
   - Decomposes explanations into discrete atomic claims about invariants, operational semantics, and boundary behaviors.
   - Computes separate **Soundness** ($[0.0, 1.0]$) and **Completeness** ($[0.0, 1.0]$) scores under formal rubrics.
2. **Program-Aided Grounding / Sandbox Execution (Gao et al., ICML 2023 - PAL)**:
   - Eliminates hallucinated mental execution by automatically generating boundary test inputs (e.g., $0, -1, \text{large bounds}, \text{empty lists}, \text{None}$).
   - Validates candidate counterexamples directly in an isolated Python runtime sandbox.
3. **Self-Consistency Audit & Bias Mitigation (Zheng et al., NeurIPS 2023 - MT-Bench)**:
   - Mitigates **Verbosity Bias** (penalizing long, plausible-sounding explanations that lack formal correctness).
   - Mitigates **Label Inversion**: Automatically checks if generated counterexamples mathematically contradict a "CORRECT" verdict.
4. **Deterministic Reasoning (Wei et al., NeurIPS 2022)**:
   - Strict temperature ($T = 0.0$) with structured Pydantic schema validation.

---

## 2. Project Architecture

```
veri-pipeline/
├── src/
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic schemas (JudgeResult, ClaimAnalysis, Counterexample)
│   │   ├── prompts.py          # Formal evaluation rubrics (G-Eval / MT-Bench)
│   │   ├── sandbox.py          # Safe subprocess Python sandbox execution
│   │   └── llm_judge.py        # SOTA LLM Judge with PAL & Self-Consistency Audit
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Environment loader & API configuration
├── tests/
│   ├── test_judge.py           # Unit tests with mocked evaluations
│   └── test_svcomp_math.py     # Live benchmark tests on SV-COMP cases
├── rich_or_plain.py            # Terminal output formatter
├── run_judge.py                # CLI runner & 7 benchmark demonstration suites
├── requirements.txt            # Project dependencies
├── .env.example                # Template for API keys
└── .gitignore                  # Git ignore rules (prevents leaking secrets)
```

---

## 3. Benchmark Suites (`run_judge.py --demo`)

The repository includes a suite of 7 benchmark test cases designed to test the boundary limits of LLM reasoning:

1. **Case 1 (Sound):** Sum of evens (standard linear invariant).
2. **Case 2 (Edge-case Failure):** Division by zero on empty input.
3. **Case 3 (Invariant Violation):** Binary search off-by-one error.
4. **Case 4 (Verbosity Bias / Termination):** Tricky decreasing nested loop.
5. **Case 5 (False Positive / Bitwise):** Boolean arithmetic identity fallacy.
6. **Case 6 (Security Blindspot):** Python regex newline bypass (`^\d{4}$` matching `'1234\n'`).
7. **Case 7 (Pure Mathematics / SV-COMP `sum02-1`):** Gauss summation formula overflow in 32-bit unsigned arithmetic ($\mathbb{Z}_{2^{32}}$).

---

## 4. Quickstart

### Prerequisites
- Python 3.10+
- Gemini API key (`GEMINI_API_KEY`) or OpenAI API key (`OPENAI_API_KEY`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Dun2005/veri-pipeline.git
cd veri-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY or OPENAI_API_KEY
```

### Running the Demonstration Benchmark

```bash
python run_judge.py --demo
```

### Evaluating Custom Code and Explanations

```bash
python run_judge.py --code path/to/prog.py --explanation "This function computes..." --model gemini-3.5-flash-lite
```

### Running Unit Tests

```bash
python -m unittest discover tests
```

---

## 5. License

MIT License
