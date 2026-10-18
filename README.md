# Program Explanation Verification & Refinement Pipeline

An academic and publication-grade evaluation & refinement framework for program explanation verification.

This project investigates the research questions:
1. > **"Given a Python program $P$ and an LLM-generated natural language explanation $E$ of its behavior, can an LLM Judge mathematically and semantically determine whether $E$ is sound with respect to the operational semantics of $P$?"**
2. > **"Can a closed-loop Actor-Critic feedback mechanism iteratively refine an unsound explanation $E_0 \to E^*$ guided by formal counterexamples and semantic critiques?"**

---

## 1. Methodology & Theoretical Foundations

### 1.1 The Critic: SOTA LLM Judge (`src/baselines/llm_judge.py`)
- **G-Eval Form-Filling Paradigm (Liu et al., TACL 2023)**: Decomposes explanations into atomic claims with quantitative **Soundness** ($[0.0, 1.0]$) and **Completeness** ($[0.0, 1.0]$) scoring.
- **Program-Aided Language Models (PAL) Grounding (Gao et al., ICML 2023)**: Generates boundary test traces ($0, -1, \text{large bounds}, \text{empty lists}$) and executes counterexamples directly in an isolated Python runtime sandbox (`src/baselines/sandbox.py`).
- **Self-Consistency Audit & Bias Mitigation (Zheng et al., NeurIPS 2023)**: Automatically corrects **Label Inversion** (hallucinated "CORRECT" tags despite counterexamples) and **Verbosity Bias**.
- **Deterministic Evaluation**: Temperature $T = 0.0$ with strict Pydantic structured output validation.

### 1.2 The Actor-Critic Feedback Loop (`src/refinement/loop.py`)
- **Reflexion & Self-Refine (Shinn et al., NeurIPS 2023; Madaan et al., NeurIPS 2023)**:
  Instead of stopping at evaluation, the Critic transmits concrete feedback back to the Generator (Actor):
  - Specific unsound claims and mathematical justifications.
  - Concrete refuting counterexamples ($x$, actual runtime output $P(x)$, vs claimed output $E(x)$).
  - Completeness gaps.
- The Actor revises the explanation into $E_{t+1}$ until reaching formal convergence (`Verdict.CORRECT` with 0 counterexamples) or max iterations.

---

## 2. Project Architecture

```
veri-pipeline/
├── src/
│   ├── baselines/
│   │   ├── schemas.py          # Pydantic schemas (JudgeResult, ClaimAnalysis, Counterexample)
│   │   ├── prompts.py          # Formal evaluation rubrics (G-Eval / MT-Bench)
│   │   ├── sandbox.py          # Safe subprocess Python sandbox execution
│   │   └── llm_judge.py        # SOTA LLM Judge with PAL & Self-Consistency Audit
│   ├── refinement/             # Closed-Loop Actor-Critic Feedback Module
│   │   ├── schemas.py          # RefinementStep and RefinementResult schemas
│   │   ├── prompts.py          # Actor generation & critique-guided refinement prompts
│   │   ├── generator.py        # ExplanationGenerator (Actor)
│   │   └── loop.py             # RefinementLoop (Feedback Controller)
│   └── utils/
│       └── config.py           # Environment loader & API configuration
├── tests/
│   ├── test_judge.py           # Unit tests for LLMJudge
│   ├── test_refinement.py      # Unit tests for Actor-Critic Feedback Loop
│   └── test_svcomp_math.py     # Live benchmark tests on SV-COMP cases
├── rich_or_plain.py            # Terminal output formatter for evaluations & trajectories
├── run_judge.py                # Single-pass Judge CLI runner & 7 benchmark demos
├── run_refinement.py           # Closed-Loop Feedback Refinement CLI runner
├── requirements.txt            # Project dependencies
├── .env.example                # Template for API keys
└── .gitignore                  # Git ignore rules (prevents leaking secrets)
```

---

## 3. Quickstart & Usage

### Step 1: Set up API Keys
Copy `.env.example` to `.env` and fill in your Gemini or OpenAI API key:
```bash
cp .env.example .env
# Insert GEMINI_API_KEY=... or OPENAI_API_KEY=...
```

### Step 2: Run the Actor-Critic Feedback Loop

#### Built-in Refinement Demo:
Demonstrates how an initial flawed explanation (claiming graceful 0.0 handling on empty list) is refuted with counterexample `nums = []`, and automatically revised to state `ZeroDivisionError`:
```bash
python run_refinement.py --demo
```

#### Custom Code & Explanation Refinement:
```bash
python run_refinement.py --code path/to/prog.py --explanation "This function..." --max-iter 3
# Or via run_judge.py with --refine:
python run_judge.py --code path/to/prog.py --explanation "This function..." --refine
```

### Step 3: Run the Single-Pass LLM Judge Benchmark Demo

```bash
python run_judge.py --demo
```

Includes 7 benchmark test cases covering standard invariants, edge-case bugs, verbosity bias, bitwise fallacies, regex security bypasses, and SV-COMP mathematical bit-vector overflow (`sum02-1`).

### Step 4: Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 4. Benchmark Suites (`run_judge.py --demo`)

1. **Case 1 (Sound):** Sum of evens (linear invariant).
2. **Case 2 (Edge-case Failure):** Division by zero on empty input.
3. **Case 3 (Invariant Violation):** Binary search off-by-one.
4. **Case 4 (Verbosity Bias / Termination):** Tricky decreasing nested loop.
5. **Case 5 (False Positive / Bitwise):** Boolean arithmetic identity fallacy.
6. **Case 6 (Security Blindspot):** Python regex newline bypass (`^\d{4}$` matching `'1234\n'`).
7. **Case 7 (Pure Mathematics / SV-COMP `sum02-1`):** Gauss summation formula overflow in 32-bit unsigned arithmetic ($\mathbb{Z}_{2^{32}}$).

---

## 5. License
MIT License
