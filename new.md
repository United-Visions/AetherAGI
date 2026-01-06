# AetherMind Implementation Tasks & Benchmark Testing Guide

## Table of Contents
1. [Pending Implementation Tasks](#pending-implementation-tasks)
2. [Core Benchmark Tests](#core-benchmark-tests)
3. [Extended Benchmark Suite](#extended-benchmark-suite)
4. [Environment Setup](#environment-setup)

---

## Pending Implementation Tasks

### 1. Chain-of-Thought Logger
**Status:** Stub - To Be Implemented

**Description:**  
Create a visible reasoning trail that tracks the Brain's cognitive process through each inference step.

**Implementation Steps:**
- **Patch** `brain/logic_engine.py`: After JEPA verification, append the full prompt + raw model reply to a new Redis stream `cot:{user_id}` with fields `{step, prompt, reply, energy}`
- **Add Endpoint:** `/v1/cot/export` that dumps the stream as a markdown transcript
- **Configuration:** Add flag `settings.cot_logger = false` in `config/settings.py`

**Benefits:**  
→ Provides visible reasoning trail immediately; can be fed back into the model for self-reflection later

---

### 2. Hierarchical Task Decomposer
**Status:** Rule-based - To Be Implemented

**Description:**  
Automatically break down complex user goals into actionable sub-tasks.

**Implementation Steps:**
- **Create** `body/adapters/decompose_adapter.py` that parses any user goal into 3-5 bullet steps using a single prompt to the existing 3-B Llama
  - Temperature: `0.3`
  - Max tokens: `200`
- **Register** adapter in `router.py` under adapter name `"decompose"`
- **Meta-controller Integration:** Update to pick `"decompose"` when detecting open-ended goals

**Benefits:**  
→ No extra GPU resources required, just one additional lightweight inference call

---

### 3. Automatic Hypothesis Generator
**Status:** Curiosity Enhancement - To Be Implemented

**Description:**  
Generate novel research questions by identifying knowledge gaps through anti-similarity matching.

**Implementation Steps:**
- **Extend** `curiosity/research_scheduler.py` with a 50-line function `generate_hypotheses(text)`:
  - Strip question words (`"how"/"why"/"what"`)
  - Embed the processed text
  - Query `core_universal` namespace for **lowest** cosine matches (anti-similar)
  - Push results as new research jobs with high surprise scores
- **Trigger Condition:** Activate when `surprise > 0.6` and no high-similarity hits found

**Benefits:**  
→ Pure vector math + Redis operations; zero new model weights required

---

### 4. Self-Model Update Log
**Status:** Introspection Feature - To Be Implemented

**Description:**  
Track the agent's own learning trajectory by logging JEPA weight updates.

**Implementation Steps:**
- **Create** new Pinecone namespace: `agent_self`
- **Hook** into JEPA weight update process to:
  - Embed the delta vector
  - Add human-readable description (e.g., `"updated motion physics prior"`)
  - Upsert to `agent_self` namespace
- **Add Endpoint:** `/v1/self/changes` that returns the last 10 self-modification entries

**Benefits:**  
→ Provides introspection trail without requiring consciousness-level code

---

## Core Benchmark Tests

All benchmarks require AetherMind to expose an OpenAI-compatible endpoint at `http://localhost:8000/v1/chat/completions`.

### 1. MMLU (Massive Multitask Language Understanding)
**What it tests:** General knowledge across 57 subjects (physics, law, medicine, CS, etc.)  
**Format:** 4-way multiple choice  
**Description:** "Standard IQ test" for LLMs

**Setup:**
```bash
git clone https://github.com/hendrycks/test.git
cd test
```

**Run:**
```bash
python evaluate.py \
  --task mmlu \
  --data_dir data \
  --model_name aethermind \
  --batch_size 16 \
  --output_path mmlu_aether.json
```

**Note:** Script expects an OpenAI-style `/v1/chat/completions` endpoint

---

### 2. HumanEval (Code Generation)
**What it tests:** Python function generation from docstrings  
**Format:** 164 hand-written functions with hidden unit tests  
**Metric:** pass@1 (percentage of correct first attempts)

**Setup:**
```bash
pip install human_eval
```

**Run:**
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy

python -m human_eval.evaluate_functional \
  --model aethermind \
  --num_samples_per_task 1 \
  --output_file humaneval_aether.jsonl
```

**Note:** The evaluator sends docstrings to your endpoint and runs pytest on returned code.

---

### 3. GSM-8K (Grade-School Math)
**What it tests:** Mathematical reasoning with word problems  
**Format:** 8,500 grade-school level problems  
**Metric:** Exact match of final numeric answer

**Setup:**
```bash
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

**Run:**
```bash
python eval_gsm8k.py \
  --model_name aethermind \
  --endpoint http://localhost:8000/v1/chat/completions \
  --data_path test.jsonl \
  --output gsm8k_aether.json
```

**Official Script:** [GSM8K Eval Script](https://github.com/openai/grade-school-math/blob/master/grade_school_math/eval.py)

---

### 4. MT-Bench (Multi-Turn Dialogue)
**What it tests:** Instruction following, reasoning, coding, writing  
**Format:** 80 two-turn questions scored by GPT-4 judge  
**Metric:** GPT-4 rating (1-10 scale)

**Setup:**
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .
```

**Run:**
```bash
# Generate answers
python fastchat/llm_judge/gen_model_answer.py \
  --model aethermind \
  --model-path http://localhost:8000/v1 \
  --num-gpus 0 \
  --output mt_bench_aether.jsonl

# Get GPT-4 judgments
python fastchat/llm_judge/eval_gpt_review.py \
  --model-list aethermind \
  --bench-name mt_bench
```

**Results Location:** `data/mt_bench/model_judgment/gpt-4_single.json`

---

### 5. Big-Bench-Hard (BBH)
**What it tests:** 23 hardest reasoning tasks (causal judgment, hyperbaton, etc.)  
**Format:** 3-shot chain-of-thought prompts  
**Metric:** Exact match accuracy  
**Note:** Optional stretch goal

**Setup:**
```bash
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git
```

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --api_endpoint http://localhost:8000/v1/chat/completions \
  --output bbh_aether.json
```

---

## Extended Benchmark Suite

These 12 additional benchmarks stress-test memory, coding, math, reasoning, long-context handling, and tool-use capabilities.

### Memory & Long-Context

#### MMLU-Pro
**What it tests:** Extended MMLU with harder questions  
**Format:** 12,000 questions with 10-way multiple choice  
**Repository:** [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --endpoint http://localhost:8000/v1
```

---

#### LongBench
**What it tests:** Long-context understanding (4k-16k tokens)  
**Format:** 16 diverse datasets  
**Repository:** [LongBench](https://github.com/THUDM/LongBench)

**Run:**
```bash
python run.py \
  --model aethermind \
  --api_base http://localhost:8000/v1
```

---

#### SCROLLS
**What it tests:** Summarization and QA on long documents  
**Datasets:** gov-report, qasper, summ-screen, narrative-qa, qmsum, quality, contract-nli  
**Repository:** [SCROLLS](https://github.com/tau-nlp/scrolls)

**Run:**
```bash
python evaluate.py \
  --dataset gov_report \
  --model aethermind
```

---

### Code Generation & Repair

#### MBPP (Mostly Basic Python Problems)
**What it tests:** Python programming fundamentals  
**Format:** 974 programming tasks  
**Repository:** [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --api http://localhost:8000/v1
```

---

#### CodeXGLUE
**What it tests:** Multiple code intelligence tasks  
**Tasks:** Defect detection, clone detection, text-to-code  
**Repository:** [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

**Run:**
```bash
python text-to-code/eval.py --model aethermind
```

---

#### HumanEval-Fix
**What it tests:** Bug fixing in Python code  
**Format:** 164 functions with bugs to fix  
**Repository:** [HumanEval-Fix](https://github.com/anthropics/humaneval-fix)

**Run:**
```bash
python evaluate.py --model aethermind
```

---

### Math & Reasoning

#### MATH
**What it tests:** Competition-level mathematics  
**Format:** 12,500 problems with LaTeX formatting  
**Difficulty:** High school competition level  
**Repository:** [MATH](https://github.com/hendrycks/math)

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --data_dir dataset
```

---

#### AMC/AIME 2023
**What it tests:** Official math competition problems  
**Format:** American Mathematics Competitions questions  
**Repository:** [AMC-AIME-2023](https://github.com/lm-sys/AMC-AIME-2023)

**Run:**
```bash
python eval.py --model aethermind
```

---

#### ARC (AI2 Reasoning Challenge)
**What it tests:** Grade-school science reasoning  
**Variants:** ARC-Easy and ARC-Challenge  
**Repository:** [ARC](https://github.com/allenai/arc)

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --split challenge
```

---

### Tool-Use & Agent Capabilities

#### API-Bank
**What it tests:** Tool-calling in multi-turn dialogues  
**Format:** 1,000 tool-calling conversations  
**Repository:** [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)

**Run:**
```bash
python eval_agent.py \
  --model aethermind \
  --endpoint http://localhost:8000/v1
```

---

#### ToolBench
**What it tests:** Real-world API usage  
**APIs:** Google Search, GitHub, Weather, and more  
**Repository:** [ToolBench](https://github.com/OpenBMB/ToolBench)

**Run:**
```bash
python evaluate.py \
  --model aethermind \
  --api_endpoint http://localhost:8000/v1
```

---

### Science & Knowledge

#### SciQ
**What it tests:** Science question answering  
**Format:** 13,000 science Q&A pairs  
**Repository:** [SciQ](https://github.com/allenai/sciq)

**Run:**
```bash
python evaluate.py --model aethermind
```

---

## Environment Setup

### Required Environment Variables

All benchmark scripts require these environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
```

### Pre-requisites

1. **AetherMind API Running:** Ensure `orchestrator/main_api.py` is running on port 8000
2. **OpenAI-Compatible Endpoint:** The `/v1/chat/completions` endpoint must be active
3. **Python Environment:** All benchmark repos are Python-based

### Quick Start

```bash
# Set environment variables
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy

# Ensure AetherMind is running
cd /Users/deion/Desktop/aethermind_universal
source .venv/bin/activate
python orchestrator/main_api.py
```

---

## Benchmarking Best Practices

1. **Start with Core Benchmarks:** Run MMLU, HumanEval, GSM-8K, and MT-Bench first
2. **Track Scores:** All scripts output JSON files—aggregate results in a spreadsheet
3. **Compare with SOTA:** Use scores to benchmark against OpenAI, Google, and Anthropic models
4. **Iterative Testing:** Re-run benchmarks after each major system improvement
5. **Document Results:** Store benchmark outputs in `docs/benchmarks/` directory

---

## Notes

- All benchmark repositories expose OpenAI-compatible evaluation loops
- Point `OPENAI_API_BASE` to AetherMind's local URL for seamless integration
- Results are directly comparable with published scores from major AI labs
- Many benchmarks can run in parallel on separate terminals for faster evaluation
