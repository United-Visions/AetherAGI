# AetherMind Progressive Benchmark System

A **progressive** benchmarking framework that moves from standard tests to harder, real-world variants.

## ⚠️ Philosophy

Don't just solve the training set. Moving from **GSM8K** (standard) to **GSM-Hard** (large numbers) to **GSM-Symbolic** (adversarial wording) proves true understanding, not just memorization.

## Quick Start (Progressive Mode)

Run the "Gauntlet" - this will load dataset levels sequentially, breaking large datasets into manageable chunks.

```bash
# Run GSM8K Gauntlet (Original -> Hard -> Symbolic)
# Splits datasets into chunks of 100 questions (default)
python -m benchmarks.progressive_runner --family gsm

# Run with smaller chunks
python -m benchmarks.progressive_runner --family gsm --chunk-size 50

# List all available families
python -m benchmarks.progressive_runner --list
```

## How It Works

1. **Splitting**: Instead of running all 1,319+ questions at once, it breaks themo chunks (e.g., 100).
2. **Progression**: 
   - **Level 1**: Standard Benchmark (e.g., GSM8K)
   - **Level 2**: Harder Variant (e.g., GSM-Hard)
   - **Level 3**: Adversarial Variant (e.g., GSM-Symbolic)
3. **Confirmation**: A pause between levels lets you decide to continue or stop.

## Available Ladders

### 🧮 Math (GSM Family)
1. **GSM-8K**: The classic grade-school matbenchmark.
2. **GSM-Hard**: Same logic, but with large/rare numbers to break memorization.
3. **GSM-Symbolic**: Variable renaming and adversarial phrasing.
4. **GSM-Plus**: Augmented variants.

### 🧠 owledge (MMLU Family)
1. **MMLU**: Standard 57-subject multiple choice.
2. **MMLU-Pro**: Harder 10-choice variant.
3. **MMLU-Redux**: Error-corrected version.

### 💻 Code (HumanEval Family)
1. **HumanEval**: Standard Python function completion.
2. **HumanEval+**: 80x more test cases.
3. **MBPP**: "Mostly Basic Python Problems" for distribution shift.
4. **MBPP+**: Stricter test cases.

## Directory Structure

```
benchmarks/
├── dataset_progression.py # Defines the levels/variants
├─?runner.py  # The chunked execution logic
├── runner.py              # Classic runner (deprecated)
├── implementations/       # Base logic for checking answers
└── results/               # Saved scores
```

## Adding a New Variant

Edit `benchmarks/dataset_progression.py` to add new variants to existiate new ones.

```python
# benchmarks/dataset_progression.py

MY_FLOW_VARIANTS = [
    DatasetVariant(name="Level 1", hf_id="org/dataset1", ...),
    DatasetVariant(name="Level 2", hf_id="org/dataset2", ...),
]
```
