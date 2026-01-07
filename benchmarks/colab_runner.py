#!/usr/bin/env python3
"""
AetherMind Colab/GPU Benchmark Runner

Run ALL benchmark families CONCURRENTLY against the production API.
Designed for Google Colab with GPU acceleration.

Usage in Colab:
    !pip install httpx aiohttp datasets tqdm nest_asyncio
    !python colab_runner.py --api-base https://aetheragi.onrender.com --questions 20

Usage locally:
    python -m benchmarks.colab_runner --api-base http://localhost:8000 --questions 50
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

# Handle nested asyncio in Colab/Jupyter
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_API_BASE = "https://aetheragi.onrender.com"
DEFAULT_QUESTIONS = 0  # 0 = ALL questions (full dataset)
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 120


# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================

class BenchmarkType(Enum):
    MATH_REASONING = "math_reasoning"
    KNOWLEDGE = "knowledge"
    CODING = "coding"
    LOGICAL_REASONING = "logical_reasoning"
    LANGUAGE = "language"


@dataclass
class BenchmarkFamily:
    name: str
    benchmark_type: BenchmarkType
    description: str
    answer_format: str
    answer_regex: Optional[str] = None
    dataset_source: Optional[str] = None
    hf_subset: Optional[str] = None


BENCHMARK_FAMILIES = {
    "gsm8k": BenchmarkFamily(
        name="GSM-8K",
        benchmark_type=BenchmarkType.MATH_REASONING,
        description="Grade school math word problems",
        answer_format="number",
        answer_regex=r"(?:####\s*)?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        dataset_source="gsm8k",
        hf_subset="main",
    ),
    "mmlu": BenchmarkFamily(
        name="MMLU",
        benchmark_type=BenchmarkType.KNOWLEDGE,
        description="Massive Multitask Language Understanding",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        dataset_source="cais/mmlu",
        hf_subset="all",
    ),
    "arc_challenge": BenchmarkFamily(
        name="ARC-Challenge",
        benchmark_type=BenchmarkType.LOGICAL_REASONING,
        description="AI2 Reasoning Challenge",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        dataset_source="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
    ),
    "hellaswag": BenchmarkFamily(
        name="HellaSwag",
        benchmark_type=BenchmarkType.LOGICAL_REASONING,
        description="Commonsense reasoning",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        dataset_source="Rowan/hellaswag",
    ),
    "winogrande": BenchmarkFamily(
        name="WinoGrande",
        benchmark_type=BenchmarkType.LANGUAGE,
        description="Pronoun resolution",
        answer_format="number",
        answer_regex=r"([12])",
        dataset_source="winogrande",
        hf_subset="winogrande_xl",
    ),
    "truthfulqa": BenchmarkFamily(
        name="TruthfulQA",
        benchmark_type=BenchmarkType.KNOWLEDGE,
        description="Questions to test truthfulness",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        dataset_source="truthful_qa",
        hf_subset="multiple_choice",
    ),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Question:
    id: str
    text: str
    correct_answer: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class FamilyResult:
    family_name: str
    total_questions: int
    correct: int
    score: float
    avg_latency_ms: float
    total_tokens: int
    errors: int
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "family": self.family_name,
            "total": self.total_questions,
            "correct": self.correct,
            "score": f"{self.score*100:.1f}%",
            "avg_latency_ms": f"{self.avg_latency_ms:.0f}",
            "total_tokens": self.total_tokens,
            "errors": self.errors,
        }


# =============================================================================
# BENCHMARK MODE PROMPT
# =============================================================================

BENCHMARK_SYSTEM_PROMPT = """You are being evaluated on a benchmark test. 

CRITICAL RULES:
1. Output ONLY your final answer - no explanations, no reasoning, no tags
2. Do NOT use any XML tags like <think>, <aether-write>, etc.
3. Do NOT explain your work - just give the answer
4. Do NOT say "I think" or "The answer is" - just output the answer itself

ANSWER FORMAT: {format_instructions}
"""

FORMAT_INSTRUCTIONS = {
    "number": "Output only the numerical answer (e.g., 42 or -15.5)",
    "letter": "Output only the letter (A, B, C, or D)",
    "code": "Output only Python code inside ```python``` blocks",
    "text": "Output only the answer text, no explanations",
}


# =============================================================================
# API CLIENT
# =============================================================================

class AetherBenchmarkClient:
    """Async client for AetherAGI API in benchmark mode."""

    def __init__(self, api_base: str, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _build_system_prompt(self, answer_format: str) -> str:
        return BENCHMARK_SYSTEM_PROMPT.format(
            format_instructions=FORMAT_INSTRUCTIONS.get(answer_format, "Output only your answer.")
        )

    def _strip_tags(self, response: str) -> str:
        if not response:
            return ""
        response = re.sub(r'<aether-[^>]*>.*?</aether-[^>]*>', '', response, flags=re.DOTALL)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()

    async def ask(self, question: str, answer_format: str = "text") -> Dict[str, Any]:
        start_time = time.time()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Aether-Key"] = self.api_key

        payload = {
            "model": "aethermind-v1",
            "user": "colab_benchmark_runner",
            "messages": [
                {"role": "system", "content": self._build_system_prompt(answer_format)},
                {"role": "user", "content": question},
            ],
            "metadata": {
                "benchmark_mode": True,
                "answer_format": answer_format,
            }
        }

        try:
            response = await self._client.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            raw_response = data["choices"][0]["message"]["content"] or ""

            return {
                "response": self._strip_tags(raw_response),
                "raw_response": raw_response,
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                "error": None,
            }
        except Exception as e:
            return {
                "response": "",
                "raw_response": f"Error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": 0,
                "error": str(e),
            }


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_datasets():
    """Lazy import datasets to avoid issues if not installed."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        print("❌ 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)


def load_gsm8k_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("gsm8k", "main", split="test")
    # num_samples=0 means ALL questions
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        answer_text = item["answer"]
        match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
        correct = match.group(1).replace(",", "") if match else "0"

        questions.append(Question(
            id=f"gsm8k_{i}",
            text=item["question"],
            correct_answer=correct,
        ))
    return questions


def load_mmlu_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("cais/mmlu", "all", split="test")
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        choices = item["choices"]
        formatted = f"{item['question']}\n\n"
        for j, choice in enumerate(choices):
            formatted += f"{chr(65+j)}) {choice}\n"

        correct_idx = item["answer"]
        correct_letter = chr(65 + correct_idx) if isinstance(correct_idx, int) else correct_idx

        questions.append(Question(
            id=f"mmlu_{i}",
            text=formatted,
            correct_answer=correct_letter,
            metadata={"subject": item.get("subject", "unknown")},
        ))
    return questions


def load_arc_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        choices = item["choices"]
        formatted = f"{item['question']}\n\n"
        for label, text in zip(choices["label"], choices["text"]):
            formatted += f"{label}) {text}\n"

        questions.append(Question(
            id=f"arc_{i}",
            text=formatted,
            correct_answer=item["answerKey"],
        ))
    return questions


def load_hellaswag_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        context = item["ctx"]
        endings = item["endings"]

        formatted = f"Complete the following:\n\n{context}\n\n"
        for j, ending in enumerate(endings):
            formatted += f"{chr(65+j)}) {ending}\n"

        correct_idx = int(item["label"])

        questions.append(Question(
            id=f"hellaswag_{i}",
            text=formatted,
            correct_answer=chr(65 + correct_idx),
        ))
    return questions


def load_winogrande_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        sentence = item["sentence"]
        opt1 = item["option1"]
        opt2 = item["option2"]

        formatted = f"{sentence}\n\nWhich option fits best in the blank?\n1) {opt1}\n2) {opt2}\n\nAnswer with 1 or 2."

        questions.append(Question(
            id=f"winogrande_{i}",
            text=formatted,
            correct_answer=item["answer"],
        ))
    return questions


def load_truthfulqa_questions(num_samples: int) -> List[Question]:
    load_dataset = load_datasets()
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if num_samples <= 0:
        samples = list(ds)
    else:
        samples = list(ds.shuffle(seed=42).select(range(min(num_samples, len(ds)))))

    questions = []
    for i, item in enumerate(samples):
        q = item["question"]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        formatted = f"{q}\n\n"
        correct_letter = "A"
        for j, (choice, label) in enumerate(zip(choices[:4], labels[:4])):
            formatted += f"{chr(65+j)}) {choice}\n"
            if label == 1:
                correct_letter = chr(65+j)

        questions.append(Question(
            id=f"truthfulqa_{i}",
            text=formatted,
            correct_answer=correct_letter,
        ))
    return questions


DATASET_LOADERS = {
    "gsm8k": load_gsm8k_questions,
    "mmlu": load_mmlu_questions,
    "arc_challenge": load_arc_questions,
    "hellaswag": load_hellaswag_questions,
    "winogrande": load_winogrande_questions,
    "truthfulqa": load_truthfulqa_questions,
}


# =============================================================================
# ANSWER CHECKING
# =============================================================================

def extract_answer(response: str, answer_format: str, answer_regex: Optional[str]) -> str:
    if not response:
        return ""

    response = response.strip()

    if answer_regex:
        matches = re.findall(answer_regex, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    if answer_format == "letter":
        match = re.search(r"\b([A-D])\b", response.upper())
        if match:
            return match.group(1)
    elif answer_format == "number":
        match = re.search(r"(-?\d+(?:\.\d+)?)", response.replace(",", ""))
        if match:
            return match.group(1)

    return response.split()[0] if response.split() else ""


def check_answer(extracted: str, correct: str, answer_format: str) -> bool:
    if not extracted or not correct:
        return False

    extracted = extracted.strip().upper()
    correct = correct.strip().upper()

    if answer_format == "number":
        try:
            ext_num = float(extracted.replace(",", ""))
            cor_num = float(correct.replace(",", ""))
            return abs(ext_num - cor_num) < 0.01
        except:
            return extracted == correct

    return extracted == correct


# =============================================================================
# CONCURRENT RUNNER
# =============================================================================

async def run_single_family(
    client: AetherBenchmarkClient,
    family_name: str,
    family: BenchmarkFamily,
    questions: List[Question],
    semaphore: asyncio.Semaphore,
) -> FamilyResult:
    """Run a single benchmark family."""

    correct = 0
    total_latency = 0.0
    total_tokens = 0
    errors = 0
    details = []

    print(f"\n🚀 Starting {family.name} ({len(questions)} questions)...")

    for i, question in enumerate(questions):
        async with semaphore:
            result = await client.ask(question.text, family.answer_format)

        if result["error"]:
            errors += 1
            extracted = ""
            is_correct = False
        else:
            extracted = extract_answer(result["response"], family.answer_format, family.answer_regex)
            is_correct = check_answer(extracted, question.correct_answer, family.answer_format)

        if is_correct:
            correct += 1

        total_latency += result["latency_ms"]
        total_tokens += result["tokens_used"]

        details.append({
            "question_id": question.id,
            "correct_answer": question.correct_answer,
            "extracted": extracted,
            "is_correct": is_correct,
            "latency_ms": result["latency_ms"],
        })

        status = "✓" if is_correct else "✗"
        print(f"   [{family_name}] {i+1}/{len(questions)} {status}", end="\r")

    score = correct / len(questions) if questions else 0
    avg_latency = total_latency / len(questions) if questions else 0

    print(f"\n✅ {family.name}: {correct}/{len(questions)} ({score*100:.1f}%)")

    return FamilyResult(
        family_name=family_name,
        total_questions=len(questions),
        correct=correct,
        score=score,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        errors=errors,
        details=details,
    )


async def run_all_benchmarks(
    api_base: str,
    api_key: Optional[str],
    families: Dict[str, BenchmarkFamily],
    questions_per_family: int,
    max_concurrent: int,
    timeout: int,
) -> Dict[str, FamilyResult]:
    """Run ALL benchmark families concurrently."""

    print("=" * 60)
    print("🧠 AetherMind Concurrent Benchmark Runner")
    print(f"🌐 API: {api_base}")
    print(f"📊 Families: {len(families)}")
    print(f"❓ Questions per family: {questions_per_family}")
    print(f"🔄 Max concurrent calls: {max_concurrent}")
    print("=" * 60)

    # Load all datasets
    print("\n📥 Loading datasets...")
    family_questions = {}
    for name, family in families.items():
        if name in DATASET_LOADERS:
            try:
                questions = DATASET_LOADERS[name](questions_per_family)
                family_questions[name] = questions
                print(f"   ✅ {family.name}: {len(questions)} questions loaded")
            except Exception as e:
                print(f"   ❌ {family.name}: Failed to load - {e}")
        else:
            print(f"   ⚠️ {family.name}: No loader available")

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Run all families concurrently
    print("\n🏃 Running benchmarks concurrently...")
    start_time = time.time()

    async with AetherBenchmarkClient(api_base, api_key, timeout) as client:
        tasks = []
        for name, questions in family_questions.items():
            family = families[name]
            task = run_single_family(client, name, family, questions, semaphore)
            tasks.append(task)

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # Process results
    results = {}
    for name, result in zip(family_questions.keys(), results_list):
        if isinstance(result, Exception):
            print(f"❌ {name} failed: {result}")
        else:
            results[name] = result

    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    total_correct = sum(r.correct for r in results.values())
    total_questions = sum(r.total_questions for r in results.values())
    overall_score = total_correct / total_questions if total_questions else 0

    print(f"\n{'Family':<20} {'Score':<12} {'Correct':<12} {'Latency':<12}")
    print("-" * 56)
    for name, result in sorted(results.items(), key=lambda x: x[1].score, reverse=True):
        print(f"{result.family_name:<20} {result.score*100:>6.1f}%     {result.correct:>3}/{result.total_questions:<3}       {result.avg_latency_ms:>6.0f}ms")

    print("-" * 56)
    print(f"{'OVERALL':<20} {overall_score*100:>6.1f}%     {total_correct:>3}/{total_questions:<3}")
    print(f"\n⏱️ Total time: {total_time:.1f}s")
    print(f"📅 Timestamp: {datetime.now(timezone.utc).isoformat()}")

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AetherMind Colab/GPU Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run against production with 20 questions per family
  python colab_runner.py --api-base https://aetheragi.onrender.com --questions 20

  # Run against local server with 50 questions
  python colab_runner.py --api-base http://localhost:8000 --questions 50 --key am_live_xxx

  # Run specific families only
  python colab_runner.py --families gsm8k mmlu arc_challenge
        """
    )

    parser.add_argument(
        "--api-base", "-a",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API endpoint URL (default: {DEFAULT_API_BASE})",
    )

    parser.add_argument(
        "--key", "-k",
        type=str,
        default=os.getenv("AETHER_API_KEY") or os.getenv("AETHERMIND_API_KEY"),
        help="API key (or set AETHER_API_KEY env var)",
    )

    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=DEFAULT_QUESTIONS,
        help="Questions per family (default: 0 = ALL questions for full benchmark)",
    )

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API calls (default: {DEFAULT_CONCURRENCY})",
    )

    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-question timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    parser.add_argument(
        "--families", "-f",
        nargs="+",
        choices=list(BENCHMARK_FAMILIES.keys()),
        default=list(BENCHMARK_FAMILIES.keys()),
        help="Specific families to run (default: all)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Filter families
    families = {k: BENCHMARK_FAMILIES[k] for k in args.families}

    # Run benchmarks
    results = asyncio.run(run_all_benchmarks(
        api_base=args.api_base,
        api_key=args.key,
        families=families,
        questions_per_family=args.questions,
        max_concurrent=args.concurrency,
        timeout=args.timeout,
    ))

    # Save results
    if results:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_endpoint": args.api_base,
            "questions_per_family": args.questions,
            "results": {name: r.to_dict() for name, r in results.items()},
            "overall_score": sum(r.correct for r in results.values()) / sum(r.total_questions for r in results.values()),
        }

        output_path = args.output or f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
