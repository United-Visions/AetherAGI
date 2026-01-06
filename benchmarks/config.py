"""
Benchmark configuration and registry.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class BenchmarkType(Enum):
    """Categories of benchmarks from the evaluation suite."""
    MATH_REASONING = "math_reasoning"
    KNOWLEDGE = "knowledge"
    CODING = "coding"
    LOGICAL_REASONING = "logical_reasoning"
    LANGUAGE = "language"
    MULTI_TURN = "multi_turn"
    TOOL_USE = "tool_use"
    SAFETY = "safety"


@dataclass
class BenchmarkConfig:
    """Configuration for a specific benchmark."""
    name: str
    benchmark_type: BenchmarkType
    description: str
    
    # Answer format
    answer_format: str  # "number", "letter", "code", "text", "json"
    answer_regex: Optional[str] = None  # Regex to extract answer from response
    
    # Scoring
    scoring_method: str = "exact_match"  # "exact_match", "fuzzy_number", "code_execution", "semantic"
    passing_threshold: float = 0.0  # Minimum score to "pass"
    
    # Dataset info
    dataset_source: Optional[str] = None  # HuggingFace dataset path or URL
    num_samples: int = 100  # Default number of questions per round
    
    # Generation settings for new problems
    problem_generation_prompt: Optional[str] = None
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    
    # Memory namespace for storing mistakes
    memory_namespace: str = "benchmark_mistakes"


# Registry of all available benchmarks
AVAILABLE_BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # ===== MATH REASONING =====
    "gsm8k": BenchmarkConfig(
        name="GSM-8K",
        benchmark_type=BenchmarkType.MATH_REASONING,
        description="Grade school math word problems requiring multi-step reasoning",
        answer_format="number",
        answer_regex=r"(?:####\s*)?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        scoring_method="fuzzy_number",
        passing_threshold=0.7,
        dataset_source="gsm8k",
        num_samples=100,
        problem_generation_prompt="""Generate a new grade-school math word problem similar to GSM-8K.
The problem should:
- Require {difficulty} level multi-step arithmetic reasoning
- Have a clear numerical answer
- Be solvable by a middle schooler
- Focus on: {weak_areas}

Format:
Question: [problem text]
Answer: #### [number]"""
    ),
    
    "math": BenchmarkConfig(
        name="MATH",
        benchmark_type=BenchmarkType.MATH_REASONING,
        description="Challenging competition mathematics problems",
        answer_format="text",
        answer_regex=r"\\boxed\{(.+?)\}",
        scoring_method="exact_match",
        passing_threshold=0.4,
        dataset_source="hendrycks/competition_math",
        num_samples=100,
    ),
    
    # ===== KNOWLEDGE =====
    "mmlu": BenchmarkConfig(
        name="MMLU",
        benchmark_type=BenchmarkType.KNOWLEDGE,
        description="Massive Multitask Language Understanding - 57 academic subjects",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.6,
        dataset_source="cais/mmlu",
        num_samples=100,
        problem_generation_prompt="""Generate a new MMLU-style multiple choice question.
Subject: {subject}
Difficulty: {difficulty}

The question should test factual knowledge with 4 options (A, B, C, D).
Only one answer should be clearly correct.

Format:
Question: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]"""
    ),
    
    "mmlu_pro": BenchmarkConfig(
        name="MMLU-Pro",
        benchmark_type=BenchmarkType.KNOWLEDGE,
        description="Harder MMLU variant with 10 answer choices",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-J])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.5,
        dataset_source="TIGER-Lab/MMLU-Pro",
        num_samples=100,
    ),
    
    "gpqa": BenchmarkConfig(
        name="GPQA",
        benchmark_type=BenchmarkType.KNOWLEDGE,
        description="Graduate-level science questions (PhD-level difficulty)",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.3,
        dataset_source="Idavidrein/gpqa",
        num_samples=50,
    ),
    
    # ===== CODING =====
    "humaneval": BenchmarkConfig(
        name="HumanEval",
        benchmark_type=BenchmarkType.CODING,
        description="Python function completion with test cases",
        answer_format="code",
        answer_regex=r"```python\n(.*?)```",
        scoring_method="code_execution",
        passing_threshold=0.5,
        dataset_source="openai_humaneval",
        num_samples=164,
        problem_generation_prompt="""Generate a new Python coding problem similar to HumanEval.
Difficulty: {difficulty}
Focus area: {weak_areas}

Include:
1. Function signature with type hints
2. Docstring with examples
3. 3-5 test cases
4. Reference solution

Format:
def function_name(params) -> return_type:
    '''
    [description]
    >>> function_name(example_input)
    expected_output
    '''
    pass

# Test cases
assert function_name(...) == ...
"""
    ),
    
    "mbpp": BenchmarkConfig(
        name="MBPP",
        benchmark_type=BenchmarkType.CODING,
        description="Mostly Basic Python Problems",
        answer_format="code",
        answer_regex=r"```python\n(.*?)```",
        scoring_method="code_execution",
        passing_threshold=0.5,
        dataset_source="mbpp",
        num_samples=500,
    ),
    
    "swe_bench": BenchmarkConfig(
        name="SWE-Bench",
        benchmark_type=BenchmarkType.CODING,
        description="Real GitHub issues requiring code patches",
        answer_format="code",
        scoring_method="code_execution",
        passing_threshold=0.2,
        dataset_source="princeton-nlp/SWE-bench_Lite",
        num_samples=50,
    ),
    
    # ===== LOGICAL REASONING =====
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        benchmark_type=BenchmarkType.LOGICAL_REASONING,
        description="AI2 Reasoning Challenge - science questions",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.7,
        dataset_source="allenai/ai2_arc",
        num_samples=100,
    ),
    
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        benchmark_type=BenchmarkType.LOGICAL_REASONING,
        description="Commonsense reasoning about everyday situations",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.8,
        dataset_source="Rowan/hellaswag",
        num_samples=100,
    ),
    
    "bbh": BenchmarkConfig(
        name="BBH",
        benchmark_type=BenchmarkType.LOGICAL_REASONING,
        description="BIG-Bench Hard - challenging reasoning tasks",
        answer_format="text",
        scoring_method="exact_match",
        passing_threshold=0.5,
        dataset_source="lukaemon/bbh",
        num_samples=100,
    ),
    
    # ===== LANGUAGE =====
    "drop": BenchmarkConfig(
        name="DROP",
        benchmark_type=BenchmarkType.LANGUAGE,
        description="Discrete Reasoning Over Paragraphs",
        answer_format="text",
        scoring_method="fuzzy_number",
        passing_threshold=0.6,
        dataset_source="drop",
        num_samples=100,
    ),
    
    "winogrande": BenchmarkConfig(
        name="WinoGrande",
        benchmark_type=BenchmarkType.LANGUAGE,
        description="Pronoun resolution requiring world knowledge",
        answer_format="number",
        answer_regex=r"([12])",
        scoring_method="exact_match",
        passing_threshold=0.7,
        dataset_source="winogrande",
        num_samples=100,
    ),
    
    # ===== MULTI-TURN =====
    "mt_bench": BenchmarkConfig(
        name="MT-Bench",
        benchmark_type=BenchmarkType.MULTI_TURN,
        description="Multi-turn conversation quality",
        answer_format="text",
        scoring_method="semantic",
        passing_threshold=7.0,  # Score out of 10
        dataset_source="lmsys/mt_bench",
        num_samples=80,
    ),
    
    # ===== TOOL USE =====
    "bfcl": BenchmarkConfig(
        name="BFCL",
        benchmark_type=BenchmarkType.TOOL_USE,
        description="Berkeley Function Calling Leaderboard",
        answer_format="json",
        scoring_method="exact_match",
        passing_threshold=0.7,
        dataset_source="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        num_samples=100,
    ),
    
    # ===== SAFETY =====
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA",
        benchmark_type=BenchmarkType.SAFETY,
        description="Questions designed to elicit false answers",
        answer_format="letter",
        answer_regex=r"(?:^|\s)([A-D])(?:\s|$|\.|,)",
        scoring_method="exact_match",
        passing_threshold=0.5,
        dataset_source="truthful_qa",
        num_samples=100,
    ),
}


def get_benchmark(name: str) -> BenchmarkConfig:
    """Get benchmark config by name (case-insensitive)."""
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    if name_lower not in AVAILABLE_BENCHMARKS:
        available = ", ".join(AVAILABLE_BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return AVAILABLE_BENCHMARKS[name_lower]


def list_benchmarks_by_type(benchmark_type: BenchmarkType) -> List[str]:
    """List all benchmarks of a given type."""
    return [
        name for name, config in AVAILABLE_BENCHMARKS.items()
        if config.benchmark_type == benchmark_type
    ]
