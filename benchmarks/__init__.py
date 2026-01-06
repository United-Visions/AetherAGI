"""
AetherMind Adaptive Benchmark System

A self-improving benchmark framework with TWO modes:

1. PROGRESSIVE MODE (Recommended) - Real variant datasets
   Uses actual HuggingFace datasets progressively:
   GSM-8K → GSM-Hard → GSM-Symbolic → GSM-Plus
   
   python -m benchmarks.progressive_runner --family gsm
   
2. ADAPTIVE MODE (Legacy) - Generated questions
   Generates practice questions targeting weak areas
   
   python -m benchmarks.runner --benchmark gsm8k --rounds 3

Progressive Mode Features:
    - Chunked execution (100 questions at a time)
    - Real variant datasets (not fake generated questions)
    - Automatic progression to harder variants
    - User prompts after completing minimum datasets

Usage:
    # PROGRESSIVE: Run GSM family with real variants
    python -m benchmarks.progressive_runner --family gsm
    
    # PROGRESSIVE: Run with smaller chunks
    python -m benchmarks.progressive_runner --family gsm --chunk-size 50
    
    # PROGRESSIVE: List available families
    python -m benchmarks.progressive_runner --list
    
    # ADAPTIVE: Run with generated questions
    python -m benchmarks.runner --benchmark gsm8k --rounds 3
    
    # Show leaderboard
    python -m benchmarks.runner --leaderboard

Dataset Families (Progressive Mode):
    - gsm: GSM-8K → GSM-Hard → GSM-Symbolic → GSM-Plus
    - mmlu: MMLU → MMLU-Pro → MMLU-Redux
    - humaneval: HumanEval → HumanEval+ → MBPP → MBPP+
"""

from .config import BenchmarkConfig, AVAILABLE_BENCHMARKS, get_benchmark
from .base import BaseBenchmark, Question, Answer, RoundResult
from .runner import BenchmarkRunner
from .score_tracker import ScoreTracker, LeaderboardTracker
from .mistake_analyzer import MistakeAnalyzer
from .question_generator import QuestionGenerator
from .dataset_progression import (
    DatasetProgressionLoader,
    DatasetVariant,
    ProgressionConfig,
    get_available_families,
    get_family_info,
)
from .progressive_runner import ProgressiveRunner

__all__ = [
    # Config
    "BenchmarkConfig",
    "AVAILABLE_BENCHMARKS",
    "get_benchmark",
    # Base classes
    "BaseBenchmark",
    "Question",
    "Answer",
    "RoundResult",
    # Runners
    "BenchmarkRunner",       # Legacy adaptive mode
    "ProgressiveRunner",     # New progressive mode
    # Dataset Progression
    "DatasetProgressionLoader",
    "DatasetVariant",
    "ProgressionConfig",
    "get_available_families",
    "get_family_info",
    # Tracking
    "ScoreTracker",
    "LeaderboardTracker",
    # Analysis
    "MistakeAnalyzer",
    "QuestionGenerator",
]
