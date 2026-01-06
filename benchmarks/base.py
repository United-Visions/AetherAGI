"""
Base benchmark class that all benchmarks inherit from.
"""
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .config import BenchmarkConfig


@dataclass
class Question:
    """A single benchmark question."""
    id: str
    text: str
    expected_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"
    category: Optional[str] = None


@dataclass
class Answer:
    """Aether's answer to a question."""
    question_id: str
    raw_response: str
    extracted_answer: Any
    is_correct: bool
    reasoning: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass 
class RoundResult:
    """Results from a single benchmark round."""
    round_number: int
    benchmark_name: str
    timestamp: str
    questions: List[Question]
    answers: List[Answer]
    score: float
    total_correct: int
    total_questions: int
    mistakes: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "round_number": self.round_number,
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "score": self.score,
            "total_correct": self.total_correct,
            "total_questions": self.total_questions,
            "questions": [
                {"id": q.id, "text": q.text, "expected": q.expected_answer, "difficulty": q.difficulty}
                for q in self.questions
            ],
            "answers": [
                {
                    "question_id": a.question_id,
                    "extracted_answer": a.extracted_answer,
                    "is_correct": a.is_correct,
                    "latency_ms": a.latency_ms,
                }
                for a in self.answers
            ],
            "mistakes": self.mistakes,
            "metadata": self.metadata,
        }


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    To add a new benchmark:
    1. Create a subclass in benchmarks/implementations/
    2. Implement load_questions() and check_answer()
    3. Register in AVAILABLE_BENCHMARKS in config.py
    """
    
    def __init__(self, config: BenchmarkConfig, data_dir: Optional[Path] = None):
        self.config = config
        self.data_dir = data_dir or Path(__file__).parent / "data" / config.name.lower().replace("-", "_")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def name(self) -> str:
        return self.config.name
    
    @abstractmethod
    def load_questions(self, num_samples: Optional[int] = None) -> List[Question]:
        """
        Load questions from the dataset.
        
        Args:
            num_samples: Number of questions to load (None = all)
            
        Returns:
            List of Question objects
        """
        pass
    
    @abstractmethod
    def check_answer(self, question: Question, response: str) -> Tuple[bool, Any]:
        """
        Check if the response answers the question correctly.
        
        Args:
            question: The question that was asked
            response: Aether's raw response
            
        Returns:
            Tuple of (is_correct, extracted_answer)
        """
        pass
    
    def extract_answer(self, response: str) -> Any:
        """
        Extract the answer from Aether's response using configured regex.
        Can be overridden for custom extraction logic.
        """
        # First, strip any aether action tags that might have leaked through
        response = self._strip_aether_tags(response)
        
        if not self.config.answer_regex:
            return response.strip()
        
        # Try to find answer using regex
        match = re.search(self.config.answer_regex, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: try to find the last occurrence of expected format
        if self.config.answer_format == "number":
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
            if numbers:
                return numbers[-1].replace(",", "")
        
        elif self.config.answer_format == "letter":
            letters = re.findall(r'\b([A-Ja-j])\b', response)
            if letters:
                return letters[-1].upper()
        
        return response.strip()
    
    def _strip_aether_tags(self, response: str) -> str:
        """Remove any AetherMind action tags that leaked through."""
        # Remove all aether-* tags
        response = re.sub(r'<aether-[^>]*>.*?</aether-[^>]*>', '', response, flags=re.DOTALL)
        response = re.sub(r'<aether-[^>]*/>', '', response)
        
        # Remove thinking tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags that look like action tags
        response = re.sub(r'</?(?:aether|think|research|write|sandbox|install|forge)[^>]*>', '', response)
        
        return response.strip()
    
    def format_question_for_aether(self, question: Question) -> str:
        """
        Format the question for Aether. Override for custom formatting.
        Default just returns the question text.
        """
        return question.text
    
    def categorize_mistake(self, question: Question, answer: Answer) -> Dict[str, Any]:
        """
        Analyze why an answer was wrong.
        Returns metadata about the mistake for memory storage.
        
        IMPORTANT: We do NOT store actual question text or expected answers
        to prevent test contamination. Only store the PATTERN of the mistake.
        """
        # Infer mistake type from question characteristics (not content)
        mistake_type = self._infer_mistake_type(question)
        
        return {
            "question_id": question.id,
            # DO NOT store question_text, expected, or actual - that's cheating!
            # Only store the pattern/category of the mistake
            "mistake_type": mistake_type,
            "category": question.category,  # e.g., "percentages", "fractions"
            "difficulty": question.difficulty,
            "timestamp": datetime.utcnow().isoformat(),
            "benchmark": self.name,
        }
    
    def _infer_mistake_type(self, question: Question) -> str:
        """
        Infer mistake type from question characteristics.
        Called during scoring before we discard question content.
        """
        q_text = question.text.lower() if question.text else ""
        
        if "percent" in q_text or "%" in q_text:
            return "percentage_calculation"
        elif "fraction" in q_text or "half" in q_text or "third" in q_text:
            return "fraction_arithmetic"
        elif "ratio" in q_text or "times as" in q_text:
            return "ratio_problem"
        elif "per hour" in q_text or "per minute" in q_text or "rate" in q_text:
            return "rate_problem"
        elif len(q_text) > 300:
            return "multi_step_reasoning"
        else:
            return "general_error"
    
    def score_round(self, questions: List[Question], answers: List[Answer]) -> RoundResult:
        """Score a complete round of questions."""
        correct = sum(1 for a in answers if a.is_correct)
        total = len(questions)
        score = correct / total if total > 0 else 0.0
        
        # Collect mistakes
        mistakes = []
        for q, a in zip(questions, answers):
            if not a.is_correct:
                mistakes.append(self.categorize_mistake(q, a))
        
        return RoundResult(
            round_number=0,  # Set by runner
            benchmark_name=self.name,
            timestamp=datetime.utcnow().isoformat(),
            questions=questions,
            answers=answers,
            score=score,
            total_correct=correct,
            total_questions=total,
            mistakes=mistakes,
        )


class NumberComparisonMixin:
    """Mixin for benchmarks that compare numerical answers."""
    
    def numbers_equal(self, expected: Any, actual: Any, tolerance: float = 0.001) -> bool:
        """Compare two numbers with tolerance for floating point."""
        try:
            # Clean and parse numbers
            exp_clean = str(expected).replace(",", "").replace("$", "").replace("%", "").strip()
            act_clean = str(actual).replace(",", "").replace("$", "").replace("%", "").strip()
            
            exp_num = float(exp_clean)
            act_num = float(act_clean)
            
            # Exact match for integers
            if exp_num == int(exp_num) and act_num == int(act_num):
                return int(exp_num) == int(act_num)
            
            # Tolerance for floats
            return abs(exp_num - act_num) < tolerance
        except (ValueError, TypeError):
            return str(expected).strip().lower() == str(actual).strip().lower()


class MultipleChoiceMixin:
    """Mixin for benchmarks with A/B/C/D style answers."""
    
    def format_choices(self, question: Question) -> str:
        """Format question with choices."""
        text = question.text
        if "choices" in question.metadata:
            choices = question.metadata["choices"]
            if isinstance(choices, list):
                for i, choice in enumerate(choices):
                    letter = chr(65 + i)  # A, B, C, D...
                    text += f"\n{letter}) {choice}"
        return text
    
    def check_letter_answer(self, expected: str, actual: str) -> bool:
        """Compare letter answers."""
        exp_clean = str(expected).strip().upper()
        act_clean = str(actual).strip().upper()
        
        # Handle "A", "A)", "A.", "(A)" formats
        exp_letter = re.match(r'\(?([A-J])\)?\.?', exp_clean)
        act_letter = re.match(r'\(?([A-J])\)?\.?', act_clean)
        
        if exp_letter and act_letter:
            return exp_letter.group(1) == act_letter.group(1)
        
        return exp_clean == act_clean


class CodeExecutionMixin:
    """Mixin for benchmarks that execute code."""
    
    def execute_python_safely(self, code: str, test_cases: List[str], timeout: float = 5.0) -> Tuple[bool, str]:
        """
        Execute Python code with test cases in a sandboxed environment.
        Returns (all_passed, error_message).
        """
        import subprocess
        import tempfile
        
        # Combine code and tests
        full_code = f"{code}\n\n# Test cases\n" + "\n".join(test_cases)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.flush()
                
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    return True, ""
                else:
                    return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
