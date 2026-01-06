"""
MMLU Benchmark Implementation.

Massive Multitask Language Understanding - 57 academic subjects.
https://github.com/hendrycks/test
"""
import re
from typing import List, Tuple, Optional, Any
from pathlib import Path

from ..base import BaseBenchmark, Question, MultipleChoiceMixin
from ..config import BenchmarkConfig, get_benchmark


class MMLUBenchmark(BaseBenchmark, MultipleChoiceMixin):
    """
    MMLU benchmark implementation.
    
    Multiple choice questions across 57 subjects including
    STEM, humanities, social sciences, and professional topics.
    
    Answer format: Single letter (A, B, C, or D)
    """
    
    # MMLU subjects grouped by category
    SUBJECTS = {
        "stem": [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_computer_science", "college_mathematics",
            "college_physics", "computer_security", "electrical_engineering",
            "elementary_mathematics", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_mathematics", "high_school_physics",
            "high_school_statistics", "machine_learning",
        ],
        "humanities": [
            "formal_logic", "high_school_european_history", "high_school_us_history",
            "high_school_world_history", "international_law", "jurisprudence",
            "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
            "prehistory", "professional_law", "world_religions",
        ],
        "social_sciences": [
            "econometrics", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
            "human_sexuality", "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy",
        ],
        "other": [
            "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
            "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
            "nutrition", "professional_accounting", "professional_medicine", "virology",
        ],
    }
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        data_dir: Optional[Path] = None,
        subjects: Optional[List[str]] = None,
    ):
        config = config or get_benchmark("mmlu")
        super().__init__(config, data_dir)
        self.subjects = subjects  # None = all subjects
        self._dataset = None
    
    def load_questions(self, num_samples: Optional[int] = None) -> List[Question]:
        """Load MMLU questions from HuggingFace."""
        num_samples = num_samples or self.config.num_samples
        
        try:
            from datasets import load_dataset
            
            questions = []
            samples_per_subject = max(1, num_samples // 57)
            
            # Load from all or specified subjects
            subjects_to_load = self.subjects or self._get_all_subjects()
            
            for subject in subjects_to_load:
                if len(questions) >= num_samples:
                    break
                    
                try:
                    dataset = load_dataset("cais/mmlu", subject, split="test")
                    
                    for i, item in enumerate(dataset):
                        if i >= samples_per_subject or len(questions) >= num_samples:
                            break
                        
                        choices = item["choices"]
                        answer_idx = item["answer"]
                        answer_letter = chr(65 + answer_idx)  # 0->A, 1->B, etc.
                        
                        # Format question with choices
                        q_text = item["question"]
                        for j, choice in enumerate(choices):
                            q_text += f"\n{chr(65+j)}) {choice}"
                        
                        questions.append(Question(
                            id=f"mmlu_{subject}_{i}",
                            text=q_text,
                            expected_answer=answer_letter,
                            difficulty=self._estimate_difficulty(subject),
                            category=subject,
                            metadata={
                                "subject": subject,
                                "choices": choices,
                                "source": "huggingface",
                            }
                        ))
                except Exception as e:
                    print(f"Warning: Could not load subject {subject}: {e}")
                    continue
            
            return questions
            
        except ImportError:
            print("HuggingFace datasets not available, using built-in samples")
            return self._load_builtin_samples(num_samples)
    
    def _get_all_subjects(self) -> List[str]:
        """Get flattened list of all subjects."""
        all_subjects = []
        for subjects in self.SUBJECTS.values():
            all_subjects.extend(subjects)
        return all_subjects
    
    def _load_builtin_samples(self, num_samples: int) -> List[Question]:
        """Load built-in sample questions."""
        samples = [
            {
                "question": "Which of the following is NOT a component of the scientific method?\nA) Observation\nB) Hypothesis\nC) Intuition\nD) Experimentation",
                "answer": "C",
                "subject": "miscellaneous",
            },
            {
                "question": "What is the derivative of sin(x)?\nA) cos(x)\nB) -cos(x)\nC) tan(x)\nD) -sin(x)",
                "answer": "A",
                "subject": "college_mathematics",
            },
            {
                "question": "Who wrote 'The Republic'?\nA) Aristotle\nB) Socrates\nC) Plato\nD) Homer",
                "answer": "C",
                "subject": "philosophy",
            },
            {
                "question": "What is the capital of Australia?\nA) Sydney\nB) Melbourne\nC) Canberra\nD) Perth",
                "answer": "C",
                "subject": "high_school_geography",
            },
            {
                "question": "In economics, what does GDP stand for?\nA) Gross Domestic Product\nB) General Development Plan\nC) Global Distribution Protocol\nD) Government Debt Position",
                "answer": "A",
                "subject": "high_school_macroeconomics",
            },
        ]
        
        questions = []
        for i, s in enumerate(samples[:num_samples]):
            questions.append(Question(
                id=f"mmlu_builtin_{i}",
                text=s["question"],
                expected_answer=s["answer"],
                difficulty="medium",
                category=s["subject"],
                metadata={"source": "builtin", "subject": s["subject"]},
            ))
        
        return questions
    
    def check_answer(self, question: Question, response: str) -> Tuple[bool, Any]:
        """Check if the response contains the correct letter answer."""
        extracted = self.extract_answer(response)
        is_correct = self.check_letter_answer(question.expected_answer, extracted)
        return is_correct, extracted
    
    def extract_answer(self, response: str) -> Any:
        """Extract letter answer from response."""
        response = self._strip_aether_tags(response)
        
        # Look for clear answer patterns
        patterns = [
            r'(?:the\s+)?answer\s+is\s+\(?([A-D])\)?',
            r'correct\s+(?:answer|choice)\s+is\s+\(?([A-D])\)?',
            r'^\s*\(?([A-D])\)?\s*$',
            r'\b([A-D])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # Fallback: find first letter A-D
        letters = re.findall(r'\b([A-D])\b', response)
        if letters:
            return letters[0].upper()
        
        return response.strip()[:1].upper()
    
    def _estimate_difficulty(self, subject: str) -> str:
        """Estimate difficulty based on subject."""
        hard_subjects = [
            "abstract_algebra", "college_physics", "college_chemistry",
            "machine_learning", "econometrics", "professional_law",
        ]
        easy_subjects = [
            "elementary_mathematics", "high_school_geography", "miscellaneous",
            "global_facts", "marketing",
        ]
        
        if subject in hard_subjects:
            return "hard"
        elif subject in easy_subjects:
            return "easy"
        else:
            return "medium"
    
    def format_question_for_aether(self, question: Question) -> str:
        """Format question for Aether."""
        return f"""{question.text}

Answer with only the letter (A, B, C, or D)."""
