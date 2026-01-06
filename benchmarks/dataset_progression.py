"""
Dataset Progression System for Adaptive Benchmarks.

Instead of generating fake practice questions, this system progressively
tests through REAL variant datasets from HuggingFace.

Progression Example (GSM family):
1. GSM-8K (Original) - Split into chunks
2. GSM-Hard (Larger numbers)
3. GSM-Symbolic (Varied wording)
4. GSM-Plus (Augmented variants)

This ensures genuine learning evaluation, not pattern memorization.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DatasetFamily(Enum):
    """Families of related benchmark datasets."""
    GSM = "gsm"           # Grade school math
    MMLU = "mmlu"         # Knowledge/understanding
    HUMANEVAL = "humaneval"  # Coding
    MATH = "math"         # Competition math
    ARC = "arc"           # Science reasoning


@dataclass
class DatasetVariant:
    """A specific variant dataset within a family."""
    name: str
    hf_id: str                          # HuggingFace dataset ID
    hf_config: Optional[str] = None     # Config/subset name
    split: str = "test"                 # Which split to use
    description: str = ""
    difficulty_rank: int = 1            # 1 = easiest, higher = harder
    question_field: str = "question"    # Field name for question text
    answer_field: str = "answer"        # Field name for answer
    answer_format: str = "number"       # Expected answer format
    parser: Optional[str] = None        # Custom parser function name
    estimated_size: int = 1000          # Approximate dataset size


@dataclass 
class ProgressionConfig:
    """Configuration for a dataset progression."""
    family: DatasetFamily
    variants: List[DatasetVariant]
    chunk_size: int = 100               # Questions per chunk
    min_datasets_before_prompt: int = 3 # Complete this many before asking to continue
    current_variant_index: int = 0
    current_chunk: int = 0
    completed_variants: List[str] = field(default_factory=list)
    scores_by_variant: Dict[str, List[float]] = field(default_factory=dict)


# ============================================================================
# GSM FAMILY - Grade School Math Progression
# ============================================================================
GSM_VARIANTS = [
    DatasetVariant(
        name="GSM-8K",
        hf_id="openai/gsm8k",
        hf_config="main",
        split="test",
        description="Original grade school math (1,319 questions)",
        difficulty_rank=1,
        question_field="question",
        answer_field="answer",
        answer_format="number",
        parser="gsm8k_answer",
        estimated_size=1319,
    ),
    DatasetVariant(
        name="GSM-Hard",
        hf_id="reasoning-machines/gsm-hard",
        split="train",  # This dataset only has train split
        description="Same problems with larger, rarer numbers (1,319 questions)",
        difficulty_rank=2,
        question_field="input",
        answer_field="target",
        answer_format="number",
        estimated_size=1319,
    ),
    DatasetVariant(
        name="GSM-Symbolic",
        hf_id="apple/GSM-Symbolic",
        hf_config="main",
        split="test",
        description="Varied wording/numbers, same logic",
        difficulty_rank=2,
        question_field="question",
        answer_field="answer",
        answer_format="number",
        parser="gsm_symbolic_answer",
        estimated_size=5000,
    ),
    DatasetVariant(
        name="GSM-Plus",
        hf_id="qintongli/GSM-Plus",
        split="test",
        description="Augmented with perturbations (10K+ questions)",
        difficulty_rank=3,
        question_field="question",
        answer_field="answer",
        answer_format="number",
        estimated_size=10000,
    ),
]


# ============================================================================
# MMLU FAMILY - Knowledge Progression  
# ============================================================================
MMLU_VARIANTS = [
    DatasetVariant(
        name="MMLU",
        hf_id="cais/mmlu",
        hf_config="all",
        split="test",
        description="57 academic subjects, 4-choice (14K questions)",
        difficulty_rank=1,
        question_field="question",
        answer_field="answer",
        answer_format="letter",
        estimated_size=14000,
    ),
    DatasetVariant(
        name="MMLU-Pro",
        hf_id="TIGER-Lab/MMLU-Pro",
        split="test",
        description="Harder 10-choice variant",
        difficulty_rank=2,
        question_field="question",
        answer_field="answer",
        answer_format="letter",
        estimated_size=12000,
    ),
    DatasetVariant(
        name="MMLU-Redux",
        hf_id="edinburgh-dawg/mmlu-redux",
        split="test",
        description="Error-corrected version",
        difficulty_rank=2,
        question_field="question",
        answer_field="answer",
        answer_format="letter",
        estimated_size=3000,
    ),
]


# ============================================================================
# HUMANEVAL FAMILY - Coding Progression
# ============================================================================
HUMANEVAL_VARIANTS = [
    DatasetVariant(
        name="HumanEval",
        hf_id="openai/openai_humaneval",
        split="test",
        description="Function completion (164 problems)",
        difficulty_rank=1,
        question_field="prompt",
        answer_field="canonical_solution",
        answer_format="code",
        estimated_size=164,
    ),
    DatasetVariant(
        name="HumanEval+",
        hf_id="evalplus/humanevalplus",
        split="test",
        description="More test cases per problem",
        difficulty_rank=2,
        question_field="prompt",
        answer_field="canonical_solution",
        answer_format="code",
        estimated_size=164,
    ),
    DatasetVariant(
        name="MBPP",
        hf_id="google-research-datasets/mbpp",
        hf_config="sanitized",
        split="test",
        description="Mostly Basic Python Problems (500)",
        difficulty_rank=1,
        question_field="text",
        answer_field="code",
        answer_format="code",
        estimated_size=500,
    ),
    DatasetVariant(
        name="MBPP+",
        hf_id="evalplus/mbppplus",
        split="test",
        description="Stricter test cases",
        difficulty_rank=2,
        question_field="text",
        answer_field="code",
        answer_format="code",
        estimated_size=500,
    ),
]


# ============================================================================
# Registry of all progressions
# ============================================================================
DATASET_PROGRESSIONS: Dict[str, ProgressionConfig] = {
    "gsm": ProgressionConfig(
        family=DatasetFamily.GSM,
        variants=GSM_VARIANTS,
        chunk_size=100,
        min_datasets_before_prompt=3,
    ),
    "mmlu": ProgressionConfig(
        family=DatasetFamily.MMLU,
        variants=MMLU_VARIANTS,
        chunk_size=200,
        min_datasets_before_prompt=2,
    ),
    "humaneval": ProgressionConfig(
        family=DatasetFamily.HUMANEVAL,
        variants=HUMANEVAL_VARIANTS,
        chunk_size=50,
        min_datasets_before_prompt=2,
    ),
}


class DatasetProgressionLoader:
    """
    Loads datasets progressively from HuggingFace.
    
    Features:
    - Chunked loading (don't run all 1319 at once)
    - Automatic progression to next variant
    - Score tracking per variant
    - User prompts after minimum datasets completed
    """
    
    def __init__(self, family: str = "gsm"):
        self.family = family.lower()
        if self.family not in DATASET_PROGRESSIONS:
            raise ValueError(f"Unknown family: {family}. Available: {list(DATASET_PROGRESSIONS.keys())}")
        
        self.config = DATASET_PROGRESSIONS[self.family]
        self._datasets_cache: Dict[str, Any] = {}
        self._current_data: List[Dict] = []
        self._total_loaded: int = 0
    
    @property
    def current_variant(self) -> DatasetVariant:
        """Get the current dataset variant."""
        return self.config.variants[self.config.current_variant_index]
    
    @property
    def has_more_chunks(self) -> bool:
        """Check if current dataset has more chunks to process."""
        return self.config.current_chunk * self.config.chunk_size < len(self._current_data)
    
    @property
    def has_more_variants(self) -> bool:
        """Check if there are more variant datasets to try."""
        return self.config.current_variant_index < len(self.config.variants) - 1
    
    @property
    def completed_minimum(self) -> bool:
        """Check if we've completed the minimum number of datasets."""
        return len(self.config.completed_variants) >= self.config.min_datasets_before_prompt
    
    def get_progress_summary(self) -> str:
        """Get a summary of progress through the dataset family."""
        lines = [
            f"\n{'='*60}",
            f"📊 {self.family.upper()} Family Progress",
            f"{'='*60}",
        ]
        
        for i, variant in enumerate(self.config.variants):
            if variant.name in self.config.completed_variants:
                scores = self.config.scores_by_variant.get(variant.name, [])
                avg = sum(scores) / len(scores) if scores else 0
                status = f"✅ Complete (avg: {avg*100:.1f}%)"
            elif i == self.config.current_variant_index:
                status = f"🔄 In Progress (chunk {self.config.current_chunk + 1})"
            else:
                status = "⏳ Pending"
            
            lines.append(f"  {i+1}. {variant.name:<20} {status}")
            lines.append(f"      └─ {variant.description}")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
    
    async def load_current_dataset(self) -> bool:
        """Load the current variant dataset into memory."""
        variant = self.current_variant
        
        if variant.name in self._datasets_cache:
            self._current_data = self._datasets_cache[variant.name]
            return True
        
        print(f"\n📥 Loading {variant.name} from HuggingFace ({variant.hf_id})...")
        logger.info(f"Loading dataset: {variant.hf_id}")
        
        try:
            from datasets import load_dataset
            
            # Load with appropriate config
            if variant.hf_config:
                dataset = load_dataset(variant.hf_id, variant.hf_config, split=variant.split)
            else:
                dataset = load_dataset(variant.hf_id, split=variant.split)
            
            # Convert to list of dicts
            self._current_data = list(dataset)
            self._datasets_cache[variant.name] = self._current_data
            
            print(f"   ✅ Loaded {len(self._current_data):,} questions")
            logger.info(f"Loaded {len(self._current_data)} questions from {variant.name}")
            
            # Calculate chunks
            num_chunks = (len(self._current_data) + self.config.chunk_size - 1) // self.config.chunk_size
            print(f"   📦 Split into {num_chunks} chunks of ~{self.config.chunk_size} questions each")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load {variant.name}: {e}")
            logger.error(f"Failed to load {variant.name}: {e}")
            return False
    
    def get_next_chunk(self) -> Tuple[List[Dict], int, int]:
        """
        Get the next chunk of questions.
        
        Returns:
            Tuple of (questions, chunk_number, total_chunks)
        """
        if not self._current_data:
            return [], 0, 0
        
        chunk_size = self.config.chunk_size
        start = self.config.current_chunk * chunk_size
        end = min(start + chunk_size, len(self._current_data))
        
        chunk = self._current_data[start:end]
        total_chunks = (len(self._current_data) + chunk_size - 1) // chunk_size
        
        return chunk, self.config.current_chunk + 1, total_chunks
    
    def advance_chunk(self):
        """Move to the next chunk."""
        self.config.current_chunk += 1
    
    def record_chunk_score(self, score: float):
        """Record score for current chunk."""
        variant_name = self.current_variant.name
        if variant_name not in self.config.scores_by_variant:
            self.config.scores_by_variant[variant_name] = []
        self.config.scores_by_variant[variant_name].append(score)
    
    def complete_current_variant(self):
        """Mark current variant as complete and prepare for next."""
        variant_name = self.current_variant.name
        if variant_name not in self.config.completed_variants:
            self.config.completed_variants.append(variant_name)
        
        # Reset chunk counter for next variant
        self.config.current_chunk = 0
        
        # Clear current data to force reload
        self._current_data = []
    
    def advance_to_next_variant(self) -> bool:
        """
        Move to the next variant dataset.
        
        Returns:
            True if there's a next variant, False if all complete
        """
        self.complete_current_variant()
        
        if self.has_more_variants:
            self.config.current_variant_index += 1
            print(f"\n🔄 Advancing to: {self.current_variant.name}")
            return True
        else:
            print(f"\n🎉 All {self.family.upper()} variants complete!")
            return False
    
    def should_prompt_to_continue(self) -> bool:
        """
        Check if we should ask user whether to continue.
        
        Only prompts after completing minimum required datasets.
        """
        return (
            self.completed_minimum and 
            not self.has_more_chunks and 
            self.has_more_variants
        )
    
    def get_final_report(self) -> Dict[str, Any]:
        """Generate final report across all variants."""
        report = {
            "family": self.family,
            "completed_variants": self.config.completed_variants,
            "scores_by_variant": {},
            "overall_average": 0.0,
        }
        
        all_scores = []
        for variant_name, scores in self.config.scores_by_variant.items():
            avg = sum(scores) / len(scores) if scores else 0
            report["scores_by_variant"][variant_name] = {
                "chunks_completed": len(scores),
                "scores": scores,
                "average": avg,
            }
            all_scores.extend(scores)
        
        report["overall_average"] = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return report


def get_available_families() -> List[str]:
    """Get list of available dataset families."""
    return list(DATASET_PROGRESSIONS.keys())


def get_family_info(family: str) -> Dict[str, Any]:
    """Get information about a dataset family."""
    if family not in DATASET_PROGRESSIONS:
        return {}
    
    config = DATASET_PROGRESSIONS[family]
    return {
        "family": family,
        "num_variants": len(config.variants),
        "variants": [
            {
                "name": v.name,
                "description": v.description,
                "difficulty_rank": v.difficulty_rank,
                "estimated_size": v.estimated_size,
            }
            for v in config.variants
        ],
        "chunk_size": config.chunk_size,
        "min_before_prompt": config.min_datasets_before_prompt,
    }
