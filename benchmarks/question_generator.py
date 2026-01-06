"""
Question generator for adaptive benchmarks.

IMPORTANT: Uses a STATIC LLM (not Aether) to generate practice questions.

Aether is ONLY the test-taker, never the question-maker. This ensures:
1. No self-evaluation bias
2. Fair benchmarking 
3. Genuine learning from mistakes

The static LLM creates NEW questions based on:
1. The benchmark's question format
2. Aether's weak areas identified from mistakes
3. Increasing difficulty levels
"""
import json
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .config import BenchmarkConfig
from .base import Question
from .aether_client import PlainLLMClient

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """A newly generated benchmark question."""
    text: str
    expected_answer: str
    difficulty: str
    category: Optional[str]
    targeting_weaknesses: List[str]


class QuestionGenerator:
    """
    Generates new benchmark questions targeting weak areas.
    
    CRITICAL: This uses a STATIC LLM (PlainLLMClient), NOT Aether.
    
    - Aether = test taker (being evaluated)
    - Static LLM = question generator (external, unbiased)
    
    This separation ensures fair evaluation - Aether cannot influence
    the questions it will be asked.
    """
    
    def __init__(self, llm_client: Optional[PlainLLMClient] = None):
        # Use a STATIC LLM, not Aether, for question generation
        self.llm = llm_client or PlainLLMClient()
    
    async def generate_questions(
        self,
        config: BenchmarkConfig,
        num_questions: int = 10,
        difficulty: str = "medium",
        weak_areas: List[str] = None,
        existing_questions: List[Question] = None,
    ) -> List[Question]:
        """
        Generate new benchmark questions.
        
        Args:
            config: Benchmark configuration
            num_questions: Number of questions to generate
            difficulty: Target difficulty ("easy", "medium", "hard")
            weak_areas: Areas where Aether struggled
            existing_questions: Questions to avoid duplicating
            
        Returns:
            List of new Question objects
        """
        weak_areas = weak_areas or []
        
        logger.info(f"🔧 Generating {num_questions} {difficulty} questions for {config.name}")
        logger.info(f"   Targeting weak areas: {weak_areas if weak_areas else 'general'}")
        print(f"\n🔧 Generating {num_questions} practice questions (difficulty: {difficulty})...")
        if weak_areas:
            print(f"   Targeting weak areas: {', '.join(weak_areas)}")
        
        # Build the generation prompt
        prompt = self._build_generation_prompt(
            config=config,
            num_questions=num_questions,
            difficulty=difficulty,
            weak_areas=weak_areas,
            examples=existing_questions[:3] if existing_questions else [],
        )
        
        logger.debug(f"   Generation prompt: {len(prompt)} chars")
        
        # Generate with LLM
        raw_response = await self.llm.generate(prompt, temperature=0.8)
        
        # Handle None or error responses
        if not raw_response:
            error_msg = "No response from LLM (returned None)"
            logger.error(f"❌ Question generation failed: {error_msg}")
            print(f"⚠️  Question generation failed: {error_msg}")
            return []
        
        if raw_response.startswith("ERROR:"):
            logger.error(f"❌ Question generation failed: {raw_response}")
            print(f"⚠️  Question generation failed: {raw_response}")
            return []
        
        logger.info(f"✅ Received LLM response: {len(raw_response)} chars")
        
        # Parse the response
        questions = self._parse_generated_questions(raw_response, config, difficulty, weak_areas)
        
        if questions:
            logger.info(f"✅ Successfully generated {len(questions)} questions")
            print(f"✅ Generated {len(questions)} practice questions")
        else:
            logger.warning("⚠️ Failed to parse any questions from LLM response")
            print("⚠️  Failed to parse questions from LLM response")
        
        return questions
    
    def _build_generation_prompt(
        self,
        config: BenchmarkConfig,
        num_questions: int,
        difficulty: str,
        weak_areas: List[str],
        examples: List[Question],
    ) -> str:
        """Build the prompt for question generation."""
        
        # Use the benchmark's configured generation prompt if available
        if config.problem_generation_prompt:
            base_prompt = config.problem_generation_prompt.format(
                difficulty=difficulty,
                weak_areas=", ".join(weak_areas) if weak_areas else "general topics",
            )
        else:
            base_prompt = f"""Generate {num_questions} new {config.name} benchmark questions.

Difficulty level: {difficulty}
Target weak areas: {', '.join(weak_areas) if weak_areas else 'general coverage'}

Each question should follow the exact format of this benchmark:
- Answer format: {config.answer_format}
- Scoring method: {config.scoring_method}
"""
        
        # Add examples if available
        if examples:
            base_prompt += "\n\nExample questions from this benchmark:\n"
            for i, q in enumerate(examples[:3], 1):
                base_prompt += f"\n--- Example {i} ---\n"
                base_prompt += f"Question: {q.text}\n"
                base_prompt += f"Answer: {q.expected_answer}\n"
        
        # Add weak area targeting
        if weak_areas:
            base_prompt += f"""

IMPORTANT: Focus on generating questions that test these specific weak areas:
{chr(10).join(f'- {area}' for area in weak_areas)}

Make the questions specifically target these weaknesses while maintaining
the benchmark's standard format.
"""
        
        # Add output format instructions
        base_prompt += f"""

Generate exactly {num_questions} questions. For each question, output in this exact format:

===QUESTION===
[question text]
===ANSWER===
[correct answer]
===END===

Make sure:
1. Each question is unique and novel
2. Answers are in the correct format ({config.answer_format})
3. Questions are at {difficulty} difficulty level
4. Questions specifically test: {', '.join(weak_areas) if weak_areas else 'general skills'}
"""
        
        return base_prompt
    
    def _parse_generated_questions(
        self,
        response: Optional[str],
        config: BenchmarkConfig,
        difficulty: str,
        weak_areas: List[str],
    ) -> List[Question]:
        """Parse the LLM response into Question objects."""
        if not response:
            return []
        
        questions = []
        
        # Split by question delimiter
        pattern = r'===QUESTION===\s*(.*?)\s*===ANSWER===\s*(.*?)\s*===END==='
        matches = re.findall(pattern, response, re.DOTALL)
        
        for i, (q_text, answer) in enumerate(matches):
            q_text = q_text.strip()
            answer = answer.strip()
            
            # Clean up answer based on format
            if config.answer_format == "number":
                # Extract number from answer
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
                if numbers:
                    answer = numbers[0].replace(",", "")
            elif config.answer_format == "letter":
                # Extract letter
                letters = re.findall(r'[A-Ja-j]', answer)
                if letters:
                    answer = letters[0].upper()
            
            questions.append(Question(
                id=f"gen_{config.name}_{i+1}",
                text=q_text,
                expected_answer=answer,
                difficulty=difficulty,
                category=weak_areas[0] if weak_areas else None,
                metadata={
                    "generated": True,
                    "targeting": weak_areas,
                    "source": "question_generator",
                },
            ))
        
        # Fallback: try simpler parsing if structured format failed
        if not questions:
            questions = self._fallback_parse(response, config, difficulty, weak_areas)
        
        return questions
    
    def _fallback_parse(
        self,
        response: str,
        config: BenchmarkConfig,
        difficulty: str,
        weak_areas: List[str],
    ) -> List[Question]:
        """Fallback parsing for less structured responses."""
        questions = []
        
        # Try to find Q: ... A: ... patterns
        qa_pattern = r'(?:Q(?:uestion)?[:\s]*|^\d+[\.\)]\s*)(.*?)(?:\n[Aa](?:nswer)?[:\s]*)(.*?)(?=(?:Q(?:uestion)?[:\s]|^\d+[\.\)]|\Z))'
        matches = re.findall(qa_pattern, response, re.MULTILINE | re.DOTALL)
        
        for i, (q_text, answer) in enumerate(matches):
            q_text = q_text.strip()
            answer = answer.strip().split("\n")[0]  # Take first line of answer
            
            if len(q_text) < 10:  # Skip too-short questions
                continue
                
            questions.append(Question(
                id=f"gen_fallback_{i+1}",
                text=q_text,
                expected_answer=answer,
                difficulty=difficulty,
                category=weak_areas[0] if weak_areas else None,
                metadata={"generated": True, "fallback_parse": True},
            ))
        
        return questions
    
    def generate_questions_sync(
        self,
        config: BenchmarkConfig,
        num_questions: int = 10,
        difficulty: str = "medium",
        weak_areas: List[str] = None,
        existing_questions: List[Question] = None,
    ) -> List[Question]:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_questions(
            config, num_questions, difficulty, weak_areas, existing_questions
        ))


class AdaptiveDifficultyGenerator:
    """
    Generates questions with adaptive difficulty based on performance.
    
    If Aether scores well, increase difficulty.
    If Aether scores poorly, focus on weak areas at same/lower difficulty.
    """
    
    def __init__(self, generator: QuestionGenerator):
        self.generator = generator
        self.difficulty_levels = ["easy", "medium", "hard", "expert"]
    
    async def generate_next_round(
        self,
        config: BenchmarkConfig,
        previous_score: float,
        previous_mistakes: List[Dict[str, Any]],
        current_difficulty: str = "medium",
        num_questions: int = 10,
    ) -> tuple[List[Question], str]:
        """
        Generate questions for the next round based on previous performance.
        
        Returns:
            Tuple of (questions, new_difficulty_level)
        """
        # Determine new difficulty
        new_difficulty = self._adjust_difficulty(
            current_difficulty, previous_score
        )
        
        # Extract weak areas from mistakes
        weak_areas = self._extract_weak_areas(previous_mistakes)
        
        # Generate questions
        questions = await self.generator.generate_questions(
            config=config,
            num_questions=num_questions,
            difficulty=new_difficulty,
            weak_areas=weak_areas,
        )
        
        return questions, new_difficulty
    
    def _adjust_difficulty(self, current: str, score: float) -> str:
        """Adjust difficulty based on score."""
        idx = self.difficulty_levels.index(current) if current in self.difficulty_levels else 1
        
        if score >= 0.9:
            # Crushing it - go harder
            idx = min(idx + 1, len(self.difficulty_levels) - 1)
        elif score >= 0.7:
            # Doing well - slight increase
            idx = min(idx + 1, len(self.difficulty_levels) - 1)
        elif score < 0.4:
            # Struggling - ease up
            idx = max(idx - 1, 0)
        # 0.4-0.7: stay at current difficulty
        
        return self.difficulty_levels[idx]
    
    def _extract_weak_areas(self, mistakes: List[Dict[str, Any]]) -> List[str]:
        """Extract weak area topics from mistakes."""
        areas = set()
        
        for m in mistakes:
            if "category" in m and m["category"]:
                areas.add(m["category"])
            if "inferred_category" in m:
                areas.add(m["inferred_category"])
        
        return list(areas)[:5]
