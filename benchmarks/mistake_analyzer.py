"""
Mistake analyzer for benchmark learning.

Categorizes errors, identifies weak areas, and stores mistakes
in Aether's episodic memory for test-time learning.

IMPORTANT: This module stores only PATTERNS and STRATEGIES,
never actual question/answer pairs. Storing Q&A would be
test contamination (cheating).
"""
import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from collections import Counter, defaultdict
from dataclasses import dataclass

if TYPE_CHECKING:
    from .base import Question, Answer


@dataclass
class MistakePattern:
    """A pattern of mistakes that can be learned from."""
    category: str
    description: str
    examples: List[Dict[str, Any]]
    frequency: int
    severity: float  # 0-1, how impactful this mistake type is


class MistakeAnalyzer:
    """
    Analyzes mistakes to identify patterns and weak areas.
    
    Uses these patterns to:
    1. Generate targeted new questions
    2. Store learnings in episodic memory
    3. Track improvement over time
    """
    
    # Common mistake categories by benchmark type
    MATH_MISTAKE_CATEGORIES = {
        "arithmetic_error": "Basic arithmetic mistakes (addition, subtraction, etc.)",
        "unit_conversion": "Errors in unit conversion or scale",
        "order_of_operations": "Wrong order of operations",
        "missing_step": "Skipped a step in multi-step reasoning",
        "misread_problem": "Misunderstood what the problem was asking",
        "rounding_error": "Incorrect rounding or precision",
        "sign_error": "Wrong sign (positive/negative)",
        "variable_confusion": "Confused variables or quantities",
    }
    
    KNOWLEDGE_MISTAKE_CATEGORIES = {
        "factual_error": "Incorrect factual knowledge",
        "temporal_confusion": "Wrong time period or date",
        "entity_confusion": "Confused similar entities (people, places, etc.)",
        "scope_error": "Answer from wrong scope/domain",
        "partial_knowledge": "Knew part of the answer but not all",
        "outdated_info": "Used outdated information",
    }
    
    CODE_MISTAKE_CATEGORIES = {
        "syntax_error": "Invalid syntax",
        "logic_error": "Correct syntax but wrong logic",
        "edge_case": "Failed edge case handling",
        "type_error": "Type mismatch or conversion error",
        "off_by_one": "Off-by-one indexing error",
        "algorithm_choice": "Used wrong algorithm for the problem",
        "incomplete_solution": "Solution doesn't handle all cases",
    }
    
    def __init__(self):
        self.mistake_patterns: Dict[str, List[MistakePattern]] = defaultdict(list)
    
    def analyze_mistakes(
        self,
        mistakes: List[Dict[str, Any]],
        benchmark_type: str = "math",
    ) -> Dict[str, Any]:
        """
        Analyze a set of mistakes and identify patterns.
        
        Args:
            mistakes: List of mistake dicts from benchmark runs
            benchmark_type: "math", "knowledge", or "code"
            
        Returns:
            Analysis dict with patterns, weak_areas, and recommendations
        """
        if not mistakes:
            return {
                "total_mistakes": 0,
                "patterns": [],
                "weak_areas": [],
                "recommendations": [],
            }
        
        # Categorize each mistake
        categorized = []
        for mistake in mistakes:
            category = self._categorize_mistake(mistake, benchmark_type)
            categorized.append({**mistake, "inferred_category": category})
        
        # Count categories
        category_counts = Counter(m["inferred_category"] for m in categorized)
        
        # Group by difficulty
        difficulty_counts = Counter(m.get("difficulty", "unknown") for m in mistakes)
        
        # Identify weak areas
        weak_areas = self._identify_weak_areas(categorized)
        
        # Generate patterns
        patterns = [
            MistakePattern(
                category=cat,
                description=self._get_category_description(cat, benchmark_type),
                examples=self._get_examples_for_category(categorized, cat, max_examples=3),
                frequency=count,
                severity=count / len(mistakes),
            )
            for cat, count in category_counts.most_common(5)
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, weak_areas)
        
        return {
            "total_mistakes": len(mistakes),
            "category_counts": dict(category_counts),
            "difficulty_breakdown": dict(difficulty_counts),
            "patterns": [
                {
                    "category": p.category,
                    "description": p.description,
                    "frequency": p.frequency,
                    "severity": p.severity,
                    "examples": p.examples,
                }
                for p in patterns
            ],
            "weak_areas": weak_areas,
            "recommendations": recommendations,
        }
    
    def _infer_mistake_type(self, question: 'Question', answer: 'Answer') -> str:
        """Infer the type of mistake without storing the actual Q&A."""
        # This is used during scoring, before we discard the sensitive data
        q_text = question.text.lower() if question else ""
        
        if "percent" in q_text or "%" in q_text:
            return "percentage_calculation"
        elif "fraction" in q_text or "half" in q_text:
            return "fraction_arithmetic"
        elif "ratio" in q_text:
            return "ratio_problem"
        elif "rate" in q_text or "per hour" in q_text:
            return "rate_problem"
        elif len(q_text) > 200:
            return "multi_step_reasoning"
        else:
            return "arithmetic_error"
    
    def _categorize_mistake(self, mistake: Dict[str, Any], benchmark_type: str) -> str:
        """Infer the category of a mistake based on its metadata."""
        # We no longer have access to question_text or expected - only patterns
        # This prevents test contamination
        return mistake.get("mistake_type", "unknown")
    
    def _get_category_description(self, category: str, benchmark_type: str) -> str:
        """Get human-readable description for a category."""
        all_categories = {
            **self.MATH_MISTAKE_CATEGORIES,
            **self.KNOWLEDGE_MISTAKE_CATEGORIES,
            **self.CODE_MISTAKE_CATEGORIES,
        }
        return all_categories.get(category, f"Unknown mistake type: {category}")
    
    def _get_examples_for_category(
        self, mistakes: List[Dict], category: str, max_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """Get example mistake patterns for a category (no actual Q&A stored)."""
        examples = [m for m in mistakes if m.get("inferred_category") == category]
        # Return only safe metadata, never actual question/answer content
        return [
            {
                "category": ex.get("category"),
                "difficulty": ex.get("difficulty"),
                "mistake_type": ex.get("mistake_type"),
            }
            for ex in examples[:max_examples]
        ]
    
    def _identify_weak_areas(self, mistakes: List[Dict]) -> List[str]:
        """Identify weak topic areas based on mistake patterns (not Q&A content)."""
        weak_areas = set()
        
        for m in mistakes:
            # Check difficulty
            if m.get("difficulty") == "easy":
                weak_areas.add(f"basic_{m.get('inferred_category', 'unknown')}")
            
            # Check category (topic area like "percentages", "fractions")
            category = m.get("category")
            if category:
                weak_areas.add(category)
            
            # Check mistake type pattern
            mistake_type = m.get("mistake_type", "")
            if mistake_type:
                weak_areas.add(mistake_type)
        
        return list(weak_areas)[:10]  # Top 10 weak areas
    
    def _generate_recommendations(
        self, patterns: List[MistakePattern], weak_areas: List[str]
    ) -> List[str]:
        """Generate learning recommendations based on analysis."""
        recommendations = []
        
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.category == "arithmetic_error":
                recommendations.append(
                    "Practice double-checking arithmetic calculations"
                )
            elif pattern.category == "missing_step":
                recommendations.append(
                    "Break down multi-step problems explicitly before solving"
                )
            elif pattern.category == "sign_error":
                recommendations.append(
                    "Pay attention to positive/negative signs throughout calculations"
                )
            elif pattern.category == "edge_case":
                recommendations.append(
                    "Always consider edge cases: empty inputs, zeros, negatives"
                )
            elif pattern.category == "factual_error":
                recommendations.append(
                    "Verify factual claims before answering"
                )
        
        if "percentages" in weak_areas:
            recommendations.append("Review percentage calculations and conversions")
        if "fractions" in weak_areas:
            recommendations.append("Practice fraction arithmetic")
        
        return recommendations[:5]
    
    def format_for_memory(self, analysis: Dict[str, Any], benchmark_name: str) -> str:
        """
        Format mistake analysis for storage in episodic memory.
        
        CRITICAL: This stores ONLY strategies and patterns, NEVER actual
        questions or answers. Storing Q&A would be test contamination.
        
        This creates a learnable summary that Aether can recall during
        future benchmark attempts.
        """
        lines = [
            f"# Benchmark Strategy Notes: {benchmark_name}",
            "",
            "## Areas Needing Practice:",
        ]
        
        for area in analysis.get("weak_areas", []):
            lines.append(f"- {area}")
        
        lines.append("\n## Common Mistake Patterns:")
        for pattern in analysis.get("patterns", []):
            lines.append(f"\n### {pattern['category']} (frequency: {pattern['frequency']})")
            lines.append(f"{pattern['description']}")
            # NO examples with actual Q&A - only the pattern type
        
        lines.append("\n## Strategies for Improvement:")
        for rec in analysis.get("recommendations", []):
            lines.append(f"- {rec}")
        
        lines.append("\n## Mental Checklist:")
        lines.append("Before submitting an answer, verify:")
        
        for pattern in analysis.get("patterns", [])[:3]:
            cat = pattern.get("category", "")
            if "arithmetic" in cat:
                lines.append("- [ ] Double-check all arithmetic calculations")
            elif "step" in cat or "multi" in cat:
                lines.append("- [ ] Ensure all steps are accounted for")
            elif "sign" in cat:
                lines.append("- [ ] Verify positive/negative signs")
            elif "percent" in cat:
                lines.append("- [ ] Confirm percentage calculations are correct")
            elif "unit" in cat:
                lines.append("- [ ] Check unit conversions")
        
        return "\n".join(lines)
