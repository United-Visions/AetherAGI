"""
Memory integration for benchmark learning.

Stores mistake patterns and learnings in Aether's episodic memory
so it can recall them during future benchmark attempts.
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Try to import Pinecone for memory storage
try:
    from mind.vector_store import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


class BenchmarkMemory:
    """
    Stores benchmark learnings in Aether's episodic memory.
    
    Uses a dedicated namespace 'benchmark_learnings' to store:
    - Mistake patterns and categories
    - Problem-solving strategies that worked
    - Weak areas identified across benchmarks
    """
    
    NAMESPACE = "benchmark_learnings"
    
    def __init__(self, vector_store: Optional[Any] = None):
        if vector_store:
            self.store = vector_store
        elif VECTOR_STORE_AVAILABLE:
            try:
                self.store = VectorStore()
            except Exception as e:
                print(f"Warning: Could not initialize VectorStore: {e}")
                self.store = None
        else:
            self.store = None
    
    def store_mistake_pattern(
        self,
        benchmark: str,
        pattern: Dict[str, Any],
        round_number: int,
    ) -> bool:
        """
        Store a mistake pattern in memory.
        
        Args:
            benchmark: Name of the benchmark
            pattern: Mistake pattern dict from analyzer
            round_number: Which round this came from
            
        Returns:
            True if stored successfully
        """
        if not self.store:
            return False
        
        # Create a text summary for embedding
        text = self._format_pattern_for_embedding(benchmark, pattern, round_number)
        
        metadata = {
            "type": "mistake_pattern",
            "benchmark": benchmark,
            "round": round_number,
            "category": pattern.get("category", "unknown"),
            "frequency": pattern.get("frequency", 1),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            self.store.upsert(
                texts=[text],
                metadatas=[metadata],
                namespace=self.NAMESPACE,
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to store mistake pattern: {e}")
            return False
    
    def store_learning_summary(
        self,
        benchmark: str,
        analysis: Dict[str, Any],
        final_score: float,
    ) -> bool:
        """
        Store a complete learning summary after benchmark completion.
        
        Args:
            benchmark: Name of the benchmark
            analysis: Full mistake analysis dict
            final_score: Final score achieved
            
        Returns:
            True if stored successfully
        """
        if not self.store:
            return False
        
        # Create comprehensive learning summary
        text = self._format_learning_summary(benchmark, analysis, final_score)
        
        metadata = {
            "type": "learning_summary",
            "benchmark": benchmark,
            "final_score": final_score,
            "weak_areas": analysis.get("weak_areas", [])[:5],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        try:
            self.store.upsert(
                texts=[text],
                metadatas=[metadata],
                namespace=self.NAMESPACE,
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to store learning summary: {e}")
            return False
    
    def recall_learnings(
        self,
        benchmark: str,
        query: str = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant learnings for a benchmark.
        
        Args:
            benchmark: Name of the benchmark
            query: Optional specific query (e.g., "percentage problems")
            top_k: Number of results to return
            
        Returns:
            List of relevant learning memories
        """
        if not self.store:
            return []
        
        search_query = query or f"learnings and mistakes from {benchmark} benchmark"
        
        try:
            results = self.store.query(
                query=search_query,
                namespace=self.NAMESPACE,
                top_k=top_k,
                filter={"benchmark": benchmark} if not query else None,
            )
            return results
        except Exception as e:
            print(f"Warning: Failed to recall learnings: {e}")
            return []
    
    def get_weak_areas(self, benchmark: str = None) -> List[str]:
        """
        Get accumulated weak areas from all benchmark runs.
        
        Args:
            benchmark: Optional specific benchmark, or all if None
            
        Returns:
            List of weak area strings, most frequent first
        """
        if not self.store:
            return []
        
        try:
            query = f"weak areas and common mistakes{' in ' + benchmark if benchmark else ''}"
            results = self.store.query(
                query=query,
                namespace=self.NAMESPACE,
                top_k=20,
            )
            
            # Extract and deduplicate weak areas
            weak_areas = []
            for r in results:
                if "weak_areas" in r.get("metadata", {}):
                    weak_areas.extend(r["metadata"]["weak_areas"])
            
            # Count frequency
            from collections import Counter
            counts = Counter(weak_areas)
            return [area for area, _ in counts.most_common(10)]
            
        except Exception as e:
            print(f"Warning: Failed to get weak areas: {e}")
            return []
    
    def _format_pattern_for_embedding(
        self,
        benchmark: str,
        pattern: Dict[str, Any],
        round_number: int,
    ) -> str:
        """Format a mistake pattern for vector embedding."""
        examples_text = ""
        for ex in pattern.get("examples", [])[:2]:
            examples_text += f"\nExample: {ex.get('question_text', '')[:200]}"
            examples_text += f"\nExpected: {ex.get('expected', '')}, Got: {ex.get('actual', '')}"
        
        return f"""Benchmark: {benchmark}
Round: {round_number}
Mistake Category: {pattern.get('category', 'unknown')}
Description: {pattern.get('description', 'Unknown error pattern')}
Frequency: {pattern.get('frequency', 1)} occurrences
Severity: {pattern.get('severity', 0):.2f}
{examples_text}

When solving similar problems in {benchmark}, avoid this mistake by:
- Carefully checking {pattern.get('category', 'the calculation')} steps
- Verifying the answer format matches expectations"""
    
    def _format_learning_summary(
        self,
        benchmark: str,
        analysis: Dict[str, Any],
        final_score: float,
    ) -> str:
        """Format a learning summary for vector embedding."""
        weak_str = ", ".join(analysis.get("weak_areas", [])[:5]) or "none identified"
        
        patterns_text = ""
        for p in analysis.get("patterns", [])[:3]:
            patterns_text += f"\n- {p.get('category', 'unknown')}: {p.get('description', '')} (frequency: {p.get('frequency', 0)})"
        
        recs_text = "\n".join(f"- {r}" for r in analysis.get("recommendations", [])[:5])
        
        return f"""Benchmark Learning Summary: {benchmark}
Final Score: {final_score * 100:.1f}%

Weak Areas Identified:
{weak_str}

Common Mistake Patterns:
{patterns_text}

Recommendations for Improvement:
{recs_text}

Key Learnings:
When taking {benchmark} benchmark questions:
1. Watch out for: {weak_str}
2. Double-check calculations before final answer
3. Make sure answer format matches expected format"""


class LocalBenchmarkMemory:
    """
    Fallback local storage for benchmark learnings when Pinecone unavailable.
    Stores in JSON files in the results directory.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        from pathlib import Path
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).parent / "results"
        self.memory_file = self.results_dir / "learnings.json"
        self._ensure_file()
    
    def _ensure_file(self):
        """Ensure the memory file exists."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            with open(self.memory_file, "w") as f:
                json.dump({"learnings": [], "weak_areas": {}}, f)
    
    def _load(self) -> Dict:
        with open(self.memory_file) as f:
            return json.load(f)
    
    def _save(self, data: Dict):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def store_learning(self, benchmark: str, content: str, metadata: Dict = None):
        """Store a learning locally."""
        data = self._load()
        data["learnings"].append({
            "benchmark": benchmark,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._save(data)
    
    def add_weak_areas(self, benchmark: str, areas: List[str]):
        """Add weak areas for a benchmark."""
        data = self._load()
        if benchmark not in data["weak_areas"]:
            data["weak_areas"][benchmark] = []
        data["weak_areas"][benchmark].extend(areas)
        # Deduplicate
        data["weak_areas"][benchmark] = list(set(data["weak_areas"][benchmark]))
        self._save(data)
    
    def get_weak_areas(self, benchmark: str = None) -> List[str]:
        """Get weak areas, optionally for a specific benchmark."""
        data = self._load()
        if benchmark:
            return data["weak_areas"].get(benchmark, [])
        # All weak areas
        all_areas = []
        for areas in data["weak_areas"].values():
            all_areas.extend(areas)
        return list(set(all_areas))
    
    def get_learnings(self, benchmark: str = None, limit: int = 10) -> List[Dict]:
        """Get stored learnings."""
        data = self._load()
        learnings = data["learnings"]
        if benchmark:
            learnings = [l for l in learnings if l["benchmark"] == benchmark]
        return learnings[-limit:]
