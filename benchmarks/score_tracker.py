"""
Score tracker and persistent storage for benchmark results.

Stores results in a directory structure:
    benchmarks/results/{benchmark_name}/
        ├── round_1/
        │   ├── questions.json
        │   ├── answers.json
        │   ├── score.json
        │   └── mistakes.json
        ├── round_2/
        │   └── ...
        └── summary.json
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from .base import RoundResult, Question, Answer


class ScoreTracker:
    """
    Tracks and persists benchmark scores across rounds.
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _benchmark_dir(self, benchmark_name: str) -> Path:
        """Get the directory for a specific benchmark."""
        safe_name = benchmark_name.lower().replace("-", "_").replace(" ", "_")
        path = self.results_dir / safe_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _round_dir(self, benchmark_name: str, round_number: int) -> Path:
        """Get the directory for a specific round."""
        path = self._benchmark_dir(benchmark_name) / f"round_{round_number}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_round(self, result: RoundResult) -> Path:
        """
        Save a round's results to disk.
        
        Returns the path where results were saved.
        """
        round_dir = self._round_dir(result.benchmark_name, result.round_number)
        
        # Save questions
        questions_data = [
            {
                "id": q.id,
                "text": q.text,
                "expected_answer": q.expected_answer,
                "difficulty": q.difficulty,
                "category": q.category,
                "metadata": q.metadata,
            }
            for q in result.questions
        ]
        with open(round_dir / "questions.json", "w") as f:
            json.dump(questions_data, f, indent=2, default=str)
        
        # Save answers
        answers_data = [
            {
                "question_id": a.question_id,
                "raw_response": a.raw_response,
                "extracted_answer": a.extracted_answer,
                "is_correct": a.is_correct,
                "reasoning": a.reasoning,
                "latency_ms": a.latency_ms,
                "tokens_used": a.tokens_used,
            }
            for a in result.answers
        ]
        with open(round_dir / "answers.json", "w") as f:
            json.dump(answers_data, f, indent=2, default=str)
        
        # Save score summary
        score_data = {
            "round_number": result.round_number,
            "benchmark_name": result.benchmark_name,
            "timestamp": result.timestamp,
            "score": result.score,
            "score_percent": f"{result.score * 100:.1f}%",
            "total_correct": result.total_correct,
            "total_questions": result.total_questions,
            "metadata": result.metadata,
        }
        with open(round_dir / "score.json", "w") as f:
            json.dump(score_data, f, indent=2)
        
        # Save mistakes for learning
        with open(round_dir / "mistakes.json", "w") as f:
            json.dump(result.mistakes, f, indent=2, default=str)
        
        # Update summary
        self._update_summary(result.benchmark_name)
        
        return round_dir
    
    def _update_summary(self, benchmark_name: str) -> None:
        """Update the overall summary file for a benchmark."""
        benchmark_dir = self._benchmark_dir(benchmark_name)
        summary_path = benchmark_dir / "summary.json"
        
        # Collect all rounds
        rounds = []
        for round_dir in sorted(benchmark_dir.glob("round_*")):
            score_file = round_dir / "score.json"
            if score_file.exists():
                with open(score_file) as f:
                    rounds.append(json.load(f))
        
        if not rounds:
            return
        
        # Calculate trends
        scores = [r["score"] for r in rounds]
        improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        
        summary = {
            "benchmark_name": benchmark_name,
            "total_rounds": len(rounds),
            "latest_score": scores[-1],
            "best_score": max(scores),
            "worst_score": min(scores),
            "improvement_from_start": improvement,
            "score_history": scores,
            "rounds": rounds,
            "last_updated": datetime.utcnow().isoformat(),
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def load_round(self, benchmark_name: str, round_number: int) -> Optional[RoundResult]:
        """Load a specific round's results."""
        round_dir = self._round_dir(benchmark_name, round_number)
        
        try:
            with open(round_dir / "questions.json") as f:
                questions_data = json.load(f)
            with open(round_dir / "answers.json") as f:
                answers_data = json.load(f)
            with open(round_dir / "score.json") as f:
                score_data = json.load(f)
            with open(round_dir / "mistakes.json") as f:
                mistakes = json.load(f)
        except FileNotFoundError:
            return None
        
        questions = [
            Question(
                id=q["id"],
                text=q["text"],
                expected_answer=q["expected_answer"],
                difficulty=q.get("difficulty", "medium"),
                category=q.get("category"),
                metadata=q.get("metadata", {}),
            )
            for q in questions_data
        ]
        
        answers = [
            Answer(
                question_id=a["question_id"],
                raw_response=a["raw_response"],
                extracted_answer=a["extracted_answer"],
                is_correct=a["is_correct"],
                reasoning=a.get("reasoning"),
                latency_ms=a.get("latency_ms", 0),
                tokens_used=a.get("tokens_used", 0),
            )
            for a in answers_data
        ]
        
        return RoundResult(
            round_number=score_data["round_number"],
            benchmark_name=score_data["benchmark_name"],
            timestamp=score_data["timestamp"],
            questions=questions,
            answers=answers,
            score=score_data["score"],
            total_correct=score_data["total_correct"],
            total_questions=score_data["total_questions"],
            mistakes=mistakes,
            metadata=score_data.get("metadata", {}),
        )
    
    def get_summary(self, benchmark_name: str) -> Optional[Dict[str, Any]]:
        """Get the summary for a benchmark."""
        summary_path = self._benchmark_dir(benchmark_name) / "summary.json"
        if not summary_path.exists():
            return None
        with open(summary_path) as f:
            return json.load(f)
    
    def get_all_mistakes(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """Get all mistakes across all rounds for a benchmark."""
        benchmark_dir = self._benchmark_dir(benchmark_name)
        all_mistakes = []
        
        for round_dir in sorted(benchmark_dir.glob("round_*")):
            mistakes_file = round_dir / "mistakes.json"
            if mistakes_file.exists():
                with open(mistakes_file) as f:
                    mistakes = json.load(f)
                    for m in mistakes:
                        m["round"] = round_dir.name
                    all_mistakes.extend(mistakes)
        
        return all_mistakes
    
    def get_latest_round_number(self, benchmark_name: str) -> int:
        """Get the latest round number for a benchmark (0 if none exist)."""
        benchmark_dir = self._benchmark_dir(benchmark_name)
        rounds = list(benchmark_dir.glob("round_*"))
        if not rounds:
            return 0
        return max(int(r.name.split("_")[1]) for r in rounds)
    
    def get_progress_report(self, benchmark_name: str) -> str:
        """Generate a human-readable progress report."""
        summary = self.get_summary(benchmark_name)
        if not summary:
            return f"No results for {benchmark_name} yet."
        
        lines = [
            f"📊 {benchmark_name} Progress Report",
            f"{'=' * 40}",
            f"Rounds completed: {summary['total_rounds']}",
            f"Latest score: {summary['latest_score'] * 100:.1f}%",
            f"Best score: {summary['best_score'] * 100:.1f}%",
            f"Improvement: {summary['improvement_from_start'] * 100:+.1f}%",
            "",
            "Score history:",
        ]
        
        for i, score in enumerate(summary['score_history'], 1):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"  Round {i}: {bar} {score * 100:.1f}%")
        
        return "\n".join(lines)


class LeaderboardTracker:
    """
    Tracks best scores across all benchmarks for leaderboard display.
    """
    
    def __init__(self, score_tracker: ScoreTracker):
        self.score_tracker = score_tracker
        self.leaderboard_path = score_tracker.results_dir / "leaderboard.json"
    
    def update(self, benchmark_name: str, score: float, round_number: int) -> None:
        """Update the leaderboard with a new score."""
        leaderboard = self._load()
        
        current = leaderboard.get(benchmark_name, {})
        if score > current.get("best_score", 0):
            current["best_score"] = score
            current["best_round"] = round_number
            current["achieved_at"] = datetime.utcnow().isoformat()
        
        current["latest_score"] = score
        current["latest_round"] = round_number
        current["updated_at"] = datetime.utcnow().isoformat()
        
        leaderboard[benchmark_name] = current
        self._save(leaderboard)
    
    def _load(self) -> Dict[str, Any]:
        if not self.leaderboard_path.exists():
            return {}
        with open(self.leaderboard_path) as f:
            return json.load(f)
    
    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.leaderboard_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_leaderboard(self) -> Dict[str, Any]:
        """Get the full leaderboard."""
        return self._load()
    
    def print_leaderboard(self) -> str:
        """Generate a printable leaderboard."""
        leaderboard = self._load()
        if not leaderboard:
            return "No benchmark results yet."
        
        lines = [
            "🏆 AetherMind Benchmark Leaderboard",
            "=" * 50,
            f"{'Benchmark':<20} {'Best':<10} {'Latest':<10} {'Trend':<10}",
            "-" * 50,
        ]
        
        for name, data in sorted(leaderboard.items()):
            best = f"{data.get('best_score', 0) * 100:.1f}%"
            latest = f"{data.get('latest_score', 0) * 100:.1f}%"
            diff = data.get('latest_score', 0) - data.get('best_score', 0)
            trend = "🔺" if diff > 0 else "🔻" if diff < 0 else "➖"
            lines.append(f"{name:<20} {best:<10} {latest:<10} {trend}")
        
        return "\n".join(lines)
