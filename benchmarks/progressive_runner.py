"""
Progressive Benchmark Runner

Runs through REAL variant datasets progressively instead of generating fake questions.

Features:
- Chunked execution (don't run all 1319 at once)
- Automatic progression to harder variants
- User prompts after minimum datasets
- Score tracking across variants

Usage:
    # Run GSM family progressively
    python -m benchmarks.progressive_runner --family gsm
    
    # Run with custom chunk size
    python -m benchmarks.progressive_runner --family gsm --chunk-size 50
    
    # Show available families
    python -m benchmarks.progressive_runner --list
"""
import argparse
import asyncio
import json
import re
import sys
import os
from pathlib import Path

# Add project root to sys.path to support both 'python -m benchmarks.progressive_runner' 
# and 'python benchmarks/progressive_runner.py'
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from benchmarks.dataset_progression import (
    DatasetProgressionLoader,
    DatasetVariant,
    get_available_families,
    get_family_info,
)
from benchmarks.aether_client import AetherBenchmarkClient
from benchmarks.score_tracker import ScoreTracker
from benchmarks.base import Question, Answer, RoundResult

# Import benchmark implementations
from benchmarks.implementations.gsm8k import GSM8KBenchmark
from benchmarks.implementations.mmlu import MMLUBenchmark
from benchmarks.implementations.humaneval import HumanEvalBenchmark

def get_benchmark_impl(family: str):
    """Factory to get the benchmark implementation for a family."""
    if family == "gsm":
        return GSM8KBenchmark()
    elif family == "mmlu":
        return MMLUBenchmark()
    elif family == "humaneval":
        return HumanEvalBenchmark()
    return None

class ProgressiveRunner:
    """
    Runner that progressively tests through real variant datasets.
    
    Instead of:
    - Round 1: GSM-8K (100 questions)
    - Round 2: LLM-generated fake questions
    - Round 3: More fake questions
    
    We now do:
    - Chunk 1: GSM-8K (questions 1-100)
    - Chunk 2: GSM-8K (questions 101-200)
    - ...
    - Chunk 14: GSM-8K (questions 1201-1319) -> COMPLETE
    - Chunk 1: GSM-Hard (questions 1-100)
    - ...
    
    All REAL benchmark data, no fake generation.
    """
    
    def __init__(
        self,
        family: str = "gsm",
        mode: str = "litellm",
        api_base: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
    ):
        self.loader = DatasetProgressionLoader(family)
        self.verbose = verbose
        
        # Override chunk size if provided
        if chunk_size:
            self.loader.config.chunk_size = chunk_size
        
        # Initialize Aether client for answering questions
        self.aether = AetherBenchmarkClient(
            mode=mode,
            api_base=api_base,
            api_key=api_key,
        )
        
        # Get benchmark implementation for scoring
        self.benchmark = get_benchmark_impl(family)
        
        self.score_tracker = ScoreTracker()
        self.all_results: List[Dict] = []
    
    async def run(self, auto_continue: bool = False) -> Dict[str, Any]:
        """
        Run the progressive benchmark.
        
        Args:
            auto_continue: If True, don't prompt to continue after minimum datasets
            
        Returns:
            Final report with all scores
        """
        self._log(self.loader.get_progress_summary())
        
        while True:
            # Load current dataset
            success = await self.loader.load_current_dataset()
            if not success:
                self._log("❌ Failed to load dataset, skipping to next...")
                if not self.loader.advance_to_next_variant():
                    break
                continue
            
            # Process all chunks of current dataset
            while self.loader.has_more_chunks:
                chunk_result = await self._run_chunk()
                self.all_results.append(chunk_result)
                
                # Record score
                self.loader.record_chunk_score(chunk_result["score"])
                self.loader.advance_chunk()
                
                # Save checkpoint after each chunk (score now included)
                self._save_checkpoint(chunk_result)
                
                # Show progress
                self._log(self.loader.get_progress_summary())
            
            # --- VERSION COMPLETE ---
            variant_name = self.loader.current_variant.name
            self._log(f"\n✅ {variant_name} COMPLETE!")
            self._show_variant_summary()
            
            # Mark version as complete so it shows up in 'completed_variants' in the JSON
            self.loader.complete_current_variant()
            
            # Save FULL results for this version before continuing or prompting
            self._log(f"   💾 Version {variant_name} finished. Saving full results...")
            self._save_checkpoint({"variant": variant_name, "status": "completed"})
            
            # Check if there are more variants in the entire family
            if not self.loader.has_more_variants:
                self._log(f"\n🎉 All {self.loader.family.upper()} variants complete!")
                break
                
            # Check if we should prompt to continue (only after minimum datasets)
            if self.loader.completed_minimum and not auto_continue:
                if not self._prompt_continue():
                    self._log("\n👋 Stopping as requested.")
                    break
            
            # Move to next variant index and loop back to load it
            self.loader.config.current_variant_index += 1
            self._log(f"\n🔄 Advancing to: {self.loader.current_variant.name}")
        
        # Generate and return final report
        return self._generate_final_report()
    
    async def _run_chunk(self) -> Dict[str, Any]:
        """Run a single chunk of questions."""
        variant = self.loader.current_variant
        chunk_data, chunk_num, total_chunks = self.loader.get_next_chunk()
        
        # Note: Batch mode is DISABLED - causes parsing issues with 0% scores
        # All benchmarks now run in sequential mode for accuracy
        mode_str = "SEQUENTIAL MODE"
        self._log(f"\n{'='*60}")
        self._log(f"📝 {variant.name} - Chunk {chunk_num}/{total_chunks} ({mode_str})")
        self._log(f"   Questions: {len(chunk_data)}")
        self._log(f"{'='*60}")
        
        # Convert to Question objects
        questions = self._parse_questions(chunk_data, variant)
        
        answers = []
        correct = 0

        # Sequential execution - each question gets its own API call for accuracy
        # (Batch mode was disabled because LLMs don't return clean numbered lists)
        for i, question in enumerate(questions):
            # Use benchmark-specific formatting if available
            if self.benchmark:
                prompt = self.benchmark.format_question_for_aether(question)
            else:
                prompt = question.text
                
            result = await self._ask_with_retry(prompt, variant.answer_format)
            
            model_answer = result.get("response", "")
            is_correct, extracted = self._check_answer(question, model_answer, variant)
            
            if is_correct:
                correct += 1
            
            answer = Answer(
                question_id=question.id,
                raw_response=model_answer,
                extracted_answer=extracted,
                is_correct=is_correct,
                latency_ms=result.get("latency_ms", 0),
                tokens_used=result.get("tokens_used", 0),
            )
            answers.append(answer)
            status = "✓" if is_correct else "✗"
            self._log(f"   [{i+1}/{len(questions)}] {status} (id: {question.id})")
            
            if not is_correct and self.verbose:
                # Print error details for failed questions
                err = extracted.get("error") if isinstance(extracted, dict) else None
                if err:
                    # Find the actual error message in the traceback
                    err_str = str(err)
                    if "ASSERTION_FAILED" in err_str:
                        clean_err = err_str.split("ASSERTION_FAILED:")[1].strip()
                    elif "ERROR:" in err_str:
                        clean_err = err_str.split("ERROR:")[1].strip()
                    else:
                        clean_err = err_str.strip()
                        
                    # Only show the last few lines of traceback if too long
                    if len(clean_err.split("\n")) > 5:
                        clean_err = "..." + "\n".join(clean_err.split("\n")[-5:])
                        
                    self._log(f"      ❌ Error: {clean_err[:500]}")

        print()
        score = correct / len(questions) if questions else 0
        self._log(f"\n📊 Chunk {chunk_num} Results:")
        self._log(f"   Score: {score * 100:.1f}% ({correct}/{len(questions)})")
        
        chunk_summary = {
            "variant": variant.name,
            "chunk": chunk_num,
            "total_chunks": total_chunks,
            "score": score,
            "correct": correct,
            "total": len(questions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
            
        return chunk_summary

    async def _ask_with_retry(self, prompt: str, answer_format: str, max_retries: int = 3) -> Dict:
        """Helper to call Aether with retry logic."""
        retry_count = 0
        result = None
        
        while retry_count <= max_retries:
            result = await self.aether.ask(
                question=prompt,
                answer_format=answer_format,
                timeout=300.0 if "Question" in prompt else 60.0
            )
            
            raw = str(result.get("raw_response", ""))
            if "429" in raw or "RateLimitError" in raw or "quota" in raw.lower():
                retry_count += 1
                wait_time = 30 * retry_count
                self._log(f"   ⚠️ Rate limit hit. Waiting {wait_time}s... (Attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                break
        return result

    def _save_checkpoint(self, last_chunk: Dict):
        """Save results to JSON file after every chunk."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / f"{self.loader.family}_progressive.json"
        
        report = self.loader.get_final_report()
        report["last_chunk"] = last_chunk
        report["all_chunk_results"] = self.all_results
        report["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        self._log(f"   💾 Checkpoint saved to: {results_path}")

    def _parse_questions(self, data: List[Dict], variant: DatasetVariant) -> List[Question]:
        """Parse raw data into Question objects."""
        questions = []
        
        for i, item in enumerate(data):
            try:
                q_text = item.get(variant.question_field, "")
                raw_answer = item.get(variant.answer_field, "")
                
                # For MMLU-style datasets, append choices to the question text
                if variant.answer_format == "letter" and "choices" in item:
                    choices = item["choices"]
                    if isinstance(choices, list):
                        for j, choice in enumerate(choices):
                            letter = chr(65 + j)  # A, B, C, D...
                            q_text += f"\n{letter}) {choice}"
                
                # Parse answer based on format
                expected = self._extract_expected_answer(raw_answer, variant)
                
                # Use all fields as metadata (critical for code tests like HumanEval)
                metadata = item.copy()
                metadata["variant"] = variant.name
                metadata["raw_answer"] = raw_answer
                
                questions.append(Question(
                    id=f"{variant.name}_{i}",
                    text=q_text,
                    expected_answer=expected,
                    metadata=metadata
                ))
            except Exception as e:
                self._log(f"   ⚠️ Failed to parse question {i}: {e}")
                continue
        
        return questions
    
    def _extract_expected_answer(self, raw_answer: str, variant: DatasetVariant) -> str:
        """Extract the expected answer from raw answer text."""
        raw_answer = str(raw_answer)
        
        if variant.parser == "gsm8k_answer":
            # GSM-8K format: "...\n#### 42"
            match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', raw_answer)
            if match:
                return match.group(1).replace(",", "")
            # Fallback: last number
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', raw_answer)
            return numbers[-1].replace(",", "") if numbers else raw_answer
        
        elif variant.parser == "gsm_symbolic_answer":
            # Similar to GSM-8K but may have different format
            match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', raw_answer)
            if match:
                return match.group(1).replace(",", "")
            return raw_answer.strip()
        
        elif variant.parser == "mmlu_index_to_letter":
            # MMLU answer is an index (0, 1, 2, 3) -> convert to letter (A, B, C, D)
            try:
                idx = int(raw_answer)
                return chr(65 + idx)  # 0->A, 1->B, 2->C, 3->D
            except (ValueError, TypeError):
                # If it's already a letter, just return it
                if raw_answer.upper() in "ABCDEFGHIJ":
                    return raw_answer.upper()
                return raw_answer
        
        elif variant.answer_format == "number":
            # Generic number extraction
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', raw_answer)
            return numbers[-1].replace(",", "") if numbers else raw_answer
        
        elif variant.answer_format == "letter":
            # Extract letter answer
            letters = re.findall(r'[A-Ja-j]', raw_answer)
            return letters[0].upper() if letters else raw_answer
        
        else:
            return raw_answer.strip()
    
    def _check_answer(self, question: Question, response: str, variant: DatasetVariant) -> tuple:
        """Check if the response is correct."""
        # Use full benchmark implementation if available
        if self.benchmark:
            return self.benchmark.check_answer(question, response)
            
        expected = str(question.expected_answer).strip()
        
        # Extract answer from response
        if variant.answer_format == "number":
            # Extract numbers from response
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
            extracted = numbers[-1].replace(",", "") if numbers else response.strip()
            
            # Compare numerically
            try:
                exp_num = float(expected.replace(",", ""))
                act_num = float(extracted.replace(",", ""))
                is_correct = abs(exp_num - act_num) < 0.001
            except:
                is_correct = expected == extracted
                
        elif variant.answer_format == "letter":
            letters = re.findall(r'[A-Ja-j]', response)
            extracted = letters[-1].upper() if letters else response.strip()
            is_correct = extracted.upper() == expected.upper()
            
        else:
            extracted = response.strip()
            is_correct = extracted.lower() == expected.lower()
        
        return is_correct, extracted
    
    def _show_variant_summary(self):
        """Show summary for completed variant."""
        variant_name = self.loader.current_variant.name
        scores = self.loader.config.scores_by_variant.get(variant_name, [])
        
        if scores:
            avg = sum(scores) / len(scores)
            self._log(f"\n📈 {variant_name} Summary:")
            self._log(f"   Chunks completed: {len(scores)}")
            self._log(f"   Average score: {avg * 100:.1f}%")
            self._log(f"   Score range: {min(scores)*100:.1f}% - {max(scores)*100:.1f}%")
    
    def _prompt_continue(self) -> bool:
        """Prompt user whether to continue to next dataset."""
        print("\n" + "="*60)
        print(f"✅ Completed {len(self.loader.config.completed_variants)} datasets!")
        print("="*60)
        
        if self.loader.has_more_variants:
            next_variant = self.loader.config.variants[self.loader.config.current_variant_index + 1]
            print(f"\n🔮 Next dataset: {next_variant.name}")
            print(f"   {next_variant.description}")
            print(f"   Difficulty: {'⭐' * next_variant.difficulty_rank}")
        
        while True:
            response = input("\n➡️  Continue to next dataset? [Y/n/q]: ").strip().lower()
            if response in ("", "y", "yes"):
                return True
            elif response in ("n", "no", "q", "quit"):
                return False
            else:
                print("   Please enter Y (yes), N (no), or Q (quit)")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        report = self.loader.get_final_report()
        report["all_chunk_results"] = self.all_results
        report["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        self._log("\n" + "="*60)
        self._log("🏁 FINAL REPORT")
        self._log("="*60)
        self._log(f"\n📊 Family: {report['family'].upper()}")
        self._log(f"📚 Variants completed: {len(report['completed_variants'])}")
        
        for variant_name, data in report["scores_by_variant"].items():
            self._log(f"\n   {variant_name}:")
            self._log(f"      Chunks: {data['chunks_completed']}")
            self._log(f"      Average: {data['average'] * 100:.1f}%")
        
        self._log(f"\n🎯 Overall Average: {report['overall_average'] * 100:.1f}%")
        self._log("="*60 + "\n")
        
        return report
    
    def _log(self, msg: str, end: str = "\n"):
        """Log message if verbose mode."""
        if self.verbose:
            print(msg, end=end, flush=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Progressive Benchmark Runner - Real variant datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GSM family progressively
  python -m benchmarks.progressive_runner --family gsm
  
  # Run with smaller chunks
  python -m benchmarks.progressive_runner --family gsm --chunk-size 50
  
  # Auto-continue without prompts
  python -m benchmarks.progressive_runner --family gsm --auto
  
  # List available families
  python -m benchmarks.progressive_runner --list

Dataset Families:
  gsm       Grade school math (GSM-8K → GSM-Hard → GSM-Symbolic → GSM-Plus)
  mmlu      Knowledge (MMLU → MMLU-Pro → MMLU-Redux)
  humaneval Coding (HumanEval → HumanEval+ → MBPP → MBPP+)
        """
    )
    
    parser.add_argument(
        "--ladder", "--family", "-f",
        dest="family",
        type=str,
        help="Dataset ladder to run (gsm/math, mmlu/knowledge, humaneval/coding)",
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        help="Questions per chunk (default varies by family)",
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-continue without prompting after minimum datasets",
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["litellm", "api"],
        default="litellm",
        help="How to call Aether: litellm (direct) or api (HTTP)",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Aether API key (for api mode)",
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000",
        help="Aether API base URL (for api mode)",
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available dataset families",
    )
    
    parser.add_argument(
        "--info", "-i",
        type=str,
        help="Show info about a specific family",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list:
        print("\n📚 Available Dataset Families:")
        print("-" * 50)
        for family in get_available_families():
            info = get_family_info(family)
            print(f"\n  {family.upper()}")
            print(f"    Variants: {info['num_variants']}")
            for v in info['variants']:
                print(f"      {'⭐' * v['difficulty_rank']} {v['name']}: {v['description'][:50]}...")
        print()
        return
    
    if args.info:
        info = get_family_info(args.info.lower())
        if info:
            print(f"\n📊 {args.info.upper()} Family")
            print("=" * 50)
            print(f"Chunk size: {info['chunk_size']} questions")
            print(f"Min before prompt: {info['min_before_prompt']} datasets")
            print(f"\nVariants ({info['num_variants']}):")
            for v in info['variants']:
                print(f"\n  {'⭐' * v['difficulty_rank']} {v['name']}")
                print(f"    {v['description']}")
                print(f"    ~{v['estimated_size']:,} questions")
        else:
            print(f"Unknown family: {args.info}")
        return
    
    if not args.family:
        parser.print_help()
        return
    
    # Handle aliases
    family = args.family.lower()
    if family == "math":
        family = "gsm"
    elif family == "knowledge":
        family = "mmlu"
    elif family == "coding":
        family = "humaneval"
    
    # Run progressive benchmark
    runner = ProgressiveRunner(
        family=family,
        mode=args.mode,
        api_base=args.api_base,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )
    
    try:
        results = asyncio.run(runner.run(auto_continue=args.auto))
        
        # Save results
        results_path = Path(__file__).parent / "results" / f"{args.family}_progressive.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
