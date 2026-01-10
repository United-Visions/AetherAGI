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

try:
    from supabase import create_client, Client
    from dotenv import load_dotenv
    load_dotenv(root_path / ".env")
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

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
        yolo_mode: bool = False,
        output_path: Optional[str] = None,
        variant_filter: Optional[str] = None,
        day_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        samples: int = 1
    ):
        self.loader = DatasetProgressionLoader(family)
        self.samples = samples
        
        # Apply variant filter if provided
        if variant_filter:
            self.loader.filter_variants(variant_filter)
            
        self.verbose = verbose
        self.yolo_mode = yolo_mode
        self.output_path = output_path
        self.day_id = day_id
        self.run_id = run_id
        
        # Initialize Supabase client
        self.supabase = None
        if SUPABASE_AVAILABLE:
            url = os.getenv("SUPABASE_URL") or os.getenv("SB_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SB_SECRET_KEY")
            
            if self.verbose:
                masked_key = key[:5] + "..." if key else "None"
                print(f"🔧 Supabase Config: URL={url}, Key={masked_key}", file=sys.stderr)

            if url and key:
                try:
                    self.supabase = create_client(url, key)
                    if self.verbose:
                        print(f"✅ Supabase connected for logging (Day: {day_id})")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Supabase connection failed: {e}")

        # Override chunk size if provided
        if chunk_size:
            self.loader.config.chunk_size = chunk_size
        
        # Initialize Aether client for answering questions
        self.aether = AetherBenchmarkClient(
            mode=mode,
            api_base=api_base,
            api_key=api_key,
            model=model or "gemini/gemini-2.5-pro"
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
                chunk_summary = await self._run_chunk()
                self.all_results.append(chunk_summary)
                
                # Record score and advance state
                self.loader.record_chunk_score(chunk_summary["score"])
                self.loader.advance_chunk()
                
                # Save checkpoint and show progress
                self._save_checkpoint(chunk_summary)
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
        
        mode_str = "YOLO BATCH MODE" if self.yolo_mode else "SEQUENTIAL MODE"
        self._log(f"\n{'='*60}")
        self._log(f"📝 {variant.name} - Chunk {chunk_num}/{total_chunks} ({mode_str})")
        self._log(f"   Questions: {len(chunk_data)}")
        self._log(f"{'='*60}")
        
        # Convert to Question objects
        questions = self._parse_questions(chunk_data, variant)
        
        answers = []
        correct = 0

        # Choose execution mode
        if self.yolo_mode:
            # YOLO BATCH MODE: Send all questions in one API call
            answers, correct = await self._run_batch_yolo(questions, variant)
        else:
            # Sequential execution - each question gets its own API call for accuracy
            answers, correct = await self._run_sequential(questions, variant)
        
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
    
    async def _run_sequential(self, questions: List[Question], variant) -> tuple:
        """Run questions sequentially (one API call per question)."""
        from collections import Counter
        
        answers = []
        correct = 0
        
        for i, question in enumerate(questions):
            # Use benchmark-specific formatting if available
            if self.benchmark:
                prompt = self.benchmark.format_question_for_aether(question)
            else:
                prompt = question.text
            
            if self.samples > 1:
                # MAJORITY VOTING (Self-Consistency)
                sample_answers = []
                extracted_values = []
                
                print(f"   Asking question {i+1} (Sampling {self.samples} times)...")
                
                for s in range(self.samples):
                    result = await self._ask_with_retry(prompt, variant.answer_format)
                    raw_ans = result.get("response", "")
                    _, extracted = self._check_answer(question, raw_ans, variant)
                    sample_answers.append(raw_ans)
                    extracted_values.append(extracted)
                    print(f"      Sample {s+1}/{self.samples}: {extracted}", end="\r")
                
                # Find most common answer
                c = Counter(extracted_values)
                most_common_answer, count = c.most_common(1)[0]
                
                # Use the raw response that produced the majority answer (just pick the first one that matches)
                idx = extracted_values.index(most_common_answer)
                model_answer = sample_answers[idx]
                extracted = most_common_answer
                
                # Recalculate correctness based on majority vote
                is_correct, _ = self._check_answer(question, model_answer, variant)
                
                self._log(f"\r      Majority Vote: {extracted} ({count}/{self.samples} votes)           ")
                
                # Synthesize a result object
                result = {
                    "response": model_answer,
                    "latency_ms": 0, # Aggregate latency not tracked easily here
                    "tokens_used": 0
                }
                
            else:
                # STANDARD SINGLE PASS
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
        
        return answers, correct
    
    async def _run_batch_yolo(self, questions: List[Question], variant) -> tuple:
        """Run all questions in a single batched API call (YOLO mode)."""
        answers = []
        correct = 0
        
        # Build batch prompt with all questions
        batch_prompt = f"Answer ALL {len(questions)} questions below. Provide ONLY the answers in order, one per line, numbered 1-{len(questions)}.\n\n"
        for i, question in enumerate(questions, 1):
            batch_prompt += f"Question {i}: {question.text}\n\n"
        
        batch_prompt += f"\nProvide exactly {len(questions)} answers in this format:\n1. [answer]\n2. [answer]\n...\n{len(questions)}. [answer]"
        
        # Single API call with all questions
        self._log(f"   🚀 Sending batch of {len(questions)} questions in ONE API call...")
        result = await self._ask_with_retry(batch_prompt, variant.answer_format, max_retries=5)
        
        # Parse the batch response
        raw_response = result.get("response", "")
        
        # Check for cold start or error messages
        if "Brain is still waking up" in raw_response or "wait" in raw_response.lower()[:100]:
            self._log(f"   ⏳ Cold start detected. Waiting 30 seconds and retrying...")
            await asyncio.sleep(30)
            result = await self._ask_with_retry(batch_prompt, variant.answer_format, max_retries=5)
            raw_response = result.get("response", "")
        
        # Log response to file for debugging
        log_path = Path(__file__).parent / "results" / "yolo_debug_log.txt"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Chunk with {len(questions)} questions\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write(raw_response)
            f.write(f"\n{'='*80}\n\n")
        
        # Log preview to console
        self._log(f"   📄 Response length: {len(raw_response)} chars")
        self._log(f"   📄 Response preview (first 300 chars):\n{raw_response[:300]}")
        self._log(f"   📄 Full response saved to: {log_path}")
        
        # Try multiple parsing strategies
        answer_lines = self._parse_batch_response(raw_response, len(questions), variant)
        
        self._log(f"   📝 Parsed {len(answer_lines)} answers from batch response")
        
        # Check each answer
        for i, question in enumerate(questions):
            if i < len(answer_lines):
                model_answer = answer_lines[i]
                is_correct, extracted = self._check_answer(question, model_answer, variant)
            else:
                # Missing answer - treat as incorrect
                self._log(f"   ⚠️ Missing answer for question {i+1}")
                is_correct = False
                extracted = "[MISSING]"
            
            if is_correct:
                correct += 1
            
            answer = Answer(
                question_id=question.id,
                raw_response=model_answer if i < len(answer_lines) else "[MISSING]",
                extracted_answer=extracted,
                is_correct=is_correct,
                latency_ms=result.get("latency_ms", 0) // len(questions),  # Divide by num questions
                tokens_used=result.get("tokens_used", 0) // len(questions),
            )
            answers.append(answer)
            
            status = "✓" if is_correct else "✗"
            self._log(f"   [{i+1}/{len(questions)}] {status} (id: {question.id})")
        
        return answers, correct
    
    def _parse_batch_response(self, response: str, expected_count: int, variant) -> List[str]:
        """Parse batch response with multiple fallback strategies."""
        answer_lines = []
        
        # Strategy 1: Look for numbered patterns (1. answer or 1) answer or 1: answer)
        for line in response.split('\n'):
            match = re.match(r'^\s*(\d+)[.:)\s]+(.+?)\s*$', line)
            if match:
                answer_lines.append(match.group(2).strip())
        
        if len(answer_lines) >= expected_count * 0.8:  # Got at least 80%
            return answer_lines[:expected_count]
        
        # Strategy 2: Extract all numbers from response (for number-format answers)
        if variant.answer_format == "number":
            self._log(f"   ⚠️ Strategy 1 failed ({len(answer_lines)} answers). Trying number extraction...")
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
            # Filter out question numbers (1-50) and keep only answer-like numbers
            potential_answers = [n for n in numbers if not (1 <= int(n.replace(',', '').replace('.', '0')) <= expected_count)]
            if len(potential_answers) >= expected_count:
                return potential_answers[:expected_count]
        
        # Strategy 3: Look for "#### NUMBER" pattern (GSM8K format)
        if variant.answer_format == "number":
            self._log(f"   ⚠️ Strategy 2 failed. Trying #### pattern...")
            gsm_answers = re.findall(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', response)
            if len(gsm_answers) >= expected_count * 0.5:
                return gsm_answers[:expected_count]
        
        # Strategy 4: Split by question markers and extract last number from each section
        if variant.answer_format == "number":
            self._log(f"   ⚠️ Strategy 3 failed. Trying section-based extraction...")
            sections = re.split(r'Question \d+:', response)
            for section in sections[1:]:  # Skip first empty section
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', section)
                if numbers:
                    # Take the last number in each section (likely the answer)
                    answer_lines.append(numbers[-1])
            
            if len(answer_lines) >= expected_count * 0.5:
                return answer_lines[:expected_count]
        
        self._log(f"   ❌ All parsing strategies failed. Found {len(answer_lines)} answers, expected {expected_count}")
        return answer_lines

    async def _ask_with_retry(self, prompt: str, answer_format: str, max_retries: int = 3) -> Dict:
        """Helper to call Aether with retry logic."""
        retry_count = 0
        result = None
        
        # Determine timeout based on batch size
        # YOLO batches with 50+ questions need MUCH longer timeouts
        if "Question 1:" in prompt and "Question 50:" in prompt:
            # Full 50-question batch - give it 15 minutes!
            timeout = 900.0
            self._log(f"   ⏱️ Using extended timeout: {timeout}s (15 min) for large batch")
        elif "Question" in prompt:
            # Smaller batch or single question with context
            timeout = 300.0
        else:
            # Single question
            timeout = 300.0
        
        while retry_count <= max_retries:
            result = await self.aether.ask(
                question=prompt,
                answer_format=answer_format,
                timeout=timeout
            )
            
            raw = str(result.get("raw_response", ""))
            
            # Check for various error conditions
            if "429" in raw or "RateLimitError" in raw or "quota" in raw.lower():
                retry_count += 1
                wait_time = 30 * retry_count
                self._log(f"   ⚠️ Rate limit hit. Waiting {wait_time}s... (Attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            elif "waking up" in raw.lower() or "wait" in raw.lower()[:200]:
                retry_count += 1
                wait_time = 30
                self._log(f"   ⏳ Cold start or timeout detected. Waiting {wait_time}s... (Attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            elif "timeout" in raw.lower() or "timed out" in raw.lower():
                retry_count += 1
                # Increase timeout for next attempt
                timeout = timeout * 1.5
                self._log(f"   ⏱️ Timeout detected. Increasing to {timeout}s and retrying... (Attempt {retry_count}/{max_retries})")
                await asyncio.sleep(5)
            else:
                break
        return result

    def _report_chunk_to_supabase(self, chunk_data: Dict):
        """Report chunk results to Supabase."""
        if not self.supabase:
            return

        try:
            # We use a broad table 'benchmark_logs' or 'benchmark_live'
            # Assuming 'benchmark_live' for real-time dashboard
            payload = {
                "run_id": self.run_id or f"local_{int(datetime.now().timestamp())}",
                "day_id": self.day_id or "day_1",
                "family": self.loader.family,
                "variant": chunk_data.get("variant"),
                "chunk_index": chunk_data.get("chunk"),
                "total_chunks": chunk_data.get("total_chunks"),
                "score": chunk_data.get("score"),
                "correct_count": chunk_data.get("correct"),
                "total_questions": chunk_data.get("total"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "yolo_mode": self.yolo_mode,
                    "model": self.aether.model
                }
            }
            # Upsert into table if exists
            self.supabase.table("benchmark_results").insert(payload).execute()
        except Exception as e:
            # Log failure to stderr so it appears in parent process logs
            print(f"❌ Supabase reporting failed: {e}", file=sys.stderr)
            # Silent fail to avoid crashing runner
            pass

    def _save_final_results_to_supabase(self, results: Dict) -> bool:
        """Save final benchmark results to Supabase. Returns True if successful."""
        if not self.supabase:
            return False

        try:
            # Compute summary stats for the final results table
            scores_by_variant = results.get("scores_by_variant", {})
            completed_variants = results.get("completed_variants", [])
            
            payload = {
                "run_id": self.run_id or f"local_{int(datetime.now().timestamp())}",
                "day_id": self.day_id or "day_1",
                "family": results.get("family", self.loader.family),
                "status": "completed",
                "overall_average": results.get("overall_average", 0),
                "completed_variants": completed_variants,
                "scores_by_variant": scores_by_variant,
                "total_questions": sum(
                    v.get("total_questions", 0) for v in scores_by_variant.values()
                ),
                "total_correct": sum(
                    v.get("total_correct", 0) for v in scores_by_variant.values()
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "yolo_mode": self.yolo_mode,
                    "model": self.aether.model,
                    "chunk_size": self.loader.config.chunk_size,
                    "all_chunk_results": results.get("all_chunk_results", [])
                }
            }
            
            # Insert final results
            self.supabase.table("benchmark_runs").upsert(
                payload, 
                on_conflict="run_id"
            ).execute()
            
            self._log(f"✅ Final results saved to Supabase (run_id: {payload['run_id']})")
            return True
            
        except Exception as e:
            print(f"❌ Supabase final save failed: {e}", file=sys.stderr)
            return False

    def _save_checkpoint(self, last_chunk: Dict):
        """Save checkpoint after every chunk - primarily to Supabase."""
        # Report to Supabase (primary storage)
        self._report_chunk_to_supabase(last_chunk)
        
        # Only save JSON locally if Supabase is not available (dev mode)
        if not self.supabase:
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            if self.output_path:
                results_path = Path(self.output_path)
                results_path.parent.mkdir(exist_ok=True)
            else:
                filename = f"{self.loader.family}_yolo_progressive.json" if self.yolo_mode else f"{self.loader.family}_progressive.json"
                results_path = results_dir / filename
            
            report = self.loader.get_final_report()
            report["last_chunk"] = last_chunk
            report["all_chunk_results"] = self.all_results
            report["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            with open(results_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self._log(f"   💾 Checkpoint saved to: {results_path}")
        else:
            self._log(f"   💾 Checkpoint saved to Supabase")

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
        "--output", "-o",
        type=str,
        help="Path to save the output JSON file. Overrides default naming.",
    )
    
    parser.add_argument(
        "--family-file",
        type=str,
        help="Path to a file containing a list of families to run in sequence.",
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
        "--samples",
        type=int,
        default=1,
        help="Number of samples per question for Self-Consistency (Majority Voting). Default 1.",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="YOLO mode: Send all questions in chunk as single batched API call (faster but less reliable)",
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        help="Filter to run only a specific variant (e.g., 'gsm8k' or 'hard')",
    )

    parser.add_argument(
        "--day-id",
        type=str,
        help="Identifier for the day/run (e.g. 'day_1')",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Unique ID for this run session",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for benchmarking (e.g. gemini/gemini-2.0-flash-exp)",
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
    
    runner = ProgressiveRunner(
        family=args.family,
        mode=args.mode,
        api_base=args.api_base,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        samples=args.samples,
        verbose=not args.quiet,
        yolo_mode=args.yolo,
        output_path=args.output,
        variant_filter=args.variant,
        day_id=args.day_id,
        run_id=args.run_id,
        model=args.model
    )
    
    try:
        results = asyncio.run(runner.run(auto_continue=args.auto))
        
        # Save to Supabase first (primary storage)
        saved_to_supabase = runner._save_final_results_to_supabase(results)
        
        # Fallback to JSON only if Supabase fails or for local dev
        if not saved_to_supabase:
            if args.output:
                results_path = Path(args.output)
            else:
                results_path = Path(__file__).parent / "results" / f"{args.family}_progressive.json"
            
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Final results saved to JSON fallback: {results_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running {args.family}: {e}")
        raise


if __name__ == "__main__":
    # This is the main entry point
    
    # We need to re-parse args here because the outer main() call consumes them
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-file", type=str)
    parser.add_argument("--family", "-f", type=str)
    
    # This is a bit of a hack to get the family file without fully re-defining all args
    known_args, remaining_argv = parser.parse_known_args()

    if known_args.family_file:
        print(f"AUTOMATION MODE: Running families from {known_args.family_file}")
        with open(known_args.family_file, 'r') as f:
            families_to_run = [line.strip() for line in f if line.strip()]
        
        base_args = [arg for arg in sys.argv[1:] if not arg.startswith('--family-file') and not arg.startswith('--family')]

        for i, family_name in enumerate(families_to_run):
            print("\n" + "="*80)
            print(f"🚀 Starting family {i+1}/{len(families_to_run)}: {family_name.upper()}")
            print("="*80)
            
            # Construct new output path for each family
            output_filename = f"benchmarks/results/{family_name}_progressive.json"
            
            # Build the command for this specific run
            run_args = sys.argv[0:1] + [
                '--family', family_name,
                '--output', output_filename,
                '--auto' # Always auto-continue in automation mode
            ] + base_args
            
            print(f"   Running with command: python {' '.join(run_args)}")
            
            # We need to set sys.argv for the main() function to parse correctly
            original_argv = sys.argv
            sys.argv = run_args
            main() # Call the main function for this family
            sys.argv = original_argv # Restore original argv
            
            print(f"\n✅ Finished family: {family_name.upper()}")
        
        print("\n🎉 All families complete!")

    else:
        # Standard single-family run
        main()

