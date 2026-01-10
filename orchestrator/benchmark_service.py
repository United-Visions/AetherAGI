"""
Path: orchestrator/benchmark_service.py
Role: Manages asynchronous execution of progressive benchmarks across multiple models.
"""

import asyncio
import json
import os
import uuid
import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
from benchmarks.progressive_runner import ProgressiveRunner
from benchmarks.dataset_progression import get_available_families, get_family_info

class BenchmarkService:
    def __init__(self, store=None, memory=None, background_tasks_dict=None):
        self.store = store
        self.memory = memory
        self.active_benchmarks: Dict[str, Dict] = {}
        self.results_cache: Dict[str, Any] = {}
        self.background_tasks = background_tasks_dict  # Reference to BACKGROUND_TASKS
        
        # Load existing results from disk
        self._load_cached_results()

    def _load_cached_results(self):
        """Load benchmark results from the benchmarks/results directory."""
        results_dir = "benchmarks/results"
        if not os.path.exists(results_dir):
            return
            
        for filename in os.listdir(results_dir):
            if filename.endswith("_progressive.json"):
                family = filename.replace("_progressive.json", "")
                try:
                    with open(os.path.join(results_dir, filename), "r") as f:
                        self.results_cache[family] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load result for {family}: {e}")

    async def get_available_benchmarks(self) -> List[Dict]:
        """Return list of available benchmark families with details."""
        families = get_available_families()
        details = []
        for f in families:
            info = get_family_info(f)
            details.append({
                "id": f,
                "name": f.upper(),
                "description": f"Benchmark suite for {f}",
                "variants": info["variants"],
                "last_results": self.results_cache.get(f)
            })
        return details

    async def start_benchmark(self, family: str, user_id: str, mode: str = "api", model: str = None) -> str:
        """Start a benchmark run as a background task."""
        benchmark_id = str(uuid.uuid4())
        
        self.active_benchmarks[benchmark_id] = {
            "id": benchmark_id,
            "family": family,
            "user_id": user_id,
            "model": model or "aethermind-v1",
            "status": "running",
            "start_time": datetime.datetime.now().isoformat(),
            "progress": 0,
            "current_chunk": 0,
            "total_chunks": 0,
            "score": 0.0,
            "results": []
        }
        
        # Start background task
        asyncio.create_task(self._run_benchmark_task(benchmark_id, family, user_id, mode, model))
        
        return benchmark_id

    async def _run_benchmark_task(self, benchmark_id: str, family: str, user_id: str, mode: str, model: str):
        """Internal task to execute the benchmark."""
        try:
            logger.info(f"Starting benchmark task {benchmark_id} for {family} (model: {model})")
            
            # Initialize the runner
            runner = ProgressiveRunner(
                family=family,
                mode=mode,
                verbose=False
            )
            
            # If using a specific model override
            if model:
                runner.aether.model = model
            
            # Override run logic to update our local state
            total_chunks = 0
            completed_chunks = 0
            
            # We wrap the existing runner's inner loops
            # Note: This is a bit of a hack since ProgressiveRunner isn't designed as a library
            # but it works for now without rewriting the whole runner logic.
            
            while True:
                success = await runner.loader.load_current_dataset()
                if not success:
                    if not runner.loader.advance_to_next_variant(): break
                    continue
                
                while runner.loader.has_more_chunks:
                    chunk_result = await runner._run_chunk()
                    runner.all_results.append(chunk_result)
                    
                    # Update local state
                    completed_chunks += 1
                    runner.loader.record_chunk_score(chunk_result["score"])
                    
                    progress_data = {
                        "progress": (completed_chunks / 100) * 100, # Approximation
                        "current_chunk": completed_chunks,
                        "score": sum(runner.loader.config.scores_by_variant.get(runner.loader.current_variant.name, [0])) / 
                                 max(1, len(runner.loader.config.scores_by_variant.get(runner.loader.current_variant.name, []))),
                        "last_activity": chunk_result.get("activity_events", [])
                    }
                    self.active_benchmarks[benchmark_id].update(progress_data)
                    
                    # Sync to BACKGROUND_TASKS if available
                    if self.background_tasks and benchmark_id in self.background_tasks:
                        self.background_tasks[benchmark_id].update(progress_data)
                    
                    runner.loader.advance_chunk()
                    runner._save_checkpoint(chunk_result)
                
                if not runner.loader.has_more_variants: break
                runner.loader.config.current_variant_index += 1
            
            # Finalize
            report = runner._generate_final_report()
            self.results_cache[family] = report
            completion_data = {
                "status": "completed",
                "progress": 100,
                "final_report": report
            }
            self.active_benchmarks[benchmark_id].update(completion_data)
            
            # Sync to BACKGROUND_TASKS if available
            if self.background_tasks and benchmark_id in self.background_tasks:
                self.background_tasks[benchmark_id].update(completion_data)
            
            logger.info(f"Benchmark task {benchmark_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Benchmark task {benchmark_id} failed: {e}")
            error_data = {
                "status": "failed",
                "error": str(e)
            }
            self.active_benchmarks[benchmark_id].update(error_data)
            
            # Sync to BACKGROUND_TASKS if available
            if self.background_tasks and benchmark_id in self.background_tasks:
                self.background_tasks[benchmark_id].update(error_data)

    async def get_status(self, benchmark_id: str) -> Dict:
        """Get the status of an active or recent benchmark."""
        return self.active_benchmarks.get(benchmark_id, {"error": "Benchmark not found"})

BENCHMARK_SERVICE = BenchmarkService()
