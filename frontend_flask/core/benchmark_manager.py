import subprocess
import sys
import os
import time
import logging
from typing import List, Dict, Optional, Any
import signal

# Configure logging
logger = logging.getLogger(__name__)

class BenchmarkManager:
    """
    Manages the execution of concurrent benchmark processes.
    Communicates via Supabase for status updates.
    """
    
    def __init__(self):
        # Maps run_id -> subprocess.Popen object
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
        # Hardcoded definitions of "All" for supported families
        self.FAMILIES = {
            "gsm": {
                "variants": ["GSM-8K", "GSM-Hard", "GSM-Symbolic", "GSM-Plus"],
                "next_day_threshold": 0.0 # Just completion needed? Or specific score?
            }
        }

    def start_benchmark(self, day_id: str, family: str, variants: List[str], model: str = None) -> Dict[str, Any]:
        """
        Start benchmark processes for the given day and variants.
        Runs variants concurrently.
        """
        # Resolve 'all' to actual variant names
        target_variants = []
        family_def = self.FAMILIES.get(family.lower(), self.FAMILIES["gsm"])
        
        if "all" in [v.lower() for v in variants]:
            target_variants = family_def["variants"]
        else:
            target_variants = variants

        started_runs = []

        for variant in target_variants:
            run_id = f"{day_id}_{variant.replace(' ', '_')}_{int(time.time())}"
            
            # Construct command
            # We use sys.executable to ensure we use the same python env
            cmd = [
                sys.executable, "-m", "benchmarks.progressive_runner",
                "--family", family,
                "--variant", variant,
                "--day-id", day_id,
                "--run-id", run_id,
                "--auto",     # Don't prompt user
                "--quiet"     # Less stdout spam, rely on Supabase logs
            ]
            
            if model:
                cmd.extend(["--model", model])
            
            logger.info(f"🚀 Launching benchmark: {' '.join(cmd)}")
            
            # Verify environment before launching
            env_debug = {k: "SET" for k in ["SB_URL", "SB_SECRET_KEY", "SUPABASE_URL", "SUPABASE_KEY"] if os.getenv(k)}
            logger.info(f"   Environment check: {env_debug}")

            try:
                # Determine workspace root (2 levels up from frontend_flask/core/benchmark_manager.py)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                # If we are in frontend_flask/core, going up 2 levels gets us to repo root
                workspace_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
                
                # Launch independent process
                # We don't use shell=True for security and better signal handling
                # Inherit stdout/stderr for debugging visibility in console
                proc = subprocess.Popen(
                    cmd,
                    cwd=workspace_root, # Run from workspace root so 'benchmarks' module is found
                    # stdout=subprocess.PIPE, 
                    # stderr=subprocess.PIPE
                )
                
                self.active_processes[run_id] = proc
                started_runs.append({
                    "variant": variant,
                    "run_id": run_id,
                    "pid": proc.pid
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to start {variant}: {e}")
                
        return {
            "status": "started",
            "day_id": day_id,
            "processes": started_runs
        }

    def stop_benchmark(self, day_id: str = None, run_id: str = None) -> List[str]:
        """
        Stop running benchmarks.
        Can stop a specific run_id or all runs for a day_id.
        """
        stopped = []
        
        # Identify which processes to kill
        to_kill = []
        for rid, proc in self.active_processes.items():
            if run_id and rid == run_id:
                to_kill.append(rid)
            elif day_id and rid.startswith(f"{day_id}_"):
                to_kill.append(rid)
        
        # Kill them
        for rid in to_kill:
            proc = self.active_processes[rid]
            if proc.poll() is None: # If still running
                try:
                    proc.terminate() # Try graceful SIGTERM
                    # Give it a tiny bit to cleanup
                    # (In a real async system we'd await wait())
                    time.sleep(0.1)
                    if proc.poll() is None:
                        proc.kill() # Force kill
                    logger.info(f"🛑 Stopped benchmark process: {rid}")
                    stopped.append(rid)
                except Exception as e:
                    logger.error(f"Failed to stop {rid}: {e}")
            
            del self.active_processes[rid]
            
        return stopped

    def get_active_status(self) -> Dict[str, Any]:
        """Check status of all tracked processes."""
        # Clean up finished processes
        finished = []
        for rid, proc in self.active_processes.items():
            if proc.poll() is not None:
                finished.append(rid)
        
        for rid in finished:
            del self.active_processes[rid]
            
        return {
            "active_count": len(self.active_processes),
            "active_runs": list(self.active_processes.keys())
        }
