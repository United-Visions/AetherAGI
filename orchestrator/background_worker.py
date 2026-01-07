"""
Path: orchestrator/background_worker.py
Role: Background task processor that works independently of user sessions

This worker:
1. Continuously polls for pending goals
2. Spawns autonomous agents to work on them
3. Runs independently even if user closes browser
4. Resumes work after server restarts
5. Handles multiple goals concurrently
"""

import asyncio
from typing import List
from loguru import logger
from datetime import datetime
import signal
import sys

from orchestrator.goal_tracker import GoalTracker, Goal, TaskStatus
from orchestrator.autonomous_agent import AutonomousAgent
from brain.logic_engine import LogicEngine
from heart.heart_orchestrator import Heart
from mind.vector_store import AetherVectorStore
from mind.episodic_memory import EpisodicMemory
from orchestrator.action_parser import ActionParser
from orchestrator.router import Router


class BackgroundWorker:
    """
    Background worker that continuously processes goals.
    Ensures task completion even when user is offline.
    """
    
    def __init__(
        self,
        brain: LogicEngine,
        heart: Heart,
        store: AetherVectorStore,
        memory: EpisodicMemory,
        action_parser: ActionParser,
        router: Router,
        poll_interval: int = 30  # Poll every 30 seconds
    ):
        self.brain = brain
        self.heart = heart
        self.store = store
        self.memory = memory
        self.action_parser = action_parser
        self.router = router
        self.poll_interval = poll_interval
        
        self.goal_tracker = GoalTracker()
        self.autonomous_agent = AutonomousAgent(
            brain=brain,
            heart=heart,
            store=store,
            memory=memory,
            action_parser=action_parser,
            goal_tracker=self.goal_tracker,
            router=router
        )
        
        self.running = False
        self.active_tasks = {}  # goal_id -> asyncio.Task
        self._shutdown_event = asyncio.Event()
        
        # Only setup signal handlers if running standalone (not under uvicorn)
        # Uvicorn handles signals itself
        if not self._is_running_under_uvicorn():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("BackgroundWorker initialized")
    
    def _is_running_under_uvicorn(self) -> bool:
        """Check if running under uvicorn to avoid signal handler conflicts."""
        import sys
        return 'uvicorn' in sys.modules
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        # Force exit after second signal
        if hasattr(self, '_received_signal'):
            logger.warning("Second signal received, forcing exit...")
            sys.exit(1)
        self._received_signal = True
    
    async def start(self):
        """Start the background worker loop."""
        logger.info("🚀 BackgroundWorker starting...")
        self.running = True
        
        while self.running:
            try:
                await self._work_cycle()
            except asyncio.CancelledError:
                logger.info("BackgroundWorker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in work cycle: {e}", exc_info=True)
            
            # Use shorter sleep intervals for faster shutdown response
            for _ in range(self.poll_interval):
                if not self.running:
                    break
                await asyncio.sleep(1)
        
        # Cleanup on shutdown
        await self._cleanup()
        logger.info("BackgroundWorker stopped")
    
    def stop(self):
        """Stop the background worker."""
        logger.info("Stopping BackgroundWorker...")
        self.running = False
    
    async def _work_cycle(self):
        """Single work cycle: poll for goals and spawn workers."""
        # Get pending goals from database
        pending_goals = await self.goal_tracker.get_pending_goals()
        
        if not pending_goals:
            logger.debug("No pending goals at this time")
            return
        
        logger.info(f"📋 Found {len(pending_goals)} pending goals")
        
        for goal in pending_goals:
            # Skip if already being processed
            if goal.goal_id in self.active_tasks:
                task = self.active_tasks[goal.goal_id]
                if not task.done():
                    logger.debug(f"Goal {goal.goal_id} already being processed")
                    continue
                else:
                    # Task finished, remove from active
                    del self.active_tasks[goal.goal_id]
            
            # Check if goal needs subtasks
            if not goal.subtasks:
                logger.info(f"🧩 Goal {goal.goal_id} needs decomposition")
                subtasks = await self.autonomous_agent.decompose_goal_into_subtasks(goal)
                
                if subtasks:
                    await self.goal_tracker.add_subtasks(goal.goal_id, subtasks)
                    # Refresh goal with new subtasks
                    goal = await self.goal_tracker.get_goal(goal.goal_id)
                else:
                    logger.error(f"Failed to decompose goal {goal.goal_id}")
                    await self.goal_tracker.mark_goal_failed(goal.goal_id, "Could not decompose into subtasks")
                    continue
            
            # Spawn autonomous agent to work on goal
            logger.info(f"🤖 Spawning autonomous agent for goal: {goal.description[:50]}...")
            task = asyncio.create_task(self._work_on_goal(goal))
            self.active_tasks[goal.goal_id] = task
    
    async def _work_on_goal(self, goal: Goal):
        """Worker task that processes a single goal."""
        try:
            success = await self.autonomous_agent.work_on_goal(goal)
            
            if success:
                logger.success(f"✅ Goal completed: {goal.description}")
            else:
                logger.error(f"❌ Goal failed: {goal.description}")
        
        except Exception as e:
            logger.error(f"Error working on goal {goal.goal_id}: {e}", exc_info=True)
            await self.goal_tracker.mark_goal_failed(goal.goal_id, str(e))
    
    async def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("Cleaning up active tasks...")
        
        # Cancel all active tasks
        for goal_id, task in self.active_tasks.items():
            if not task.done():
                logger.info(f"Cancelling task for goal {goal_id}")
                task.cancel()
        
        # Wait for all tasks to finish
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        logger.info("Cleanup complete")
    
    async def submit_goal(
        self,
        user_id: str,
        description: str,
        priority: int = 5,
        metadata: dict = None
    ) -> str:
        """
        Submit a new goal for background processing.
        
        Args:
            user_id: User identifier
            description: Goal description
            priority: Priority level (1-10)
            metadata: Additional context
        
        Returns:
            Goal ID
        """
        goal = await self.goal_tracker.create_goal(
            user_id=user_id,
            description=description,
            priority=priority,
            metadata=metadata
        )
        
        logger.info(f"📨 Goal submitted: {goal.goal_id}")
        return goal.goal_id
    
    def get_goal_status(self, goal_id: str) -> dict:
        """
        Get current status of a goal.
        
        Args:
            goal_id: Goal identifier
        
        Returns:
            Status dict with progress and state
        """
        # This will be called from the API endpoint
        # Returns synchronous status without awaiting
        pass


# Global worker instance (initialized in main_api.py startup)
background_worker = None


def get_background_worker() -> BackgroundWorker:
    """Get the global background worker instance."""
    global background_worker
    if background_worker is None:
        raise RuntimeError("BackgroundWorker not initialized")
    return background_worker


def set_background_worker(worker: BackgroundWorker):
    """Set the global background worker instance."""
    global background_worker
    background_worker = worker
    logger.info("Global BackgroundWorker instance set")


async def run_background_worker_standalone():
    """
    Run the background worker as a standalone process.
    
    Usage:
        python -m orchestrator.background_worker
    """
    from brain.logic_engine import LogicEngine
    from heart.heart_orchestrator import Heart
    from mind.vector_store import AetherVectorStore
    from mind.episodic_memory import EpisodicMemory
    from orchestrator.action_parser import ActionParser
    from orchestrator.router import Router
    
    logger.info("Initializing standalone background worker...")
    
    # Initialize components
    brain = LogicEngine()
    store = AetherVectorStore()
    memory = EpisodicMemory(store)
    heart = Heart()
    router = Router()
    # ActionParser with all dependencies enables action execution
    action_parser = ActionParser(router=router, store=store, memory=memory)
    
    # Create and start worker
    worker = BackgroundWorker(
        brain=brain,
        heart=heart,
        store=store,
        memory=memory,
        action_parser=action_parser,
        router=router
    )
    
    await worker.start()


if __name__ == "__main__":
    # Run as standalone process
    asyncio.run(run_background_worker_standalone())
