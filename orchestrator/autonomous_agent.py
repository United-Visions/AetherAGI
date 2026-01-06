"""
Path: orchestrator/autonomous_agent.py
Role: Self-healing autonomous agent that continuously works until goals are completed

This agent:
1. Analyzes execution failures and generates fix strategies
2. Breaks complex tasks into smaller subtasks dynamically
3. Retries failed operations with learned corrections
4. Continues working independently without user intervention
5. Resumes unfinished work after restarts
"""

import asyncio
from typing import List, Dict, Optional, Any
from loguru import logger
from datetime import datetime

from orchestrator.goal_tracker import GoalTracker, Goal, SubTask, TaskStatus
from orchestrator.active_inference import ActiveInferenceLoop
from orchestrator.action_parser import ActionParser, ActionTag
from brain.logic_engine import LogicEngine
from heart.heart_orchestrator import Heart
from mind.vector_store import AetherVectorStore
from mind.episodic_memory import EpisodicMemory


class AutonomousAgent:
    """
    AGI-level autonomous agent with self-healing and persistent goal completion.
    
    Core Loop:
    1. Get goal from GoalTracker
    2. Generate execution plan (Brain)
    3. Execute subtasks
    4. Analyze results (success/failure)
    5. If failure → diagnose and generate fix
    6. If success → move to next subtask
    7. Repeat until goal complete
    """
    
    def __init__(
        self,
        brain: LogicEngine,
        heart: Heart,
        store: AetherVectorStore,
        memory: EpisodicMemory,
        action_parser: ActionParser,
        goal_tracker: GoalTracker,
        router=None
    ):
        self.brain = brain
        self.heart = heart
        self.store = store
        self.memory = memory
        self.action_parser = action_parser
        self.goal_tracker = goal_tracker
        self.router = router
        
        # Use the action_executor from the action_parser if available
        if action_parser.action_executor:
            self.action_executor = action_parser.action_executor
        else:
            # Create our own if action_parser doesn't have one
            from orchestrator.action_parser import ActionExecutor
            from orchestrator.router import Router
            self.router = router or Router()
            self.action_executor = ActionExecutor(self.router, store, memory)
        
        logger.info("AutonomousAgent initialized with self-healing capabilities")
    
    async def work_on_goal(self, goal: Goal) -> bool:
        """
        Autonomously work on a goal until completion or failure.
        
        Args:
            goal: Goal object to work on
        
        Returns:
            True if goal completed, False if failed
        """
        logger.info(f"🎯 Starting autonomous work on goal: {goal.description}")
        
        # Update goal status to in_progress
        goal.status = TaskStatus.IN_PROGRESS
        
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get progress
            progress = goal.get_progress()
            logger.info(f"📊 Goal progress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)")
            
            # Check if goal is complete
            if progress['completed'] == progress['total']:
                await self.goal_tracker.mark_goal_completed(goal.goal_id)
                logger.success(f"✅ Goal completed: {goal.description}")
                return True
            
            # Get next actionable subtasks (dependencies met)
            actionable_subtasks = goal.get_next_actionable_subtasks()
            
            if not actionable_subtasks:
                # No more actionable subtasks - check if blocked
                if progress['failed'] > 0:
                    logger.error(f"❌ Goal blocked by {progress['failed']} failed subtasks")
                    await self.goal_tracker.mark_goal_failed(goal.goal_id, "Subtasks failed beyond max attempts")
                    return False
                else:
                    # All tasks complete
                    await self.goal_tracker.mark_goal_completed(goal.goal_id)
                    logger.success(f"✅ Goal completed: {goal.description}")
                    return True
            
            # Execute next subtask
            for subtask in actionable_subtasks[:1]:  # Process one at a time
                success = await self._execute_subtask_with_healing(goal, subtask)
                
                if not success and subtask.attempt_count >= subtask.max_attempts:
                    logger.error(f"❌ Subtask failed beyond max attempts: {subtask.description}")
                    await self.goal_tracker.update_subtask_status(
                        goal.goal_id,
                        subtask.subtask_id,
                        TaskStatus.FAILED,
                        error_message=subtask.error_message
                    )
                
                # Refresh goal data
                goal = await self.goal_tracker.get_goal(goal.goal_id)
        
        logger.warning(f"⚠️ Goal reached max iterations ({max_iterations})")
        await self.goal_tracker.mark_goal_failed(goal.goal_id, f"Exceeded max iterations ({max_iterations})")
        return False
    
    async def _execute_subtask_with_healing(self, goal: Goal, subtask: SubTask) -> bool:
        """
        Execute a subtask with self-healing retry logic.
        
        Args:
            goal: Parent goal
            subtask: Subtask to execute
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🔧 Executing subtask: {subtask.description}")
        
        # Mark as in progress
        await self.goal_tracker.update_subtask_status(
            goal.goal_id,
            subtask.subtask_id,
            TaskStatus.IN_PROGRESS
        )
        
        # Create ActionTag from subtask
        action_tag = ActionTag(
            tag_type=subtask.action_type,
            content=subtask.action_params.get("content", ""),
            attributes=subtask.action_params.get("attributes", {})
        )
        
        # Execute action using the action_executor
        result = await self.action_executor.execute(action_tag, goal.user_id)
        
        # Analyze result
        if result["success"]:
            logger.success(f"✅ Subtask completed: {subtask.description}")
            await self.goal_tracker.update_subtask_status(
                goal.goal_id,
                subtask.subtask_id,
                TaskStatus.COMPLETED,
                execution_result=result
            )
            return True
        else:
            # Failure - attempt self-healing
            logger.warning(f"⚠️ Subtask failed: {result.get('error', 'Unknown error')}")
            
            # Check if we should retry
            if subtask.attempt_count < subtask.max_attempts:
                # Generate fix strategy using Brain
                fix_strategy = await self._generate_fix_strategy(subtask, result)
                
                if fix_strategy:
                    logger.info(f"🔄 Attempting fix: {fix_strategy['description']}")
                    
                    # Update subtask with fix
                    subtask.action_params = fix_strategy["updated_params"]
                    await self.goal_tracker.update_subtask_status(
                        goal.goal_id,
                        subtask.subtask_id,
                        TaskStatus.RETRYING,
                        error_message=result.get("error")
                    )
                    
                    return False  # Will retry in next iteration
                else:
                    logger.error("❌ Could not generate fix strategy")
                    await self.goal_tracker.update_subtask_status(
                        goal.goal_id,
                        subtask.subtask_id,
                        TaskStatus.FAILED,
                        execution_result=result,
                        error_message=result.get("error")
                    )
                    return False
            else:
                # Max attempts reached
                logger.error(f"❌ Max attempts reached for subtask: {subtask.description}")
                await self.goal_tracker.update_subtask_status(
                    goal.goal_id,
                    subtask.subtask_id,
                    TaskStatus.FAILED,
                    execution_result=result,
                    error_message=result.get("error")
                )
                return False
    
    async def _generate_fix_strategy(self, subtask: SubTask, failure_result: Dict) -> Optional[Dict]:
        """
        Use Brain to analyze failure and generate fix strategy.
        
        Args:
            subtask: Failed subtask
            failure_result: Execution result with error details
        
        Returns:
            Fix strategy with updated parameters, or None if cannot fix
        """
        # Create diagnostic prompt for Brain
        diagnostic_prompt = f"""
TASK FAILURE ANALYSIS:

Original Task: {subtask.description}
Action Type: {subtask.action_type}
Attempt: {subtask.attempt_count + 1}/{subtask.max_attempts}

FAILURE DETAILS:
Error: {failure_result.get('error', 'Unknown error')}
Output: {failure_result.get('output', 'No output')}
Metadata: {failure_result.get('metadata', {})}

ORIGINAL PARAMETERS:
{subtask.action_params}

INSTRUCTIONS:
1. Analyze why this task failed
2. Identify the root cause
3. Propose a specific fix
4. Provide updated parameters that will work

Respond with a JSON object:
{{
    "diagnosis": "root cause of failure",
    "fix_description": "what needs to change",
    "updated_params": {{...updated parameters...}}
}}
"""
        
        try:
            # Get fix from Brain
            response = await self.brain.generate_thought(
                user_input=diagnostic_prompt,
                context_vector=None,
                emotion_vector={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                predicted_flourishing=0.5
            )
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                fix_strategy = json.loads(json_match.group())
                logger.info(f"🧠 Brain diagnosis: {fix_strategy.get('diagnosis', 'N/A')}")
                logger.info(f"🔧 Fix strategy: {fix_strategy.get('fix_description', 'N/A')}")
                return fix_strategy
            else:
                logger.warning("Brain did not return valid JSON fix strategy")
                return None
        
        except Exception as e:
            logger.error(f"Failed to generate fix strategy: {e}", exc_info=True)
            return None
    
    async def decompose_goal_into_subtasks(self, goal: Goal) -> List[SubTask]:
        """
        Use Brain to decompose a high-level goal into actionable subtasks.
        
        Args:
            goal: Goal to decompose
        
        Returns:
            List of SubTask objects
        """
        decomposition_prompt = f"""
GOAL DECOMPOSITION:

User Goal: {goal.description}
Priority: {goal.priority}/10
Domain: {goal.metadata.get('domain', 'general')}

INSTRUCTIONS:
Break this goal into small, actionable subtasks. Each subtask should:
1. Be independently executable
2. Have clear success criteria
3. Include specific action type (aether-write, aether-sandbox, aether-install, etc.)
4. Specify dependencies on other subtasks

Respond with a JSON array:
[
    {{
        "description": "Clear task description",
        "action_type": "aether-write|aether-sandbox|aether-install|aether-command",
        "action_params": {{
            "content": "file content or code",
            "attributes": {{"path": "file.py", "language": "python"}}
        }},
        "dependencies": []  // List of subtask descriptions this depends on
    }},
    ...
]

Example for "Create a Flask calculator app":
[
    {{
        "description": "Install Flask package",
        "action_type": "aether-install",
        "action_params": {{"packages": ["flask"], "attributes": {{}}}},
        "dependencies": []
    }},
    {{
        "description": "Create app.py with Flask routes",
        "action_type": "aether-write",
        "action_params": {{
            "content": "from flask import Flask...",
            "attributes": {{"path": "calculator/app.py"}}
        }},
        "dependencies": ["Install Flask package"]
    }},
    {{
        "description": "Test calculator endpoints",
        "action_type": "aether-sandbox",
        "action_params": {{
            "content": "import requests; response = requests.get('http://localhost:5000/add/5/3')...",
            "attributes": {{"language": "python", "test": "true"}}
        }},
        "dependencies": ["Create app.py with Flask routes"]
    }}
]
"""
        
        try:
            response = await self.brain.generate_thought(
                user_input=decomposition_prompt,
                context_vector=None,
                emotion_vector={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                predicted_flourishing=0.5
            )
            
            # Parse JSON response
            import json
            import re
            import uuid
            
            # Extract JSON array
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                subtask_specs = json.loads(json_match.group())
                
                # Convert to SubTask objects
                subtasks = []
                description_to_id = {}
                
                for spec in subtask_specs:
                    subtask_id = str(uuid.uuid4())
                    description = spec["description"]
                    description_to_id[description] = subtask_id
                    
                    subtask = SubTask(
                        subtask_id=subtask_id,
                        goal_id=goal.goal_id,
                        description=description,
                        action_type=spec["action_type"],
                        action_params=spec["action_params"],
                        dependencies=[]  # Will map after all created
                    )
                    subtasks.append(subtask)
                
                # Map dependency names to IDs
                for i, spec in enumerate(subtask_specs):
                    dep_names = spec.get("dependencies", [])
                    subtasks[i].dependencies = [description_to_id.get(name, "") for name in dep_names if name in description_to_id]
                
                logger.success(f"🧩 Decomposed goal into {len(subtasks)} subtasks")
                return subtasks
            else:
                logger.warning("Brain did not return valid JSON subtask array")
                return []
        
        except Exception as e:
            logger.error(f"Failed to decompose goal: {e}", exc_info=True)
            return []
