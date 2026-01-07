"""
Path: orchestrator/goal_tracker.py
Role: Persistent goal and subtask tracking system for autonomous task completion

This system enables AetherMind to:
1. Store complex goals with subtasks
2. Track execution status at each step
3. Resume unfinished work after restarts
4. Ensure task completion even with errors/interruptions
"""

import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from loguru import logger
from orchestrator.supabase_client import SupabaseClient

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    BLOCKED = "blocked"

class SubTask:
    """Represents a single actionable subtask"""
    
    def __init__(
        self,
        subtask_id: str,
        goal_id: str,
        description: str,
        action_type: str,
        action_params: Dict[str, Any],
        dependencies: List[str] = None,
        status: TaskStatus = TaskStatus.PENDING,
        attempt_count: int = 0,
        max_attempts: int = 3,
        execution_result: Optional[Dict] = None,
        error_message: Optional[str] = None
    ):
        self.subtask_id = subtask_id
        self.goal_id = goal_id
        self.description = description
        self.action_type = action_type
        self.action_params = action_params
        self.dependencies = dependencies or []
        self.status = status
        self.attempt_count = attempt_count
        self.max_attempts = max_attempts
        self.execution_result = execution_result
        self.error_message = error_message
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "subtask_id": self.subtask_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "action_type": self.action_type,
            "action_params": self.action_params,
            "dependencies": self.dependencies,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "execution_result": self.execution_result,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubTask':
        """Create SubTask from dictionary"""
        status = TaskStatus(data['status']) if isinstance(data['status'], str) else data['status']
        return cls(
            subtask_id=data['subtask_id'],
            goal_id=data['goal_id'],
            description=data['description'],
            action_type=data['action_type'],
            action_params=data['action_params'],
            dependencies=data.get('dependencies', []),
            status=status,
            attempt_count=data.get('attempt_count', 0),
            max_attempts=data.get('max_attempts', 3),
            execution_result=data.get('execution_result'),
            error_message=data.get('error_message')
        )

class Goal:
    """Represents a high-level user goal with subtasks"""
    
    def __init__(
        self,
        goal_id: str,
        user_id: str,
        description: str,
        subtasks: List[SubTask] = None,
        status: TaskStatus = TaskStatus.PENDING,
        priority: int = 5,
        metadata: Optional[Dict] = None
    ):
        self.goal_id = goal_id
        self.user_id = user_id
        self.description = description
        self.subtasks = subtasks or []
        self.status = status
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "goal_id": self.goal_id,
            "user_id": self.user_id,
            "description": self.description,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        """Create Goal from dictionary"""
        status = TaskStatus(data['status']) if isinstance(data['status'], str) else data['status']
        subtasks = [SubTask.from_dict(st) for st in data.get('subtasks', [])]
        return cls(
            goal_id=data['goal_id'],
            user_id=data['user_id'],
            description=data['description'],
            subtasks=subtasks,
            status=status,
            priority=data.get('priority', 5),
            metadata=data.get('metadata', {})
        )
    
    def get_next_actionable_subtasks(self) -> List[SubTask]:
        """Get subtasks ready to execute (pending, no unmet dependencies)"""
        completed_ids = {st.subtask_id for st in self.subtasks if st.status == TaskStatus.COMPLETED}
        
        actionable = []
        for subtask in self.subtasks:
            if subtask.status in [TaskStatus.PENDING, TaskStatus.RETRYING]:
                # Check if all dependencies are completed
                if all(dep_id in completed_ids for dep_id in subtask.dependencies):
                    actionable.append(subtask)
        
        return actionable
    
    def get_progress(self) -> Dict[str, Any]:
        """Calculate goal progress"""
        total = len(self.subtasks)
        if total == 0:
            return {"completed": 0, "total": 0, "percentage": 0}
        
        completed = sum(1 for st in self.subtasks if st.status == TaskStatus.COMPLETED)
        failed = sum(1 for st in self.subtasks if st.status == TaskStatus.FAILED)
        
        return {
            "completed": completed,
            "failed": failed,
            "total": total,
            "percentage": (completed / total) * 100
        }

class GoalTracker:
    """
    Manages persistent goal and subtask tracking in Supabase.
    Enables autonomous task completion with error recovery.
    """
    
    def __init__(self):
        self.supabase = SupabaseClient().client
        logger.info("GoalTracker initialized with Supabase backend")
    
    async def create_goal(
        self,
        user_id: str,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> Goal:
        """
        Create a new goal for a user.
        
        Args:
            user_id: User identifier
            description: High-level goal description
            priority: Priority level (1-10, higher = more urgent)
            metadata: Additional context (domain, complexity, etc.)
        
        Returns:
            Created Goal object
        """
        goal_id = str(uuid.uuid4())
        goal = Goal(
            goal_id=goal_id,
            user_id=user_id,
            description=description,
            priority=priority,
            metadata=metadata
        )
        
        try:
            # Store in Supabase
            response = self.supabase.table('goals').insert(goal.to_dict()).execute()
            logger.success(f"Created goal {goal_id} for user {user_id}: {description[:50]}...")
            return goal
        except Exception as e:
            logger.error(f"Failed to create goal: {e}")
            raise
    
    async def add_subtasks(self, goal_id: str, subtasks: List[SubTask]) -> None:
        """
        Add subtasks to an existing goal.
        
        Args:
            goal_id: Goal identifier
            subtasks: List of SubTask objects to add
        """
        try:
            # Get current goal
            goal = await self.get_goal(goal_id)
            goal.subtasks.extend(subtasks)
            goal.updated_at = datetime.now().isoformat()
            
            # Update in Supabase
            response = self.supabase.table('goals').update(
                {"subtasks": [st.to_dict() for st in goal.subtasks], "updated_at": goal.updated_at}
            ).eq('goal_id', goal_id).execute()
            
            logger.info(f"Added {len(subtasks)} subtasks to goal {goal_id}")
        except Exception as e:
            logger.error(f"Failed to add subtasks: {e}")
            raise
    
    async def update_subtask_status(
        self,
        goal_id: str,
        subtask_id: str,
        status: TaskStatus,
        execution_result: Optional[Dict] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update the status and results of a subtask.
        
        Args:
            goal_id: Goal identifier
            subtask_id: Subtask identifier
            status: New status
            execution_result: Result data from execution
            error_message: Error message if failed
        """
        try:
            goal = await self.get_goal(goal_id)
            
            for subtask in goal.subtasks:
                if subtask.subtask_id == subtask_id:
                    subtask.status = status
                    subtask.updated_at = datetime.now().isoformat()
                    
                    if execution_result:
                        subtask.execution_result = execution_result
                    if error_message:
                        subtask.error_message = error_message
                    if status == TaskStatus.RETRYING:
                        subtask.attempt_count += 1
                    
                    break
            
            # Update goal status based on subtask statuses
            goal.status = self._calculate_goal_status(goal)
            goal.updated_at = datetime.now().isoformat()
            
            # Save to Supabase
            response = self.supabase.table('goals').update(
                {
                    "subtasks": [st.to_dict() for st in goal.subtasks],
                    "status": goal.status.value,
                    "updated_at": goal.updated_at
                }
            ).eq('goal_id', goal_id).execute()
            
            logger.debug(f"Updated subtask {subtask_id} status to {status.value}")
        except Exception as e:
            logger.error(f"Failed to update subtask status: {e}")
            raise
    
    async def get_goal(self, goal_id: str) -> Goal:
        """
        Retrieve a goal by ID.
        
        Args:
            goal_id: Goal identifier
        
        Returns:
            Goal object
        """
        try:
            response = self.supabase.table('goals').select('*').eq('goal_id', goal_id).execute()
            
            if not response.data:
                raise ValueError(f"Goal {goal_id} not found")
            
            return Goal.from_dict(response.data[0])
        except Exception as e:
            logger.error(f"Failed to get goal {goal_id}: {e}")
            raise
    
    async def get_pending_goals(self, user_id: Optional[str] = None) -> List[Goal]:
        """
        Get all goals that need processing (pending, in_progress, retrying).
        
        Args:
            user_id: Filter by user (optional)
        
        Returns:
            List of Goal objects
        """
        try:
            query = self.supabase.table('goals').select('*').in_(
                'status',
                [TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value, TaskStatus.RETRYING.value]
            )
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            response = query.order('priority', desc=True).execute()
            
            goals = [Goal.from_dict(data) for data in response.data]
            logger.info(f"Found {len(goals)} pending goals")
            return goals
        except Exception as e:
            logger.error(f"Failed to get pending goals: {e}")
            return []

    async def get_user_goals(
        self,
        user_id: str,
        include_completed: bool = True,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return recent goals for a user with computed progress metadata."""
        try:
            query = self.supabase.table('goals').select('*').eq('user_id', user_id)

            if not include_completed:
                query = query.in_(
                    'status',
                    [TaskStatus.PENDING.value, TaskStatus.IN_PROGRESS.value, TaskStatus.RETRYING.value]
                )

            query = query.order('updated_at', desc=True)
            if limit:
                query = query.limit(limit)

            response = query.execute()

            goals_summary: List[Dict[str, Any]] = []
            for row in response.data:
                goal = Goal.from_dict(row)
                progress = goal.get_progress()
                goals_summary.append({
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "status": goal.status.value if isinstance(goal.status, TaskStatus) else goal.status,
                    "priority": goal.priority,
                    "progress": progress.get("percentage", 0),
                    "metadata": goal.metadata,
                    "updated_at": goal.updated_at,
                    "subtasks": [st.to_dict() for st in goal.subtasks]
                })

            logger.debug(f"Fetched {len(goals_summary)} goals for user {user_id}")
            return goals_summary
        except Exception as e:
            logger.error(f"Failed to get goals for user {user_id}: {e}")
            return []
    
    async def mark_goal_completed(self, goal_id: str) -> None:
        """Mark a goal as completed."""
        try:
            response = self.supabase.table('goals').update(
                {"status": TaskStatus.COMPLETED.value, "updated_at": datetime.now().isoformat()}
            ).eq('goal_id', goal_id).execute()
            
            logger.success(f"Goal {goal_id} marked as completed")
        except Exception as e:
            logger.error(f"Failed to mark goal completed: {e}")
            raise
    
    async def mark_goal_failed(self, goal_id: str, reason: str) -> None:
        """Mark a goal as failed with reason."""
        try:
            response = self.supabase.table('goals').update(
                {
                    "status": TaskStatus.FAILED.value,
                    "updated_at": datetime.now().isoformat(),
                    "metadata": {"failure_reason": reason}
                }
            ).eq('goal_id', goal_id).execute()
            
            logger.error(f"Goal {goal_id} marked as failed: {reason}")
        except Exception as e:
            logger.error(f"Failed to mark goal as failed: {e}")
            raise
    
    def _calculate_goal_status(self, goal: Goal) -> TaskStatus:
        """Calculate goal status based on subtask statuses"""
        if not goal.subtasks:
            return TaskStatus.PENDING
        
        statuses = [st.status for st in goal.subtasks]
        
        # All completed = goal completed
        if all(s == TaskStatus.COMPLETED for s in statuses):
            return TaskStatus.COMPLETED
        
        # Any failed beyond max attempts = goal failed
        failed_subtasks = [st for st in goal.subtasks if st.status == TaskStatus.FAILED and st.attempt_count >= st.max_attempts]
        if failed_subtasks:
            return TaskStatus.FAILED
        
        # Any in progress or retrying = goal in progress
        if any(s in [TaskStatus.IN_PROGRESS, TaskStatus.RETRYING] for s in statuses):
            return TaskStatus.IN_PROGRESS
        
        return TaskStatus.PENDING
