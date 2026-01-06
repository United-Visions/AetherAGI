"""
Path: orchestrator/action_parser.py
Role: Parse and execute AetherMind action tags from Brain responses
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from loguru import logger
from datetime import datetime

class ActionTag:
    """Represents a parsed action tag."""
    
    def __init__(self, tag_type: str, content: str, attributes: Dict[str, str]):
        self.tag_type = tag_type
        self.content = content
        self.attributes = attributes
        self.timestamp = datetime.now().isoformat()
    
    def to_activity_event(self, user_id: str) -> Dict:
        """Convert to activity event for frontend display."""
        event_id = f"{self.tag_type}_{self.timestamp}"
        
        # Map tag types to activity types (18 total activity types)
        activity_type_map = {
            "aether-write": "file_change",
            "aether-sandbox": "code_execution",
            "aether-forge": "tool_creation",
            "aether-install": "package_installation",
            "aether-research": "research",
            "aether-command": "ui_command",
            "aether-test": "test_execution",
            "aether-git": "git_operation",
            "aether-self-mod": "self_modification",
            "aether-plan": "planning",
            "aether-switch-domain": "domain_switch",
            "aether-memory-save": "memory_consolidation",
            "aether-solo-research": "autonomous_research",
            "aether-surprise": "surprise_detection",
            "aether-deploy": "deployment",
            "aether-heart": "emotional_processing",
            "aether-body-switch": "body_switch"
        }
        
        activity_type = activity_type_map.get(self.tag_type, "unknown")
        
        # Extract relevant data based on tag type
        data = self._extract_data()
        
        # Generate title
        title = self._generate_title()
        
        return {
            "id": event_id,
            "type": activity_type,
            "status": "in_progress",
            "title": title,
            "details": self.attributes.get("description", "Processing..."),
            "timestamp": self.timestamp,
            "data": data
        }
    
    def _extract_data(self) -> Dict:
        """Extract type-specific data from tag."""
        if self.tag_type == "aether-write":
            return {
                "files": [self.attributes.get("path", "unknown")],
                "code": self.content,
                "language": self.attributes.get("language", "python")
            }
        
        elif self.tag_type == "aether-sandbox":
            return {
                "code": self.content,
                "language": self.attributes.get("language", "python"),
                "test": self.attributes.get("test", "false") == "true"
            }
        
        elif self.tag_type == "aether-forge":
            return {
                "tool_name": self.attributes.get("tool_name", "unknown"),
                "description": self.attributes.get("description", ""),
                "code": self.content
            }
        
        elif self.tag_type == "aether-install":
            packages = self.content.strip().split()
            return {
                "packages": packages,
                "count": len(packages)
            }
        
        elif self.tag_type == "aether-research":
            return {
                "query": self.attributes.get("query", self.content),
                "namespace": self.attributes.get("namespace", "core_universal")
            }
        
        elif self.tag_type == "aether-command":
            return {
                "action": self.attributes.get("action", "unknown"),
                "target": self.attributes.get("target", "")
            }
        
        elif self.tag_type == "aether-test":
            return {
                "file": self.attributes.get("file", ""),
                "test_file": self.attributes.get("test_file", ""),
                "code": self.content
            }
        
        elif self.tag_type == "aether-git":
            return {
                "action": self.attributes.get("action", "commit"),
                "message": self.attributes.get("message", ""),
                "details": self.content
            }
        
        elif self.tag_type == "aether-self-mod":
            return {
                "file": self.attributes.get("file", ""),
                "patch": self.content
            }
        
        elif self.tag_type == "aether-plan":
            return {
                "deadline_days": self.attributes.get("deadline_days", "7"),
                "user_id": self.attributes.get("user_id", ""),
                "plan": self.content
            }
        
        elif self.tag_type == "aether-switch-domain":
            return {
                "domain": self.attributes.get("domain", "general"),
                "user_id": self.attributes.get("user_id", ""),
                "reason": self.content
            }
        
        elif self.tag_type == "aether-memory-save":
            return {
                "user_id": self.attributes.get("user_id", ""),
                "type": self.attributes.get("type", "knowledge_cartridge"),
                "content": self.content
            }
        
        elif self.tag_type == "aether-solo-research":
            return {
                "query": self.attributes.get("query", ""),
                "tools": self.attributes.get("tools", "browser").split(","),
                "priority": self.attributes.get("priority", "medium"),
                "goal": self.content
            }
        
        elif self.tag_type == "aether-surprise":
            return {
                "score": float(self.attributes.get("score", "0.0")),
                "concept": self.attributes.get("concept", ""),
                "info": self.content
            }
        
        elif self.tag_type == "aether-deploy":
            return {
                "target": self.attributes.get("target", "render"),
                "service": self.attributes.get("service", "backend"),
                "config": self.content
            }
        
        elif self.tag_type == "aether-heart":
            return {
                "user_id": self.attributes.get("user_id", ""),
                "emotion": self.attributes.get("emotion", "neutral"),
                "context": self.content
            }
        
        elif self.tag_type == "aether-body-switch":
            return {
                "adapter": self.attributes.get("adapter", "chat"),
                "reason": self.content
            }
        
        return {}
    
    def _generate_title(self) -> str:
        """Generate human-readable title."""
        if self.tag_type == "aether-write":
            path = self.attributes.get("path", "file")
            return f"Creating {path}"
        
        elif self.tag_type == "aether-sandbox":
            return "Executing code in sandbox"
        
        elif self.tag_type == "aether-forge":
            tool = self.attributes.get("tool_name", "tool")
            return f"Forging {tool}"
        
        elif self.tag_type == "aether-install":
            packages = self.content.strip().split()
            count = len(packages)
            return f"Installing {count} package{'s' if count != 1 else ''}"
        
        elif self.tag_type == "aether-research":
            namespace = self.attributes.get("namespace", "knowledge")
            return f"Researching {namespace}"
        
        elif self.tag_type == "aether-command":
            cmd = self.attributes.get("action", "unknown")
            return f"UI Command: {cmd}"
        
        elif self.tag_type == "aether-test":
            file = self.attributes.get("file", "code")
            return f"Testing {file}"
        
        elif self.tag_type == "aether-git":
            action = self.attributes.get("action", "commit")
            return f"Git {action}"
        
        elif self.tag_type == "aether-self-mod":
            file = self.attributes.get("file", "system")
            return f"Self-modifying {file}"
        
        elif self.tag_type == "aether-plan":
            days = self.attributes.get("deadline_days", "?")
            return f"Planning {days}-day project"
        
        elif self.tag_type == "aether-switch-domain":
            domain = self.attributes.get("domain", "general")
            return f"Switching to {domain} domain"
        
        elif self.tag_type == "aether-memory-save":
            mem_type = self.attributes.get("type", "knowledge")
            return f"Saving {mem_type}"
        
        elif self.tag_type == "aether-solo-research":
            query = self.attributes.get("query", "topic")[:30]
            return f"Autonomous research: {query}"
        
        elif self.tag_type == "aether-surprise":
            score = self.attributes.get("score", "0.0")
            concept = self.attributes.get("concept", "")
            return f"Surprise detected ({score}): {concept}"
        
        elif self.tag_type == "aether-deploy":
            target = self.attributes.get("target", "unknown")
            service = self.attributes.get("service", "app")
            return f"Deploying {service} to {target}"
        
        elif self.tag_type == "aether-heart":
            emotion = self.attributes.get("emotion", "neutral")
            return f"Emotional processing: {emotion}"
        
        elif self.tag_type == "aether-body-switch":
            adapter = self.attributes.get("adapter", "unknown")
            return f"Switching to {adapter} adapter"
        
        return "Processing action"


class ActionParser:
    """Parse AetherMind action tags from Brain responses."""
    
    # Tag patterns (17 action tag types + 2 special tags: think, summary)
    TAG_PATTERNS = {
        "aether-write": r'<aether-write\s+(.*?)>(.*?)</aether-write>',
        "aether-sandbox": r'<aether-sandbox\s+(.*?)>(.*?)</aether-sandbox>',
        "aether-forge": r'<aether-forge\s+(.*?)>(.*?)</aether-forge>',
        "aether-install": r'<aether-install>(.*?)</aether-install>',
        "aether-research": r'<aether-research\s+(.*?)>(.*?)</aether-research>',
        "aether-command": r'<aether-command\s+(.*?)>(.*?)</aether-command>',
        "aether-test": r'<aether-test\s+(.*?)>(.*?)</aether-test>',
        "aether-git": r'<aether-git\s+(.*?)>(.*?)</aether-git>',
        "aether-self-mod": r'<aether-self-mod\s+(.*?)>(.*?)</aether-self-mod>',
        "aether-plan": r'<aether-plan\s+(.*?)>(.*?)</aether-plan>',
        "aether-switch-domain": r'<aether-switch-domain\s+(.*?)>(.*?)</aether-switch-domain>',
        "aether-memory-save": r'<aether-memory-save\s+(.*?)>(.*?)</aether-memory-save>',
        "aether-solo-research": r'<aether-solo-research\s+(.*?)>(.*?)</aether-solo-research>',
        "aether-surprise": r'<aether-surprise\s+(.*?)>(.*?)</aether-surprise>',
        "aether-deploy": r'<aether-deploy\s+(.*?)>(.*?)</aether-deploy>',
        "aether-heart": r'<aether-heart\s+(.*?)>(.*?)</aether-heart>',
        "aether-body-switch": r'<aether-body-switch\s+(.*?)>(.*?)</aether-body-switch>',
        "think": r'<think>(.*?)</think>',  # Thinking process visualization
        "aether-chat-summary": r'<aether-chat-summary>(.*?)</aether-chat-summary>',
    }
    
    def __init__(self, router=None, store=None, memory=None):
        """
        Initialize ActionParser with optional dependencies.
        
        Args:
            router: Router instance for forwarding intents (optional)
            store: AetherVectorStore instance for knowledge retrieval (optional)
            memory: EpisodicMemory instance for memory operations (optional)
        
        If all three are provided, an ActionExecutor is created for executing actions.
        """
        self.router = router
        self.store = store
        self.memory = memory
        
        # Create ActionExecutor if all dependencies are provided
        if router is not None and store is not None and memory is not None:
            self.action_executor = ActionExecutor(router, store, memory)
            logger.info("ActionParser initialized with ActionExecutor")
        else:
            self.action_executor = None
            logger.info("ActionParser initialized (parsing only, no executor)")
    
    def parse(self, brain_response: str) -> Tuple[List[ActionTag], str]:
        """
        Parse all action tags from brain response.
        
        Returns:
            (action_tags, cleaned_response)
        """
        action_tags = []
        cleaned_response = brain_response
        
        for tag_type, pattern in self.TAG_PATTERNS.items():
            matches = re.finditer(pattern, brain_response, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                # Extract attributes and content
                if len(match.groups()) == 2:
                    attr_string = match.group(1)
                    content = match.group(2).strip()
                elif len(match.groups()) == 1:
                    attr_string = match.group(1) if tag_type != "aether-chat-summary" else ""
                    content = match.group(1) if tag_type == "aether-chat-summary" else ""
                else:
                    continue
                
                # Parse attributes
                attributes = self._parse_attributes(attr_string)
                
                # Create action tag
                if tag_type != "aether-chat-summary":  # Don't create activity for summary
                    action_tag = ActionTag(tag_type, content, attributes)
                    action_tags.append(action_tag)
                    logger.info(f"Parsed {tag_type}: {action_tag._generate_title()}")
                
                # Remove tag from cleaned response
                cleaned_response = cleaned_response.replace(match.group(0), "")
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response).strip()
        
        logger.info(f"Parsed {len(action_tags)} action tags")
        return action_tags, cleaned_response
    
    def _parse_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse key="value" attributes from tag."""
        attributes = {}
        
        # Match key="value" or key='value'
        attr_pattern = r'(\w+)=["\']([^"\']*)["\']'
        matches = re.findall(attr_pattern, attr_string)
        
        for key, value in matches:
            attributes[key] = value
        
        return attributes
    
    def extract_code_blocks(self, brain_response: str) -> List[Dict]:
        """
        Extract markdown code blocks (```language ... ```).
        
        Returns list of {language, code} dicts.
        """
        code_blocks = []
        
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, brain_response, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()
            
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        logger.debug(f"Extracted {len(code_blocks)} code blocks")
        return code_blocks
    
    def parse_thinking(self, brain_response: str) -> Tuple[List[str], str]:
        """
        Extract <think> tags and return thinking steps + cleaned response.
        
        Returns:
            (thinking_steps, cleaned_response)
        """
        thinking_steps = []
        cleaned_response = brain_response
        
        # Extract all <think> tags
        think_pattern = r'<think>(.*?)</think>'
        matches = re.finditer(think_pattern, brain_response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            thinking_content = match.group(1).strip()
            
            # Split into bullet points or lines
            if '\n' in thinking_content:
                # Multi-line thinking
                lines = [line.strip() for line in thinking_content.split('\n') if line.strip()]
                thinking_steps.extend(lines)
            else:
                # Single line
                thinking_steps.append(thinking_content)
            
            # Remove from cleaned response
            cleaned_response = cleaned_response.replace(match.group(0), "")
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response).strip()
        
        logger.info(f"Extracted {len(thinking_steps)} thinking steps")
        return thinking_steps, cleaned_response


class ActionExecutor:
    """Execute parsed action tags."""
    
    def __init__(self, router, store, memory):
        self.router = router
        self.store = store
        self.memory = memory
        logger.info("ActionExecutor initialized")
    
    async def execute(self, action_tag: ActionTag, user_id: str) -> Dict:
        """
        Execute an action tag and return detailed result.
        
        Returns:
            {
                "success": bool,
                "result": str,
                "error": Optional[str],
                "output": Optional[str],  # Actual stdout/stderr
                "metadata": Dict  # Additional execution info
            }
        """
        execution_start = datetime.now()
        
        try:
            if action_tag.tag_type == "aether-write":
                result = await self._execute_write(action_tag)
            
            elif action_tag.tag_type == "aether-sandbox":
                result = await self._execute_sandbox(action_tag)
            
            elif action_tag.tag_type == "aether-forge":
                result = await self._execute_forge(action_tag)
            
            elif action_tag.tag_type == "aether-install":
                result = await self._execute_install(action_tag)
            
            elif action_tag.tag_type == "aether-research":
                result = await self._execute_research(action_tag, user_id)
            
            elif action_tag.tag_type == "aether-command":
                result = await self._execute_command(action_tag)
            
            else:
                result = {
                    "success": False,
                    "result": "",
                    "error": f"Unknown tag type: {action_tag.tag_type}",
                    "output": None,
                    "metadata": {}
                }
            
            # Add execution metadata
            execution_duration = (datetime.now() - execution_start).total_seconds()
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["execution_time"] = execution_duration
            result["metadata"]["timestamp"] = datetime.now().isoformat()
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to execute {action_tag.tag_type}: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }
    
    async def _execute_write(self, tag: ActionTag) -> Dict:
        """Execute file write operation."""
        path = tag.attributes.get("path", "output.txt")
        content = tag.content
        
        # Create workspace directory if it doesn't exist
        workspace_dir = os.path.expanduser("~/AetherMind_Workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Full path for the file
        full_path = os.path.join(workspace_dir, path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            # Actually write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = os.path.getsize(full_path)
            logger.info(f"✅ File written: {full_path} ({file_size} bytes)")
            
            return {
                "success": True,
                "result": f"Created {path} in ~/AetherMind_Workspace/",
                "output": f"File: {path} ({file_size} bytes)",
                "error": None,
                "metadata": {
                    "file_path": full_path,
                    "file_size": file_size,
                    "content_length": len(content)
                }
            }
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "output": None,
                "error": str(e),
                "metadata": {"attempted_path": full_path}
            }
    
    async def _execute_sandbox(self, tag: ActionTag) -> Dict:
        """Execute code in sandbox."""
        language = tag.attributes.get("language", "python")
        code = tag.content
        is_test = tag.attributes.get("test", "false") == "true"
        
        # Use PracticeAdapter
        intent = json.dumps({
            "language": "python" if language == "python" else "bash",
            "code": code,
            "tests": [] if not is_test else ["# Test code"]
        })
        
        try:
            result = await self.router.adapters["practice"].execute(intent)
            
            return {
                "success": True,
                "result": result,
                "output": result,  # Actual execution output
                "error": None,
                "metadata": {
                    "language": language,
                    "code_length": len(code),
                    "is_test": is_test
                }
            }
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "output": None,
                "error": str(e),
                "metadata": {"language": language}
            }
    
    async def _execute_forge(self, tag: ActionTag) -> Dict:
        """Execute ToolForge operation."""
        action = tag.attributes.get("action", "generate")
        tool_name = tag.attributes.get("tool", "custom_tool")
        
        # Parse JSON content
        try:
            tool_spec = json.loads(tag.content)
            tool_spec["action"] = action
            tool_spec["name"] = tool_name
            
            intent = json.dumps(tool_spec)
            result = self.router.adapters["toolforge"].execute(intent)
            
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "result": "",
                "error": f"Invalid JSON: {e}"
            }
    
    async def _execute_install(self, tag: ActionTag) -> Dict:
        """Execute package installation."""
        packages = tag.attributes.get("packages", "").split()
        
        results = []
        for pkg in packages:
            intent = json.dumps({
                "action": "pypi_install",
                "name": pkg
            })
            result = self.router.adapters["toolforge"].execute(intent)
            results.append(f"{pkg}: {result}")
        
        return {
            "success": True,
            "result": "\n".join(results),
            "error": None
        }
    
    async def _execute_research(self, tag: ActionTag, user_id: str) -> Dict:
        """Execute knowledge base research."""
        query = tag.attributes.get("query", "")
        namespace = tag.attributes.get("namespace", "core_universal")
        
        # Query vector store
        contexts, state_vec = self.store.query_context(query, namespace=namespace)
        
        return {
            "success": True,
            "result": "\n".join(contexts[:5]),  # Top 5 results
            "error": None
        }
    
    async def _execute_command(self, tag: ActionTag) -> Dict:
        """Execute system command."""
        cmd_type = tag.attributes.get("type", "unknown")
        
        # These are UI-level commands, just return instruction
        return {
            "success": True,
            "result": f"User should execute: {cmd_type}",
            "error": None
        }
