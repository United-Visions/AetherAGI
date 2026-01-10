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
        
        # Map tag types to activity types (21 total activity types)
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
            "aether-body-switch": "body_switch",
            # NEW: App/Sandbox control
            "aether-app-mode": "app_mode_control",
            "aether-app-preview": "app_preview",
            "aether-app-log": "app_log",
            # NEW: Game world control
            "aether-game": "game_action"
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
        
        # NEW: App/Sandbox control tags
        elif self.tag_type == "aether-app-mode":
            return {
                "action": self.attributes.get("action", "open"),  # open, close
                "app_name": self.attributes.get("app_name", ""),
                "template": self.attributes.get("template", "blank"),
                "description": self.content
            }
        
        elif self.tag_type == "aether-app-preview":
            return {
                "url": self.attributes.get("url", ""),
                "port": self.attributes.get("port", "5000"),
                "auto_refresh": self.attributes.get("auto_refresh", "true") == "true"
            }
        
        elif self.tag_type == "aether-app-log":
            return {
                "level": self.attributes.get("level", "info"),  # info, warning, error, success
                "message": self.content
            }
        
        # NEW: PlayCanvas game editor control
        elif self.tag_type == "aether-playcanvas":
            try:
                content_json = json.loads(self.content)
            except:
                content_json = {}
            
            project_id = self.attributes.get("project_id", "")
            data = {
                "action": self.attributes.get("action", "create_entity"),
                "project_id": project_id,
                "data": content_json
            }

            if project_id:
                data["project_url"] = f"https://playcanvas.com/project/{project_id}/overview"
            
            return data
        
        # NEW: Mixamo character/animation integration
        elif self.tag_type == "aether-mixamo":
            try:
                content_json = json.loads(self.content)
            except:
                content_json = {}
            
            return {
                "action": self.attributes.get("action", "download_character"),
                "character": self.attributes.get("character", ""),
                "data": content_json
            }
        
        # NEW: Meshy.ai text-to-3D generation
        elif self.tag_type == "aether-meshy":
            try:
                content_json = json.loads(self.content)
            except:
                content_json = {}
            
            return {
                "action": self.attributes.get("action", "generate"),
                "prompt": self.attributes.get("prompt", ""),
                "data": content_json
            }
        
        # NEW: SketchFab model library
        elif self.tag_type == "aether-sketchfab":
            try:
                content_json = json.loads(self.content)
            except:
                content_json = {}
            
            return {
                "action": self.attributes.get("action", "search"),
                "query": self.attributes.get("query", ""),
                "data": content_json
            }
        
        # NEW: Game world control
        elif self.tag_type == "aether-game":
            try:
                content_json = json.loads(self.content)
            except:
                content_json = {}
            
            return {
                "action": self.attributes.get("action", "move"),
                "target": self.attributes.get("target", "player"),
                "params": content_json
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
        
        # NEW: App/Sandbox control tags
        elif self.tag_type == "aether-app-mode":
            action = self.attributes.get("action", "open")
            app_name = self.attributes.get("app_name", "app")
            return f"{'Opening' if action == 'open' else 'Closing'} App Mode: {app_name}"
        
        elif self.tag_type == "aether-app-preview":
            port = self.attributes.get("port", "5000")
            return f"Updating preview on port {port}"
        
        elif self.tag_type == "aether-app-log":
            level = self.attributes.get("level", "info")
            return f"[{level.upper()}] Build log"
        
        # NEW: PlayCanvas game editor control
        elif self.tag_type == "aether-playcanvas":
            action = self.attributes.get("action", "unknown")
            project_id = self.attributes.get("project_id", "?")
            return f"PlayCanvas: {action} (project {project_id})"
        
        # NEW: Mixamo character/animation integration
        elif self.tag_type == "aether-mixamo":
            action = self.attributes.get("action", "unknown")
            character = self.attributes.get("character", "character")
            return f"Mixamo: {action} ({character})"
        
        # NEW: Meshy.ai text-to-3D generation
        elif self.tag_type == "aether-meshy":
            action = self.attributes.get("action", "unknown")
            prompt = self.attributes.get("prompt", "model")
            return f"Meshy: Generating 3D model ({prompt[:30]}...)"
        
        # NEW: SketchFab model library
        elif self.tag_type == "aether-sketchfab":
            action = self.attributes.get("action", "unknown")
            query = self.attributes.get("query", "models")
            return f"SketchFab: {action} ({query})"
        
        # NEW: Game world control
        elif self.tag_type == "aether-game":
            action = self.attributes.get("action", "move")
            target = self.attributes.get("target", "player")
            return f"Game: {action} {target}"
        
        return "Processing action"


class ActionParser:
    """Parse AetherMind action tags from Brain responses."""
    
    # Tag patterns (20 action tag types + 2 special tags: think, summary)
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
        # NEW: App/Sandbox control tags
        "aether-app-mode": r'<aether-app-mode\s*(.*?)>(.*?)</aether-app-mode>',
        "aether-app-preview": r'<aether-app-preview\s*(.*?)>(.*?)</aether-app-preview>',
        "aether-app-log": r'<aether-app-log\s*(.*?)>(.*?)</aether-app-log>',
        # NEW: PlayCanvas game editor control
        "aether-playcanvas": r'<aether-playcanvas\s+(.*?)>(.*?)</aether-playcanvas>',
        # NEW: Mixamo character/animation integration
        "aether-mixamo": r'<aether-mixamo\s+(.*?)>(.*?)</aether-mixamo>',
        # NEW: Meshy.ai text-to-3D generation
        "aether-meshy": r'<aether-meshy\s+(.*?)>(.*?)</aether-meshy>',
        # NEW: SketchFab model library
        "aether-sketchfab": r'<aether-sketchfab\s+(.*?)>(.*?)</aether-sketchfab>',
        # NEW: 3D Embodiment control (real-time doll commands)
        "aether-3d-move": r'<aether-3d-move\s+(.*?)>(.*?)</aether-3d-move>',
        "aether-3d-look": r'<aether-3d-look\s+(.*?)>(.*?)</aether-3d-look>',
        "aether-3d-animation": r'<aether-3d-animation\s+(.*?)>(.*?)</aether-3d-animation>',
        "aether-3d-explore": r'<aether-3d-explore\s+(.*?)>(.*?)</aether-3d-explore>',
        "aether-3d-teleport": r'<aether-3d-teleport\s+(.*?)>(.*?)</aether-3d-teleport>',
        "aether-3d-emotion": r'<aether-3d-emotion\s+(.*?)>(.*?)</aether-3d-emotion>',
        "aether-3d-state": r'<aether-3d-state>(.*?)</aether-3d-state>',
        "aether-3d-perception": r'<aether-3d-perception>(.*?)</aether-3d-perception>',
        # NEW: Active vision control (camera/gaze)
        "aether-3d-vision": r'<aether-3d-vision\s+(.*?)>(.*?)</aether-3d-vision>',
        # OLD: Game environment control (3D world actions) - deprecated in favor of specific 3d tags
        "aether-game": r'<aether-game\s+(.*?)>(.*?)</aether-game>',
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
            
            elif action_tag.tag_type == "aether-playcanvas":
                result = await self._execute_playcanvas(action_tag)
            
            elif action_tag.tag_type == "aether-mixamo":
                result = await self._execute_mixamo(action_tag)
            
            elif action_tag.tag_type == "aether-meshy":
                result = await self._execute_meshy(action_tag)
            
            elif action_tag.tag_type == "aether-sketchfab":
                result = await self._execute_sketchfab(action_tag)
            
            elif action_tag.tag_type == "aether-game":
                result = await self._execute_game(action_tag)
            
            # 3D Embodiment control tags
            elif action_tag.tag_type in ["aether-3d-move", "aether-3d-look", "aether-3d-animation",
                                          "aether-3d-explore", "aether-3d-teleport", "aether-3d-emotion",
                                          "aether-3d-state", "aether-3d-perception", "aether-3d-vision"]:
                result = await self._execute_3d_embodiment(action_tag)
            
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
    
    async def _execute_playcanvas(self, tag: ActionTag) -> Dict:
        """Execute PlayCanvas editor automation via browser control."""
        action = tag.attributes.get("action", "")
        project_id = tag.attributes.get("project_id", "")
        data = tag.data
        
        # Build intent JSON for PlayCanvasEditorAdapter
        intent = json.dumps({
            "action": action,
            "project_id": project_id,
            **data  # Merge additional data fields
        })
        
        try:
            # Forward to PlayCanvas adapter via router
            if "playcanvas_editor" in self.router.adapters:
                result = await self.router.adapters["playcanvas_editor"].execute(intent)
                result_json = json.loads(result)
                if project_id:
                    result_json.setdefault("project_id", project_id)
                    result_json.setdefault(
                        "project_url",
                        f"https://playcanvas.com/project/{project_id}/overview"
                    )
                
                return {
                    "success": result_json.get("status") == "success",
                    "result": result_json.get("message", ""),
                    "error": result_json.get("message") if result_json.get("status") == "error" else None,
                    "metadata": result_json
                }
            else:
                return {
                    "success": False,
                    "result": "",
                    "error": "PlayCanvas adapter not enabled. Set ENABLE_PLAYCANVAS_EDITOR=true in config/settings.yaml"
                }
        except Exception as e:
            logger.error(f"PlayCanvas execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }    
    async def _execute_meshy(self, tag: ActionTag) -> Dict:
        """Execute Meshy.ai text-to-3D generation."""
        action = tag.attributes.get("action", "generate")
        prompt = tag.attributes.get("prompt", "")
        data = tag.data
        
        # Build intent JSON for MeshyAdapter
        intent = json.dumps({
            "action": action,
            "prompt": prompt,
            **data  # Merge additional data fields
        })
        
        try:
            # Forward to Meshy adapter via router
            if "meshy" in self.router.adapters:
                result = await self.router.adapters["meshy"].execute(intent)
                result_json = json.loads(result)
                
                return {
                    "success": result_json.get("status") == "success",
                    "result": result_json.get("message", ""),
                    "error": result_json.get("message") if result_json.get("status") == "error" else None,
                    "files": [result_json.get("file")] if result_json.get("file") else [],
                    "metadata": {
                        "prompt": prompt,
                        "format": result_json.get("format", ""),
                        "task_id": result_json.get("task_id", "")
                    }
                }
            else:
                return {
                    "success": False,
                    "result": "",
                    "error": "Meshy adapter not enabled"
                }
        except Exception as e:
            logger.error(f"Meshy execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }
    
    async def _execute_sketchfab(self, tag: ActionTag) -> Dict:
        """Execute SketchFab model search/download."""
        action = tag.attributes.get("action", "search")
        query = tag.attributes.get("query", "")
        data = tag.data
        
        # Build intent JSON for SketchfabAdapter
        intent = json.dumps({
            "action": action,
            "query": query,
            **data  # Merge additional data fields
        })
        
        try:
            # Forward to SketchFab adapter via router
            if "sketchfab" in self.router.adapters:
                result = await self.router.adapters["sketchfab"].execute(intent)
                result_json = json.loads(result)
                
                return {
                    "success": result_json.get("status") == "success",
                    "result": result_json.get("message", "") or json.dumps(result_json.get("models", []), indent=2),
                    "error": result_json.get("message") if result_json.get("status") == "error" else None,
                    "files": [result_json.get("file")] if result_json.get("file") else [],
                    "metadata": {
                        "query": query,
                        "count": result_json.get("count", 0),
                        "models": result_json.get("models", [])
                    }
                }
            else:
                return {
                    "success": False,
                    "result": "",
                    "error": "SketchFab adapter not enabled"
                }
        except Exception as e:
            logger.error(f"SketchFab execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }
    
    async def _execute_mixamo(self, tag: ActionTag) -> Dict:
        """Execute Mixamo character/animation download."""
        action = tag.attributes.get("action", "download_character")
        character = tag.attributes.get("character", "")
        data = tag.data
        
        # Build intent JSON for MixamoAdapter
        intent = json.dumps({
            "action": action,
            "character": character,
            **data  # Merge additional data fields
        })
        
        try:
            # Forward to Mixamo adapter via router
            if "mixamo" in self.router.adapters:
                result = await self.router.adapters["mixamo"].execute(intent)
                result_json = json.loads(result)
                
                return {
                    "success": result_json.get("status") == "success",
                    "result": result_json.get("message", ""),
                    "error": result_json.get("message") if result_json.get("status") == "error" else None,
                    "files": result_json.get("files", []),
                    "metadata": {
                        "character": character,
                        "animations": result_json.get("animations", []),
                        "download_dir": result_json.get("download_dir", "")
                    }
                }
            else:
                return {
                    "success": False,
                    "result": "",
                    "error": "Mixamo adapter not enabled"
                }
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": f"PlayCanvas execution failed: {str(e)}"
            }
    
    async def _execute_game(self, tag: ActionTag) -> Dict:
        """Execute game world action - sends command to Unity/Three.js environment."""
        action = tag.attributes.get("action", "move")
        target = tag.attributes.get("target", "player")
        
        try:
            params = json.loads(tag.content) if tag.content.strip() else {}
        except json.JSONDecodeError:
            params = {}
        
        # Build the game command
        game_command = {
            "action": action,
            "target": target,
            "params": params
        }
        
        try:
            # Import the Unity adapter and queue the command
            from body.adapters.unity_adapter import UNITY_ADAPTER
            
            result = UNITY_ADAPTER.execute(json.dumps(game_command))
            result_json = json.loads(result)
            
            logger.info(f"🎮 Game command queued: {action} -> {target}")
            
            return {
                "success": result_json.get("status") == "queued",
                "result": f"Executed {action} on {target}",
                "error": None,
                "output": result,
                "metadata": {
                    "action": action,
                    "target": target,
                    "params": params,
                    "queue_length": result_json.get("queue_length", 0)
                }
            }
        except Exception as e:
            logger.error(f"Game action failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "metadata": {"action": action, "target": target}
            }
    
    async def _execute_3d_embodiment(self, tag: ActionTag) -> Dict:
        """Execute 3D embodiment control commands - real-time doll control via WebSocket."""
        try:
            # Import the embodiment adapter
            from body.adapters.embodiment_3d_adapter import get_embodiment_adapter
            embodiment_adapter = get_embodiment_adapter()
            
            # Map tag types to action types
            action_type_map = {
                "aether-3d-move": "move",
                "aether-3d-look": "look",
                "aether-3d-animation": "animation",
                "aether-3d-explore": "explore",
                "aether-3d-teleport": "teleport",
                "aether-3d-emotion": "emotion",
                "aether-3d-state": "get_state",
                "aether-3d-perception": "get_perception",
                "aether-3d-vision": "vision"
            }
            
            action_type = action_type_map.get(tag.tag_type, "unknown")
            
            # Parse content as JSON or use attributes
            params = {}
            if tag.content.strip():
                try:
                    params = json.loads(tag.content)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text description
                    params = {"description": tag.content}
            
            # Merge attributes into params
            params.update(tag.attributes)
            
            # Build the intent JSON
            intent = {
                "action_type": action_type,
                **params
            }
            
            # Execute via adapter (broadcasts to WebSocket clients)
            result = await embodiment_adapter.execute(json.dumps(intent))
            
            logger.info(f"🎮 3D Embodiment: {action_type} executed")
            
            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {
                    "action_type": action_type,
                    "connected_clients": len(embodiment_adapter.websocket_clients),
                    "params": params
                }
            }
            
        except Exception as e:
            logger.error(f"3D Embodiment execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "metadata": {"tag_type": tag.tag_type}
            }
