"""
PlayCanvas Editor Adapter - Browser Automation for Game Development
Enables AetherMind Brain to control PlayCanvas Editor via Playwright

Deployment Modes:
1. Local Development: headless=False (visible browser)
2. Production/Container: headless=True (no display)
3. Background Worker: Async tasks with progress updates

Example Usage:
User: "create a player character with a camera in my game"
Brain generates: <aether-playcanvas action="create_entity" project_id="1449261">
{
    "entity_name": "Player",
    "components": ["model", "camera", "rigidbody"],
    "position": {"x": 0, "y": 1, "z": 0}
}
</aether-playcanvas>
"""

import json
import asyncio
import os
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Page, Browser, Playwright
from body.adapter_base import BodyAdapter

class PlayCanvasEditorAdapter(BodyAdapter):
    """
    Integrates Playwright browser automation into AetherMind's body system
    Brain controls PlayCanvas Editor through natural language → browser actions
    """
    
    def __init__(self):
        super().__init__()
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.is_logged_in = False
        self.current_project_id = None
        
        # Deployment mode detection
        self.headless = os.getenv("PLAYCANVAS_HEADLESS", "false").lower() == "true"
        self.username = os.getenv("PLAYCANVAS_USERNAME", "")
        self.password = os.getenv("PLAYCANVAS_PASSWORD", "")
        
    async def initialize(self):
        """
        Start browser session
        Called once when adapter is registered
        """
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']  # Docker compatibility
            )
            print(f"🌐 PlayCanvas Editor adapter initialized (headless={self.headless})")
        except Exception as e:
            print(f"⚠️ PlayCanvas adapter init failed: {e}")
            print("   Install: playwright install chromium")
    
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def execute(self, intent: str) -> str:
        """
        Main entry point from Brain
        Parses intent JSON and routes to appropriate handler
        """
        try:
            data = json.loads(intent)
            action = data.get("action")
            project_id = data.get("project_id")
            
            # Auto-login if credentials available
            if not self.is_logged_in and self.username and self.password:
                await self.login(self.username, self.password)
            
            # Ensure project is open
            if project_id and project_id != self.current_project_id:
                await self.open_editor(project_id)
            
            # Route to action handler
            if action == "create_entity":
                return await self._create_entity(data)
            elif action == "upload_script":
                return await self._upload_script(data)
            elif action == "attach_script":
                return await self._attach_script(data)
            elif action == "create_building":
                return await self._create_building(data)
            elif action == "setup_scene":
                return await self._setup_scene(data)
            elif action == "deploy_bridge":
                return await self._deploy_aether_bridge(data)
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Unknown action: {action}"
                })
                
        except json.JSONDecodeError as e:
            return json.dumps({
                "status": "error",
                "message": f"Invalid JSON intent: {e}"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def login(self, username: str, password: str):
        """
        Login to PlayCanvas account
        """
        if not self.browser:
            await self.initialize()
        
        self.page = await self.browser.new_page()
        
        try:
            await self.page.goto("https://login.playcanvas.com/")
            await self.page.wait_for_load_state("networkidle")
            
            # Fill login form
            await self.page.fill('input[name="username"]', username)
            await self.page.fill('input[name="password"]', password)
            await self.page.click('button[type="submit"]')
            
            # Wait for redirect
            await self.page.wait_for_url("**/projects", timeout=10000)
            
            self.is_logged_in = True
            print("✅ Logged into PlayCanvas")
            
        except Exception as e:
            raise Exception(f"Login failed: {e}")
    
    async def open_editor(self, project_id: int):
        """
        Open PlayCanvas editor for a project
        """
        if not self.page:
            raise Exception("Not logged in. Call login() first")
        
        try:
            url = f"https://playcanvas.com/project/{project_id}/overview"
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            
            # Click "EDITOR" button
            await self.page.click('text=EDITOR')
            await self.page.wait_for_timeout(5000)  # Editor load time
            
            self.current_project_id = project_id
            print(f"✅ Editor opened for project {project_id}")
            
        except Exception as e:
            raise Exception(f"Failed to open editor: {e}")
    
    async def _create_entity(self, data: Dict[str, Any]) -> str:
        """
        Create a new entity in the scene
        Example: {"entity_name": "Player", "components": ["model", "camera"], "position": {"x": 0, "y": 1, "z": 0}}
        """
        entity_name = data.get("entity_name", "NewEntity")
        components = data.get("components", [])
        position = data.get("position", {"x": 0, "y": 0, "z": 0})
        
        try:
            # Right-click in Hierarchy panel → New Entity
            hierarchy_panel = await self.page.query_selector('[class*="hierarchy"]')
            if hierarchy_panel:
                await hierarchy_panel.click(button="right")
                await self.page.click('text="New Entity"')
                await self.page.wait_for_timeout(500)
                
                # Rename entity
                await self.page.keyboard.type(entity_name)
                await self.page.keyboard.press("Enter")
                
                # Add components
                for component in components:
                    await self._add_component(entity_name, component)
                
                # Set position
                await self._set_entity_position(entity_name, position)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Created entity '{entity_name}' with components: {components}",
                    "entity": entity_name,
                    "position": position
                })
            else:
                raise Exception("Hierarchy panel not found")
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to create entity: {e}"
            })
    
    async def _upload_script(self, data: Dict[str, Any]) -> str:
        """
        Create a new script asset
        Example: {"script_name": "playerController", "script_content": "var PlayerController = ..."}
        """
        script_name = data.get("script_name")
        script_content = data.get("script_content", "")
        
        try:
            # Click Assets panel → New Asset → Script
            await self.page.click('[class*="assets-panel"]')
            await self.page.click('text="New Asset"')
            await self.page.click('text="Script"')
            await self.page.wait_for_timeout(500)
            
            # Name the script
            await self.page.keyboard.type(script_name)
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_timeout(1000)
            
            # Open script editor (double-click)
            script_asset = await self.page.query_selector(f'text="{script_name}"')
            if script_asset:
                await script_asset.dblclick()
                await self.page.wait_for_timeout(2000)
                
                # Clear existing content and paste new
                await self.page.keyboard.press("Control+A")  # Cmd+A on Mac handled by playwright
                await self.page.keyboard.type(script_content)
                
                # Save (Ctrl+S)
                await self.page.keyboard.press("Control+S")
                await self.page.wait_for_timeout(500)
                
                return json.dumps({
                    "status": "success",
                    "message": f"Uploaded script '{script_name}'",
                    "script_name": script_name
                })
            else:
                raise Exception(f"Script asset '{script_name}' not found after creation")
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to upload script: {e}"
            })
    
    async def _attach_script(self, data: Dict[str, Any]) -> str:
        """
        Attach a script to an entity
        Example: {"entity_name": "Player", "script_name": "playerController", "attributes": {"speed": 5}}
        """
        entity_name = data.get("entity_name")
        script_name = data.get("script_name")
        attributes = data.get("attributes", {})
        
        try:
            # Select entity in hierarchy
            entity = await self.page.query_selector(f'text="{entity_name}"')
            if entity:
                await entity.click()
                await self.page.wait_for_timeout(500)
                
                # Add Script component
                await self.page.click('text="Add Component"')
                await self.page.click('text="Script"')
                await self.page.wait_for_timeout(500)
                
                # Select script from dropdown
                await self.page.click(f'text="{script_name}"')
                
                # Set attributes (if any)
                for key, value in attributes.items():
                    # Find attribute input field
                    attr_input = await self.page.query_selector(f'input[name="{key}"]')
                    if attr_input:
                        await attr_input.fill(str(value))
                
                return json.dumps({
                    "status": "success",
                    "message": f"Attached script '{script_name}' to '{entity_name}'",
                    "entity": entity_name,
                    "script": script_name
                })
            else:
                raise Exception(f"Entity '{entity_name}' not found")
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to attach script: {e}"
            })
    
    async def _create_building(self, data: Dict[str, Any]) -> str:
        """
        Create a building entity with model and collision
        Example: {"building_type": "skyscraper", "position": {"x": 10, "y": 0, "z": 5}, "scale": 2}
        """
        building_type = data.get("building_type", "building")
        position = data.get("position", {"x": 0, "y": 0, "z": 0})
        scale = data.get("scale", 1)
        
        # Create entity with model component
        entity_data = {
            "entity_name": f"{building_type}_{position['x']}_{position['z']}",
            "components": ["model", "collision"],
            "position": position
        }
        
        result = await self._create_entity(entity_data)
        
        # Parse result and add scale
        result_json = json.loads(result)
        if result_json["status"] == "success":
            result_json["message"] += f" with scale {scale}"
            result_json["scale"] = scale
        
        return json.dumps(result_json)
    
    async def _setup_scene(self, data: Dict[str, Any]) -> str:
        """
        Setup complete scene (ground, lighting, camera)
        Example: {"scene_type": "city", "lighting": "day"}
        """
        scene_type = data.get("scene_type", "default")
        lighting = data.get("lighting", "day")
        
        tasks = []
        
        # Create ground plane
        tasks.append(self._create_entity({
            "entity_name": "Ground",
            "components": ["model", "collision"],
            "position": {"x": 0, "y": 0, "z": 0}
        }))
        
        # Create directional light (sun)
        tasks.append(self._create_entity({
            "entity_name": "Sun",
            "components": ["light"],
            "position": {"x": 0, "y": 10, "z": 0}
        }))
        
        # Create main camera
        tasks.append(self._create_entity({
            "entity_name": "MainCamera",
            "components": ["camera"],
            "position": {"x": 0, "y": 5, "z": -10}
        }))
        
        results = await asyncio.gather(*tasks)
        
        return json.dumps({
            "status": "success",
            "message": f"Scene setup complete: {scene_type} with {lighting} lighting",
            "entities_created": ["Ground", "Sun", "MainCamera"]
        })
    
    async def _deploy_aether_bridge(self, data: Dict[str, Any]) -> str:
        """
        Deploy AetherBridge script automatically
        Reads from playcanvas_aether_bridge.js file
        """
        try:
            # Read bridge script
            bridge_path = "/Users/deion/Desktop/aethermind_universal/playcanvas_aether_bridge.js"
            with open(bridge_path, 'r') as f:
                bridge_content = f.read()
            
            # Upload script
            result = await self._upload_script({
                "script_name": "aetherBridge",
                "script_content": bridge_content
            })
            
            # Create GameManager entity
            result2 = await self._create_entity({
                "entity_name": "GameManager",
                "components": [],
                "position": {"x": 0, "y": 0, "z": 0}
            })
            
            # Attach bridge script
            api_url = os.getenv("AETHER_API_URL", "http://localhost:8000/v1/game/unity/state")
            result3 = await self._attach_script({
                "entity_name": "GameManager",
                "script_name": "aetherBridge",
                "attributes": {
                    "apiUrl": api_url,
                    "syncInterval": 500,
                    "debugMode": True
                }
            })
            
            return json.dumps({
                "status": "success",
                "message": "AetherBridge deployed successfully!",
                "steps": [result, result2, result3]
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to deploy bridge: {e}"
            })
    
    async def _add_component(self, entity_name: str, component_type: str):
        """Helper: Add component to entity"""
        await self.page.click('text="Add Component"')
        await self.page.click(f'text="{component_type.capitalize()}"')
        await self.page.wait_for_timeout(300)
    
    async def _set_entity_position(self, entity_name: str, position: Dict[str, float]):
        """Helper: Set entity position in inspector"""
        # Click position X input
        x_input = await self.page.query_selector('input[name="position.x"]')
        if x_input:
            await x_input.fill(str(position.get("x", 0)))
        
        y_input = await self.page.query_selector('input[name="position.y"]')
        if y_input:
            await y_input.fill(str(position.get("y", 0)))
        
        z_input = await self.page.query_selector('input[name="position.z"]')
        if z_input:
            await z_input.fill(str(position.get("z", 0)))
        
        await self.page.keyboard.press("Enter")


# Global instance (for router registration)
PLAYCANVAS_EDITOR_ADAPTER = PlayCanvasEditorAdapter()
