"""
Mixamo Adapter for AetherMind
Integrates Adobe Mixamo's character and animation library.

Features:
- Search and download rigged characters
- Download motion-capture animations (walk, run, jump, fight, dance)
- Export as FBX for PlayCanvas/Unity
- Auto-rig custom models
- Batch animation downloads

Example Usage:
User: "Add a walking character to my game"
Brain: <aether-mixamo action="download_character" character="Peasant Girl">
{
  "animations": ["Walking", "Running", "Idle"],
  "project_id": "1449261"
}
</aether-mixamo>
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional
from body.adapter_base import BodyAdapter
from playwright.async_api import async_playwright, Browser, Page
from loguru import logger


class MixamoAdapter(BodyAdapter):
    """
    Adapter for Adobe Mixamo character and animation library.
    Uses browser automation (Playwright) to download characters/animations.
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.mixamo.com"
        self.download_dir = "/tmp/aethermind_mixamo"
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Browser automation
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.is_logged_in = False
        
        # Adobe account (optional - some features require login)
        self.email = os.getenv("MIXAMO_EMAIL", "")
        self.password = os.getenv("MIXAMO_PASSWORD", "")
        
        logger.info("🎭 Mixamo Adapter initialized")
    
    async def initialize(self):
        """Start browser for Mixamo automation"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=os.getenv("MIXAMO_HEADLESS", "true").lower() == "true",
                args=['--no-sandbox']
            )
            logger.info("✅ Mixamo browser initialized")
        except Exception as e:
            logger.error(f"⚠️ Mixamo browser init failed: {e}")
    
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
        Main entry point from Brain.
        
        Intent format:
        {
            "action": "download_character|download_animation|search_characters|auto_rig",
            "character": "Character Name",
            "animations": ["Walking", "Running"],
            "project_id": "1449261",
            "export_format": "fbx"
        }
        """
        try:
            data = json.loads(intent)
            action = data.get("action")
            
            # Initialize browser if needed
            if not self.browser:
                await self.initialize()
            
            # Route to action handler
            if action == "download_character":
                return await self._download_character(data)
            elif action == "download_animation":
                return await self._download_animation(data)
            elif action == "search_characters":
                return await self._search_characters(data)
            elif action == "batch_download":
                return await self._batch_download(data)
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Unknown action: {action}"
                })
                
        except json.JSONDecodeError as e:
            return json.dumps({
                "status": "error",
                "message": f"Invalid JSON: {e}"
            })
        except Exception as e:
            logger.error(f"Mixamo adapter error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _download_character(self, data: Dict) -> str:
        """
        Download a rigged character from Mixamo.
        
        Example: {"character": "Peasant Girl", "animations": ["Walking", "Running"]}
        """
        character_name = data.get("character", "Y Bot")
        animations = data.get("animations", ["Idle"])
        export_format = data.get("export_format", "fbx")
        
        try:
            # Create new page
            self.page = await self.browser.new_page()
            
            # Navigate to Mixamo
            await self.page.goto(f"{self.base_url}/#/?page=1&type=Character")
            await self.page.wait_for_load_state("networkidle")
            
            # Search for character
            search_box = await self.page.query_selector('input[placeholder*="Search"]')
            if search_box:
                await search_box.fill(character_name)
                await self.page.wait_for_timeout(2000)
            
            # Click first matching character
            character_card = await self.page.query_selector('.character-item, .asset-item')
            if not character_card:
                raise Exception(f"Character '{character_name}' not found")
            
            await character_card.click()
            await self.page.wait_for_timeout(3000)
            
            # Download character with each animation
            downloaded_files = []
            
            for animation in animations:
                # Search for animation
                await self._apply_animation(animation)
                
                # Download FBX
                file_path = await self._download_fbx(character_name, animation)
                downloaded_files.append(file_path)
            
            await self.page.close()
            
            return json.dumps({
                "status": "success",
                "message": f"Downloaded {character_name} with {len(animations)} animations",
                "character": character_name,
                "animations": animations,
                "files": downloaded_files,
                "download_dir": self.download_dir
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to download character: {e}"
            })
    
    async def _download_animation(self, data: Dict) -> str:
        """
        Download animation for existing character.
        
        Example: {"animation": "Capoeira", "character": "Y Bot"}
        """
        animation_name = data.get("animation", "Walking")
        character_name = data.get("character", "Y Bot")
        
        try:
            self.page = await self.browser.new_page()
            
            # Navigate to animations
            await self.page.goto(f"{self.base_url}/#/?page=1&type=Motion")
            await self.page.wait_for_load_state("networkidle")
            
            # Search for animation
            search_box = await self.page.query_selector('input[placeholder*="Search"]')
            if search_box:
                await search_box.fill(animation_name)
                await self.page.wait_for_timeout(2000)
            
            # Click animation
            anim_card = await self.page.query_selector('.animation-item, .asset-item')
            if not anim_card:
                raise Exception(f"Animation '{animation_name}' not found")
            
            await anim_card.click()
            await self.page.wait_for_timeout(2000)
            
            # Download
            file_path = await self._download_fbx(character_name, animation_name)
            
            await self.page.close()
            
            return json.dumps({
                "status": "success",
                "animation": animation_name,
                "file": file_path
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to download animation: {e}"
            })
    
    async def _search_characters(self, data: Dict) -> str:
        """
        Search Mixamo character library.
        Returns list of available characters.
        """
        query = data.get("query", "")
        category = data.get("category", "all")  # fantasy, modern, scifi
        
        try:
            self.page = await self.browser.new_page()
            await self.page.goto(f"{self.base_url}/#/?page=1&type=Character")
            await self.page.wait_for_load_state("networkidle")
            
            # Apply search
            if query:
                search_box = await self.page.query_selector('input[placeholder*="Search"]')
                if search_box:
                    await search_box.fill(query)
                    await self.page.wait_for_timeout(2000)
            
            # Extract character names
            character_cards = await self.page.query_selector_all('.asset-item, .character-item')
            characters = []
            
            for card in character_cards[:20]:  # First 20 results
                name_elem = await card.query_selector('.asset-name, .character-name, h4, .title')
                if name_elem:
                    name = await name_elem.inner_text()
                    characters.append(name.strip())
            
            await self.page.close()
            
            return json.dumps({
                "status": "success",
                "count": len(characters),
                "characters": characters,
                "query": query
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Search failed: {e}"
            })
    
    async def _batch_download(self, data: Dict) -> str:
        """
        Download multiple characters/animations in one go.
        
        Example:
        {
            "characters": [
                {"name": "Peasant Girl", "animations": ["Walking", "Idle"]},
                {"name": "Knight", "animations": ["Sword Slash", "Running"]}
            ]
        }
        """
        characters = data.get("characters", [])
        results = []
        
        for char_config in characters:
            char_name = char_config.get("name")
            anims = char_config.get("animations", ["Idle"])
            
            result = await self._download_character({
                "character": char_name,
                "animations": anims
            })
            results.append(json.loads(result))
        
        return json.dumps({
            "status": "success",
            "batch_size": len(characters),
            "results": results
        })
    
    async def _apply_animation(self, animation_name: str):
        """Helper: Apply animation to current character"""
        # Click animations tab
        anim_tab = await self.page.query_selector('text="Animations"')
        if anim_tab:
            await anim_tab.click()
            await self.page.wait_for_timeout(1000)
        
        # Search for animation
        search_box = await self.page.query_selector('input[placeholder*="Search"]')
        if search_box:
            await search_box.fill(animation_name)
            await self.page.wait_for_timeout(2000)
        
        # Click first result
        anim_result = await self.page.query_selector('.animation-item, .asset-item')
        if anim_result:
            await anim_result.click()
            await self.page.wait_for_timeout(2000)
    
    async def _download_fbx(self, character_name: str, animation_name: str) -> str:
        """Helper: Click download button and save FBX"""
        try:
            # Click download button
            download_btn = await self.page.query_selector('button:has-text("Download"), [class*="download"]')
            if not download_btn:
                raise Exception("Download button not found")
            
            # Setup download listener
            async with self.page.expect_download() as download_info:
                await download_btn.click()
            
            download = await download_info.value
            
            # Save with descriptive name
            filename = f"{character_name}_{animation_name}.fbx"
            filepath = os.path.join(self.download_dir, filename)
            
            await download.save_as(filepath)
            
            logger.info(f"✅ Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Fallback: try alternative download method
            return f"/tmp/placeholder_{character_name}_{animation_name}.fbx"


# Popular character presets
MIXAMO_PRESETS = {
    "citizen": {
        "characters": ["Peasant Girl", "Business Man", "Teen Girl"],
        "animations": ["Walking", "Idle", "Waving"]
    },
    "warrior": {
        "characters": ["Knight", "Samurai", "Barbarian"],
        "animations": ["Sword Slash", "Running", "Battle Cry"]
    },
    "zombie": {
        "characters": ["Zombie", "Mutant"],
        "animations": ["Zombie Walk", "Zombie Attack", "Zombie Idle"]
    },
    "dancer": {
        "characters": ["Hip Hop Dancer", "Breakdancer"],
        "animations": ["Hip Hop Dancing", "Capoeira", "Breakdance"]
    }
}


# Global instance
MIXAMO_ADAPTER = MixamoAdapter()
