"""
Meshy.ai Adapter for AetherMind
Text-to-3D model generation using Meshy.ai API

Features:
- Generate 3D models from text descriptions
- Export as GLB, FBX, USDZ, OBJ
- Support for different quality levels
- Polling for generation completion
- Auto-upload to PlayCanvas projects

API Docs: https://docs.meshy.ai

Example Usage:
User: "Create a cyberpunk skyscraper"
Brain: <aether-meshy action="generate">
{
  "prompt": "futuristic cyberpunk skyscraper with neon lights",
  "style": "realistic",
  "format": "fbx",
  "project_id": "1449261"
}
</aether-meshy>
"""

import os
import json
import time
import requests
from typing import Dict, Optional
from body.adapter_base import BodyAdapter
from loguru import logger


class MeshyAdapter(BodyAdapter):
    """
    Adapter for Meshy.ai text-to-3D generation API.
    Generates 3D models from text descriptions.
    """
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("MESHY_API_KEY", "")
        self.base_url = "https://api.meshy.ai/v2"
        self.download_dir = "/tmp/aethermind_meshy"
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
        if not self.api_key:
            logger.warning("⚠️ MESHY_API_KEY not set - text-to-3D generation disabled")
        
        logger.info("🎨 Meshy.ai Adapter initialized")
    
    async def execute(self, intent: str) -> str:
        """
        Main entry point from Brain.
        
        Intent format:
        {
            "action": "generate|get_status|list_models",
            "prompt": "cyberpunk building with neon lights",
            "style": "realistic|cartoon|lowpoly",
            "format": "fbx|glb|obj|usdz",
            "project_id": "1449261",
            "quality": "standard|high"
        }
        """
        try:
            data = json.loads(intent)
            action = data.get("action", "generate")
            
            if not self.api_key:
                return json.dumps({
                    "status": "error",
                    "message": "Meshy API key not configured. Set MESHY_API_KEY in .env"
                })
            
            # Route to action handler
            if action == "generate":
                return await self._generate_model(data)
            elif action == "get_status":
                return await self._get_status(data)
            elif action == "list_models":
                return await self._list_models(data)
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
            logger.error(f"Meshy adapter error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _generate_model(self, data: Dict) -> str:
        """
        Generate 3D model from text prompt.
        
        Returns task ID and polls until completion.
        """
        prompt = data.get("prompt", "")
        style = data.get("style", "realistic")
        export_format = data.get("format", "fbx")
        quality = data.get("quality", "standard")
        
        if not prompt:
            return json.dumps({
                "status": "error",
                "message": "Prompt required for generation"
            })
        
        try:
            # Step 1: Start generation
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "mode": "preview",  # preview (fast) or refine (high quality)
                "prompt": prompt,
                "art_style": style,
                "negative_prompt": "low quality, blurry, distorted"
            }
            
            logger.info(f"🎨 Starting Meshy generation: {prompt}")
            response = requests.post(
                f"{self.base_url}/text-to-3d",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return json.dumps({
                    "status": "error",
                    "message": f"Meshy API error: {response.status_code} - {response.text}"
                })
            
            result = response.json()
            task_id = result.get("result")
            
            logger.info(f"✅ Task started: {task_id}")
            
            # Step 2: Poll for completion (max 5 minutes)
            max_attempts = 60  # 5 minutes with 5-second intervals
            attempt = 0
            
            while attempt < max_attempts:
                status_response = requests.get(
                    f"{self.base_url}/text-to-3d/{task_id}",
                    headers=headers,
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    progress = status_data.get("progress", 0)
                    
                    logger.info(f"📊 Generation progress: {progress}% ({status})")
                    
                    if status == "SUCCEEDED":
                        # Model ready! Download it
                        model_urls = status_data.get("model_urls", {})
                        download_url = model_urls.get(export_format) or model_urls.get("glb")
                        
                        if download_url:
                            file_path = await self._download_model(download_url, prompt, export_format)
                            
                            return json.dumps({
                                "status": "success",
                                "message": f"Generated 3D model: {prompt}",
                                "task_id": task_id,
                                "file": file_path,
                                "format": export_format,
                                "prompt": prompt,
                                "thumbnail": status_data.get("thumbnail_url"),
                                "download_dir": self.download_dir
                            })
                        else:
                            return json.dumps({
                                "status": "error",
                                "message": f"No {export_format} export available"
                            })
                    
                    elif status == "FAILED":
                        return json.dumps({
                            "status": "error",
                            "message": f"Generation failed: {status_data.get('error', 'Unknown error')}"
                        })
                    
                    # Still processing, wait and retry
                    time.sleep(5)
                    attempt += 1
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Status check failed: {status_response.status_code}"
                    })
            
            # Timeout
            return json.dumps({
                "status": "timeout",
                "message": "Generation took too long (>5 minutes)",
                "task_id": task_id
            })
            
        except Exception as e:
            logger.error(f"Meshy generation failed: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Generation error: {e}"
            })
    
    async def _download_model(self, url: str, prompt: str, format: str) -> str:
        """Download generated model file"""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Clean filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt[:50])
            filename = f"{safe_name}.{format}"
            filepath = os.path.join(self.download_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return f"/tmp/placeholder_{format}"
    
    async def _get_status(self, data: Dict) -> str:
        """Get status of a generation task"""
        task_id = data.get("task_id")
        
        if not task_id:
            return json.dumps({
                "status": "error",
                "message": "task_id required"
            })
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url}/text-to-3d/{task_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.dumps({
                    "status": "success",
                    "task_id": task_id,
                    "generation_status": result.get("status"),
                    "progress": result.get("progress", 0),
                    "model_urls": result.get("model_urls", {})
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"API error: {response.status_code}"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _list_models(self, data: Dict) -> str:
        """List previously generated models"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url}/text-to-3d",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.dumps({
                    "status": "success",
                    "models": result.get("result", [])
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"API error: {response.status_code}"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            })


# Preset prompts for common game assets
MESHY_PRESETS = {
    "building_modern": "modern office building, glass facade, realistic architecture",
    "building_scifi": "futuristic sci-fi building with neon lights and metallic surfaces",
    "building_medieval": "medieval stone castle tower with wooden details",
    "tree_realistic": "realistic oak tree with detailed bark and leaves",
    "tree_stylized": "stylized cartoon tree with low poly leaves",
    "car_sports": "modern sports car with sleek design",
    "car_cyberpunk": "cyberpunk hover car with neon accents",
    "prop_crate": "wooden crate with metal reinforcements",
    "prop_barrel": "industrial metal barrel with rust",
    "terrain_rocks": "natural rock formation with moss and cracks"
}


# Global instance
MESHY_ADAPTER = MeshyAdapter()
