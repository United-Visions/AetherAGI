"""
SketchFab Adapter for AetherMind
Search and download 3D models from SketchFab library

Features:
- Search 500k+ free 3D models
- Filter by license (CC, downloadable)
- Download GLTF/GLB models
- Auto-upload to PlayCanvas

API Docs: https://docs.sketchfab.com/data-api

Example Usage:
User: "Find a car model for my game"
Brain: <aether-sketchfab action="search">
{
  "query": "sports car",
  "downloadable": true,
  "license": "cc"
}
</aether-sketchfab>
"""

import os
import json
import requests
from typing import Dict, List, Optional
from body.adapter_base import BodyAdapter
from loguru import logger


class SketchfabAdapter(BodyAdapter):
    """
    Adapter for SketchFab 3D model library API.
    Search and download free CC-licensed models.
    """
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SKETCHFAB_API_KEY", "")  # Optional for public search
        self.base_url = "https://api.sketchfab.com/v3"
        self.download_dir = "/tmp/aethermind_sketchfab"
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
        logger.info("🗿 SketchFab Adapter initialized")
    
    async def execute(self, intent: str) -> str:
        """
        Main entry point from Brain.
        
        Intent format:
        {
            "action": "search|download|get_model",
            "query": "sports car",
            "downloadable": true,
            "license": "cc|cc-by|cc0",
            "model_id": "abc123",
            "max_results": 10
        }
        """
        try:
            data = json.loads(intent)
            action = data.get("action", "search")
            
            # Route to action handler
            if action == "search":
                return await self._search_models(data)
            elif action == "download":
                return await self._download_model(data)
            elif action == "get_model":
                return await self._get_model_details(data)
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
            logger.error(f"SketchFab adapter error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _search_models(self, data: Dict) -> str:
        """
        Search SketchFab library.
        
        Returns list of downloadable models matching query.
        """
        query = data.get("query", "")
        downloadable = data.get("downloadable", True)
        license_filter = data.get("license", "")  # cc, cc-by, cc0
        max_results = data.get("max_results", 10)
        
        try:
            # Build search params
            params = {
                "type": "models",
                "q": query,
                "downloadable": str(downloadable).lower(),
                "count": min(max_results, 24),  # SketchFab max per page
                "sort_by": "-likeCount"  # Most popular first
            }
            
            # Add license filter if specified
            if license_filter:
                params["licenses"] = license_filter
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            logger.info(f"🔍 Searching SketchFab: {query}")
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                return json.dumps({
                    "status": "error",
                    "message": f"SketchFab API error: {response.status_code} - {response.text}"
                })
            
            result = response.json()
            models = []
            
            for item in result.get("results", []):
                models.append({
                    "id": item.get("uid"),
                    "name": item.get("name"),
                    "author": item.get("user", {}).get("displayName"),
                    "license": item.get("license", {}).get("label"),
                    "thumbnail": item.get("thumbnails", {}).get("images", [{}])[0].get("url"),
                    "view_url": item.get("viewerUrl"),
                    "downloadable": item.get("isDownloadable", False),
                    "likes": item.get("likeCount", 0),
                    "face_count": item.get("faceCount", 0)
                })
            
            logger.info(f"✅ Found {len(models)} models")
            
            return json.dumps({
                "status": "success",
                "query": query,
                "count": len(models),
                "models": models
            })
            
        except Exception as e:
            logger.error(f"SketchFab search failed: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Search error: {e}"
            })
    
    async def _download_model(self, data: Dict) -> str:
        """
        Download a model by ID.
        
        Requires SketchFab API key for download access.
        """
        model_id = data.get("model_id")
        
        if not model_id:
            return json.dumps({
                "status": "error",
                "message": "model_id required for download"
            })
        
        if not self.api_key:
            return json.dumps({
                "status": "error",
                "message": "SketchFab API key required for downloads. Set SKETCHFAB_API_KEY in .env"
            })
        
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            # Step 1: Get download URL
            logger.info(f"⬇️ Requesting download for model: {model_id}")
            response = requests.get(
                f"{self.base_url}/models/{model_id}/download",
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                return json.dumps({
                    "status": "error",
                    "message": f"Download request failed: {response.status_code}"
                })
            
            download_data = response.json()
            
            # Find GLTF download (preferred for games)
            gltf_url = None
            for fmt in download_data.get("gltf", {}).values():
                if isinstance(fmt, dict) and "url" in fmt:
                    gltf_url = fmt["url"]
                    break
            
            if not gltf_url:
                return json.dumps({
                    "status": "error",
                    "message": "No GLTF export available for this model"
                })
            
            # Step 2: Download the file
            logger.info(f"📥 Downloading from: {gltf_url}")
            download_response = requests.get(gltf_url, timeout=120)
            download_response.raise_for_status()
            
            # Save file
            filename = f"{model_id}.zip"  # SketchFab downloads as ZIP
            filepath = os.path.join(self.download_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(download_response.content)
            
            logger.info(f"✅ Downloaded: {filename}")
            
            return json.dumps({
                "status": "success",
                "message": f"Downloaded model {model_id}",
                "model_id": model_id,
                "file": filepath,
                "format": "gltf",
                "download_dir": self.download_dir,
                "note": "ZIP contains GLTF + textures - extract before importing"
            })
            
        except Exception as e:
            logger.error(f"SketchFab download failed: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Download error: {e}"
            })
    
    async def _get_model_details(self, data: Dict) -> str:
        """Get detailed info about a specific model"""
        model_id = data.get("model_id")
        
        if not model_id:
            return json.dumps({
                "status": "error",
                "message": "model_id required"
            })
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            
            response = requests.get(
                f"{self.base_url}/models/{model_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                model = response.json()
                return json.dumps({
                    "status": "success",
                    "model": {
                        "id": model.get("uid"),
                        "name": model.get("name"),
                        "description": model.get("description"),
                        "author": model.get("user", {}).get("displayName"),
                        "license": model.get("license", {}).get("label"),
                        "downloadable": model.get("isDownloadable"),
                        "face_count": model.get("faceCount"),
                        "vertex_count": model.get("vertexCount"),
                        "thumbnail": model.get("thumbnails", {}).get("images", [{}])[0].get("url"),
                        "tags": [tag.get("name") for tag in model.get("tags", [])]
                    }
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


# Popular search presets
SKETCHFAB_PRESETS = {
    "vehicles": "downloadable car sports vehicle",
    "buildings": "downloadable building architecture",
    "nature": "downloadable tree plant nature",
    "characters": "downloadable character rigged",
    "props": "downloadable props game asset",
    "furniture": "downloadable furniture interior",
    "weapons": "downloadable weapon gun sword",
    "scifi": "downloadable sci-fi futuristic",
    "medieval": "downloadable medieval fantasy"
}


# Global instance
SKETCHFAB_ADAPTER = SketchfabAdapter()
