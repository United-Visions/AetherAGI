"""
PlayCanvas API Adapter for AetherMind
Programmatically create and manage PlayCanvas projects via REST API.

Features:
- Create projects from templates
- Upload scripts and assets
- Build and publish apps
- Manage scenes and branches
- Integrate with ToolForge for dynamic game generation
"""

import os
import json
import requests
from typing import Dict, List, Optional
from body.adapter_base import BodyAdapter


class PlayCanvasAdapter(BodyAdapter):
    """
    Adapter for PlayCanvas REST API integration.
    Allows AetherMind to programmatically create and manage game projects.
    """
    
    def __init__(self):
        self.base_url = "https://playcanvas.com/api"
        self.access_token = os.getenv("PLAYCANVAS_API_TOKEN")
        
        if not self.access_token:
            print("⚠️ PLAYCANVAS_API_TOKEN not set in .env - API features disabled")
        
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def execute(self, intent: str) -> str:
        """
        Execute PlayCanvas operations based on intent.
        
        Intent format (JSON):
        {
            "action": "create_project|upload_script|list_projects|get_project|download_app",
            "params": {...}
        }
        """
        try:
            intent_data = json.loads(intent)
            action = intent_data.get("action")
            params = intent_data.get("params", {})
            
            if action == "upload_script":
                return await self.upload_script(params)

            elif action == "get_project":
                return await self.get_project(params)
            elif action == "list_scenes":
                return await self.list_scenes(params)
            elif action == "download_app":
                return await self.download_app(params)
            elif action == "create_asset":
                return await self.create_asset(params)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
                
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON intent"})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def create_project(self, params: Dict) -> str:
        """
        Create a new PlayCanvas project.
        
        Params:
        - name: Project name
        - description: Project description (optional)
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        project_data = {
            "name": params.get("name", "AetherMind Generated Project"),
            "description": params.get("description", "Created by AetherMind")
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/projects",
                headers=self.headers,
                json=project_data,
                timeout=10
            )
            
            if response.status_code == 201:
                project = response.json()
                return json.dumps({
                    "status": "success",
                    "project_id": project["id"],
                    "name": project["name"],
                    "url": f"https://playcanvas.com/project/{project['id']}/overview"
                })
            else:
                return json.dumps({
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def upload_script(self, params: Dict) -> str:
        """
        Upload a script to a PlayCanvas project.
        
        Params:
        - project_id: Target project ID
        - branch_id: Branch ID (usually 'master')
        - name: Script name
        - content: JavaScript code content
        - parent: Parent asset ID (optional)
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        # Prepare multipart form data
        files = {
            'name': (None, params.get("name", "aetherScript.js")),
            'projectId': (None, str(params.get("project_id"))),
            'branchId': (None, params.get("branch_id", "master")),
            'preload': (None, 'true'),
            'file': (params.get("name", "script.js"), params.get("content", ""), 'application/javascript')
        }
        
        if "parent" in params:
            files['parent'] = (None, str(params["parent"]))
        
        try:
            # Override headers for multipart
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            response = requests.post(
                f"{self.base_url}/assets",
                headers=headers,
                files=files,
                timeout=15
            )
            
            if response.status_code == 201:
                asset = response.json()
                return json.dumps({
                    "status": "success",
                    "asset_id": asset["id"],
                    "name": asset["name"],
                    "url": asset.get("file", {}).get("url", "")
                })
            else:
                return json.dumps({
                    "error": f"Upload failed: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def list_projects(self) -> str:
        """List all PlayCanvas projects for the authenticated user."""
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        try:
            response = requests.get(
                f"{self.base_url}/projects",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                projects = data.get("result", [])
                
                return json.dumps({
                    "status": "success",
                    "count": len(projects),
                    "projects": [
                        {
                            "id": p["id"],
                            "name": p["name"],
                            "created_at": p.get("createdAt"),
                            "modified_at": p.get("modifiedAt")
                        }
                        for p in projects
                    ]
                })
            else:
                return json.dumps({
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def get_project(self, params: Dict) -> str:
        """
        Get details of a specific project.
        
        Params:
        - project_id: Project ID
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        project_id = params.get("project_id")
        if not project_id:
            return json.dumps({"error": "project_id required"})
        
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                project = response.json()
                return json.dumps({
                    "status": "success",
                    "project": project
                })
            else:
                return json.dumps({
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def list_scenes(self, params: Dict) -> str:
        """
        List all scenes in a project.
        
        Params:
        - project_id: Project ID
        - branch_id: Branch ID (optional, defaults to master)
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        project_id = params.get("project_id")
        branch_id = params.get("branch_id", "master")
        
        if not project_id:
            return json.dumps({"error": "project_id required"})
        
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/scenes?branchId={branch_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                scenes = data.get("result", [])
                
                return json.dumps({
                    "status": "success",
                    "count": len(scenes),
                    "scenes": scenes
                })
            else:
                return json.dumps({
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def download_app(self, params: Dict) -> str:
        """
        Download a built app from PlayCanvas.
        
        Params:
        - project_id: Project ID
        - branch_id: Branch ID
        - name: Build name
        - scenes: List of scene IDs to include
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        project_id = params.get("project_id")
        if not project_id:
            return json.dumps({"error": "project_id required"})
        
        build_data = {
            "project_id": project_id,
            "name": params.get("name", "AetherMind Build"),
            "scenes": params.get("scenes", []),
            "branch_id": params.get("branch_id", "master"),
            "description": params.get("description", ""),
            "preload_bundle": params.get("preload_bundle", False),
            "optimization_options": {
                "concatenate": True,
                "minify": True,
                "sourcemaps": False
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/apps/download",
                headers=self.headers,
                json=build_data,
                timeout=30
            )
            
            if response.status_code == 201:
                download = response.json()
                return json.dumps({
                    "status": "success",
                    "download_url": download.get("download_url"),
                    "created_at": download.get("created_at")
                })
            else:
                return json.dumps({
                    "error": f"Build failed: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def create_asset(self, params: Dict) -> str:
        """
        Create a new asset (script, JSON, HTML, CSS, shader, text).
        
        Params:
        - project_id: Target project
        - branch_id: Target branch
        - name: Asset name
        - type: Asset type (script, json, html, css, shader, text)
        - content: Asset content
        """
        if not self.access_token:
            return json.dumps({"error": "API token not configured"})
        
        asset_type = params.get("type", "script")
        content_type_map = {
            "script": "application/javascript",
            "json": "application/json",
            "html": "text/html",
            "css": "text/css",
            "shader": "text/plain",
            "text": "text/plain"
        }
        
        files = {
            'name': (None, params.get("name", f"asset.{asset_type}")),
            'projectId': (None, str(params.get("project_id"))),
            'branchId': (None, params.get("branch_id", "master")),
            'preload': (None, str(params.get("preload", True)).lower()),
            'file': (
                params.get("name", f"file.{asset_type}"),
                params.get("content", ""),
                content_type_map.get(asset_type, "text/plain")
            )
        }
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.post(
                f"{self.base_url}/assets",
                headers=headers,
                files=files,
                timeout=15
            )
            
            if response.status_code == 201:
                asset = response.json()
                return json.dumps({
                    "status": "success",
                    "asset_id": asset["id"],
                    "name": asset["name"],
                    "type": asset["type"],
                    "state": asset["state"]
                })
            else:
                return json.dumps({
                    "error": f"Asset creation failed: {response.status_code}",
                    "message": response.text
                })
                
        except Exception as e:
            return json.dumps({"error": str(e)})


# Global instance for API access
PLAYCANVAS_ADAPTER = PlayCanvasAdapter()
