#!/usr/bin/env python3
"""
PlayCanvas CLI Tool
Command-line interface for PlayCanvas API operations.

Usage:
    python playcanvas_cli.py list-projects
    python playcanvas_cli.py create-project "My Game"
    python playcanvas_cli.py upload-script <project_id> <script_path>
    python playcanvas_cli.py deploy-aether-bridge <project_id>
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from body.adapters.playcanvas_adapter import PlayCanvasAdapter


async def main():
    adapter = PlayCanvasAdapter()
    
    if not adapter.access_token:
        print("❌ Error: PLAYCANVAS_API_TOKEN not set in .env")
        print("\n📝 Setup Instructions:")
        print("1. Go to https://playcanvas.com/account")
        print("2. Scroll to 'API Tokens' section")
        print("3. Click 'Generate Token'")
        print("4. Copy token and add to .env:")
        print("   PLAYCANVAS_API_TOKEN=your_token_here")
        return
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1]
    
    if command == "list-projects" or command == "list":
        print("\n⚠️  PlayCanvas API doesn't support listing all projects.")
        print("\n📝 To use the API, you need a project ID from PlayCanvas Editor:")
        print("   1. Go to https://playcanvas.com/projects")
        print("   2. Open any project")
        print("   3. Copy ID from URL: playcanvas.com/project/[THIS_IS_THE_ID]/overview")
        print("\n💡 Then use: python playcanvas_cli.py get-project <project_id>")
        print("            python playcanvas_cli.py deploy-aether-bridge <project_id>\n")
        return
    
    elif command == "create-project" or command == "create":
        if len(sys.argv) < 3:
            print("❌ Error: Project name required")
            print("Usage: python playcanvas_cli.py create-project 'My Game'")
            return
        
        name = sys.argv[2]
        description = sys.argv[3] if len(sys.argv) > 3 else "Created via AetherMind CLI"
        
        intent = json.dumps({
            "action": "create_project",
            "params": {
                "name": name,
                "description": description
            }
        })
        
        print(f"🚀 Creating project: {name}...")
        result = await adapter.execute(intent)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            print(f"\n✅ Project created successfully!")
            print(f"   ID: {data['project_id']}")
            print(f"   Name: {data['name']}")
            print(f"   URL: {data['url']}\n")
    
    elif command == "upload-script" or command == "upload":
        if len(sys.argv) < 4:
            print("❌ Error: Project ID and script path required")
            print("Usage: python playcanvas_cli.py upload-script <project_id> <script_path>")
            return
        
        project_id = sys.argv[2]
        script_path = sys.argv[3]
        
        if not os.path.exists(script_path):
            print(f"❌ Error: Script not found: {script_path}")
            return
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        script_name = os.path.basename(script_path)
        
        intent = json.dumps({
            "action": "upload_script",
            "params": {
                "project_id": int(project_id),
                "branch_id": "master",
                "name": script_name,
                "content": content
            }
        })
        
        print(f"📤 Uploading {script_name} to project {project_id}...")
        result = await adapter.execute(intent)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            print(f"\n✅ Script uploaded successfully!")
            print(f"   Asset ID: {data['asset_id']}")
            print(f"   Name: {data['name']}\n")
    
    elif command == "deploy-aether-bridge" or command == "deploy":
        if len(sys.argv) < 3:
            print("❌ Error: Project ID required")
            print("Usage: python playcanvas_cli.py deploy-aether-bridge <project_id>")
            return
        
        project_id = sys.argv[2]
        
        # Read the aether bridge script
        bridge_path = Path(__file__).parent.parent / "playcanvas_aether_bridge.js"
        
        if not bridge_path.exists():
            print(f"❌ Error: Bridge script not found at {bridge_path}")
            return
        
        with open(bridge_path, 'r') as f:
            content = f.read()
        
        intent = json.dumps({
            "action": "upload_script",
            "params": {
                "project_id": int(project_id),
                "branch_id": "master",
                "name": "aetherBridge.js",
                "content": content
            }
        })
        
        print(f"🧠 Deploying AetherMind Bridge to project {project_id}...")
        result = await adapter.execute(intent)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            print(f"\n✅ AetherMind Bridge deployed successfully!")
            print(f"   Asset ID: {data['asset_id']}")
            print(f"\n📝 Next steps:")
            print(f"   1. Open project: https://playcanvas.com/project/{project_id}/overview")
            print(f"   2. Create/select an entity (e.g., GameManager)")
            print(f"   3. Add Script component")
            print(f"   4. Select 'aetherBridge' script")
            print(f"   5. Configure API URL: http://localhost:8000/v1/game/unity/state")
            print(f"   6. Launch your game!\n")
    
    elif command == "get-project" or command == "info":
        if len(sys.argv) < 3:
            print("❌ Error: Project ID required")
            print("Usage: python playcanvas_cli.py get-project <project_id>")
            return
        
        project_id = sys.argv[2]
        
        intent = json.dumps({
            "action": "get_project",
            "params": {"project_id": int(project_id)}
        })
        
        result = await adapter.execute(intent)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            project = data["project"]
            print(f"\n📦 Project Details:\n")
            print(f"   ID: {project['id']}")
            print(f"   Name: {project['name']}")
            print(f"   Created: {project.get('createdAt')}")
            print(f"   Modified: {project.get('modifiedAt')}")
            print(f"   URL: https://playcanvas.com/project/{project['id']}/overview\n")
    
    elif command == "list-scenes" or command == "scenes":
        if len(sys.argv) < 3:
            print("❌ Error: Project ID required")
            print("Usage: python playcanvas_cli.py list-scenes <project_id>")
            return
        
        project_id = sys.argv[2]
        
        intent = json.dumps({
            "action": "list_scenes",
            "params": {"project_id": int(project_id)}
        })
        
        result = await adapter.execute(intent)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            print(f"\n🎬 Found {data['count']} scene(s):\n")
            for scene in data["scenes"]:
                print(f"   • {scene.get('name', 'Untitled')}")
                print(f"     ID: {scene['id']}\n")
    
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()


def print_usage():
    print("""
🎮 PlayCanvas CLI - AetherMind Integration

Commands:
  list-projects              List all your PlayCanvas projects
  create-project <name>      Create a new project
  upload-script <id> <path>  Upload a script to a project
  deploy-aether-bridge <id>  Deploy AetherMind bridge to project
  get-project <id>           Get project details
  list-scenes <id>           List scenes in a project

Examples:
  python playcanvas_cli.py list-projects
  python playcanvas_cli.py create-project "My Game"
  python playcanvas_cli.py deploy-aether-bridge 123456
  python playcanvas_cli.py upload-script 123456 ./my_script.js

Setup:
  1. Get API token from https://playcanvas.com/account
  2. Add to .env: PLAYCANVAS_API_TOKEN=your_token_here
  3. Run commands above
""")


if __name__ == "__main__":
    asyncio.run(main())
