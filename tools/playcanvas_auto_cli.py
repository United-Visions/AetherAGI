#!/usr/bin/env python3
"""
PlayCanvas Auto-Setup CLI
Controls PlayCanvas editor through browser automation
"""

import asyncio
import requests
import sys

BASE_URL = "http://localhost:8001"


def start_server():
    """Instructions to start the automation server"""
    print("🚀 Starting PlayCanvas Automation Server...")
    print("\nRun this in another terminal:")
    print("  python tools/playcanvas_automation_server.py")
    print("\nThen come back here and run your commands!")


def deploy_aether_bridge(project_id: int):
    """Fully automated deployment"""
    print(f"🧠 Deploying AetherBridge to project {project_id}...")
    print("\n⚠️  Make sure the automation server is running!")
    print("    python tools/playcanvas_automation_server.py")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()['service']}\n")
    except:
        print("❌ Server not running! Start it first:")
        print("    python tools/playcanvas_automation_server.py\n")
        return
    
    print("📋 Steps that will happen:")
    print("  1. Browser will open (you'll see it!)")
    print("  2. Login to PlayCanvas")
    print("  3. Open project editor")
    print("  4. Create AetherBridge script")
    print("  5. Create GameManager entity")
    print("  6. Attach script with config")
    print("\n🔑 Auto-login configured")
    # print("Press ENTER when ready...")
    # input()
    
    # Get login credentials
    username = "united-visions"
    password = "Mc1417182613."
    
    print(f"\n🌐 Opening browser and logging in as {username}...")
    response = requests.post(f"{BASE_URL}/login", json={
        "username": username,
        "password": password
    })
    
    if response.status_code != 200:
        try:
            error_json = response.json()
            print(f"❌ Login failed: {error_json}")
        except requests.exceptions.JSONDecodeError:
            print(f"❌ Login failed with non-JSON response (Status {response.status_code}):")
            print(response.text[:500]) # Print first 500 chars of error
        print("\n💡 Check 'playcanvas_automation.log' for detailed error logs and screenshots.")
        return
    
    print("✅ Logged in!")
    
    # Deploy
    print(f"\n🚀 Deploying to project {project_id}...")
    response = requests.post(f"{BASE_URL}/deploy-aether-bridge/{project_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ {result['message']}")
        print("\n📝 Next steps:")
        for step in result['next_steps']:
            print(f"   • {step}")
    else:
        print(f"❌ Deployment failed: {response.json()}")
        print("\n💡 Check 'playcanvas_automation.log' for detailed debugging info.")


def main():
    if len(sys.argv) < 2:
        print("""
🎮 PlayCanvas Automation CLI

Commands:
  start-server                Start the automation server
  deploy <project_id>         Deploy AetherBridge to project

Examples:
  python tools/playcanvas_auto_cli.py start-server
  python tools/playcanvas_auto_cli.py deploy 1449261
""")
        return
    
    command = sys.argv[1]
    
    if command == "start-server":
        start_server()
    elif command == "deploy":
        if len(sys.argv) < 3:
            print("❌ Error: Project ID required")
            print("Usage: python tools/playcanvas_auto_cli.py deploy <project_id>")
            return
        project_id = int(sys.argv[2])
        deploy_aether_bridge(project_id)
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
