#!/bin/bash
# Test automation server and deploy AetherBridge

echo "🧪 Testing automation server..."

# Check if server is responding
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/)

if [ "$response" = "200" ]; then
    echo "✅ Automation server is running!"
    echo ""
    echo "🚀 Starting deployment..."
    echo ""
    
    # Run deployment
    python tools/playcanvas_auto_cli.py deploy 1449261
else
    echo "❌ Server not responding (HTTP $response)"
    echo "   Check automation_server.log for errors"
    echo ""
    echo "   Start server with:"
    echo "   python tools/playcanvas_automation_server.py"
fi
