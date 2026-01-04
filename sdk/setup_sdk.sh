#!/bin/bash

# AetherMind SDK Quick Setup Script
# Builds and tests all SDKs locally before distribution

set -e  # Exit on error

echo "🚀 AetherMind SDK Setup & Distribution"
echo "======================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check dependencies
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found. Please install it first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $1 found${NC}"
}

echo "Checking dependencies..."
check_command python3
check_command node
check_command npm

echo ""
echo "========================================="
echo "🐍 Building Python SDK"
echo "========================================="

cd sdk/python

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install build twine

# Build
echo "Building package..."
python -m build

# Test install
echo "Testing local install..."
pip install dist/aethermind-*.whl

# Verify
python -c "from aethermind import AetherMindClient; print('✓ Python SDK works!')"

echo -e "${GREEN}✓ Python SDK built successfully${NC}"
echo ""
echo "To publish to PyPI:"
echo "  cd sdk/python"
echo "  twine upload --repository testpypi dist/*  # Test first"
echo "  twine upload dist/*  # Production"

deactivate
cd ../..

echo ""
echo "========================================="
echo "📦 Building JavaScript SDK"
echo "========================================="

cd sdk/javascript

# Install dependencies
npm install

# Build TypeScript
npm run build

# Test
echo "Testing build..."
node -e "const {AetherMindClient} = require('./dist/index.js'); console.log('✓ JS SDK works!')"

echo -e "${GREEN}✓ JavaScript SDK built successfully${NC}"
echo ""
echo "To publish to npm:"
echo "  cd sdk/javascript"
echo "  npm login"
echo "  npm publish --access public"

cd ../..

echo ""
echo "========================================="
echo "✨ Setup Complete!"
echo "========================================="
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Get API Key:"
echo "   → Sign up at https://aethermind.ai"
echo "   → Create API key in dashboard"
echo ""
echo "2. Test SDKs:"
echo ""
echo "   Python:"
echo "   $ pip install aethermind"
echo "   $ export AETHERMIND_API_KEY=am_live_your_key"
echo "   $ python -c 'from aethermind import AetherMindClient; client = AetherMindClient(); print(client.chat({\"message\": \"Hello!\"}))'"
echo ""
echo "   JavaScript:"
echo "   $ npm install @aethermind/sdk"
echo "   $ export AETHERMIND_API_KEY=am_live_your_key"
echo "   $ node -e 'const {AetherMindClient}=require(\"@aethermind/sdk\");const c=new AetherMindClient({apiKey:process.env.AETHERMIND_API_KEY});c.chat({message:\"Hello\"}).then(console.log)'"
echo ""
echo "3. View Documentation:"
echo "   → https://aethermind.ai/documentation"
echo "   → https://github.com/United-Visions/AetherAGI"
echo ""
echo "4. Join Community:"
echo "   → Discord: https://discord.gg/aethermind"
echo "   → Email: dev@aethermind.ai"
echo ""
echo -e "${GREEN}Happy Building! 🚀${NC}"
