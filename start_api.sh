#!/bin/bash

# AetherMind FastAPI Backend - Quick Start Script

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║      AetherMind FastAPI Backend - Quick Start            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found${NC}"
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check if requirements are installed
echo -e "${BLUE}Checking dependencies...${NC}"
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Dependencies not installed${NC}"
    echo -e "${BLUE}Installing requirements...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env file not found${NC}"
    echo -e "${YELLOW}Creating .env template...${NC}"
    cat > .env << 'EOF'
# AetherMind API Configuration

# Required API Keys
PINECONE_API_KEY=your_pinecone_key_here
RUNPOD_API_KEY=your_runpod_key_here
ADMIN_SECRET=your_admin_secret_here

# Optional Services
FIRECRAWL_API_KEY=your_firecrawl_key_here
PERCEPTION_SERVICE_URL=http://localhost:8001

# Database (Optional)
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_key_here
EOF
    echo -e "${GREEN}✓ .env template created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file with your actual API keys${NC}"
    echo -e "${BLUE}Opening .env for editing...${NC}"
    sleep 2
    ${EDITOR:-nano} .env
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if running in orchestrator directory
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "orchestrator" ]; then
    echo -e "${BLUE}Changing to orchestrator directory...${NC}"
    cd orchestrator
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}⚠ Port 8000 is already in use${NC}"
    echo -e "${BLUE}Finding alternative port...${NC}"
    PORT=8001
else
    PORT=8000
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting AetherMind FastAPI Backend on port $PORT${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Available endpoints:${NC}"
echo -e "  • http://localhost:$PORT/docs - Interactive API docs"
echo -e "  • http://localhost:$PORT/redoc - ReDoc documentation"
echo -e "  • http://localhost:$PORT/v1/chat - SDK chat endpoint"
echo -e "  • http://localhost:$PORT/v1/memory/search - Memory search"
echo -e "  • http://localhost:$PORT/v1/tools/create - ToolForge"
echo -e "  • http://localhost:$PORT/v1/usage - Usage stats"
echo -e "  • http://localhost:$PORT/v1/namespaces - List namespaces"
echo -e "  • http://localhost:$PORT/v1/knowledge/cartridge - Knowledge cartridges"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start FastAPI with uvicorn
uvicorn main_api:app --reload --host 0.0.0.0 --port $PORT
