import os
import sys
import yaml
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE any imports that depend on them
# Load environment variables from parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Ensure we can import orchestrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask import Flask
from quart import Quart, render_template, request, redirect, url_for, session, jsonify
import logging
import httpx
import uuid
import markdown
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
from orchestrator.auth_manager_supabase import AuthManagerSupabase, UserRole
from brain.jepa_aligner import JEPAAligner
import numpy as np

# Load settings for pilot users
def load_settings():
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    try:
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {'pilot_users': []}

SETTINGS = load_settings()
PILOT_USERS = SETTINGS.get('pilot_users', [])

app = Quart(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-change-me")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

auth_mgr = AuthManagerSupabase()

# --- AETHER BRAIN (JEPA) ---
# Initialize the Joint Embedding Predictive Architecture for vehicle control
# We use a small dimension (16) for the simple vehicle telemetry vector
VEHICLE_DIM = 16
jepa_brain = JEPAAligner(dimension=VEHICLE_DIM, energy_threshold=0.3)
last_vehicle_state = np.zeros(VEHICLE_DIM)

# --- GitHub OAuth ---
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Dynamically select redirect URI based on environment
# Check if running on localhost/development vs production
def is_development():
    """Detect if running in development environment"""
    # Check for common development indicators
    return (
        os.getenv("FLASK_ENV") == "development" or
        os.getenv("FLASK_DEBUG") == "1" or
        app.debug or
        "127.0.0.1" in request.host if request else True
    )

# Select appropriate redirect URI
GITHUB_REDIRECT_URI_PROD = os.getenv("GITHUB_REDIRECT_URI", "https://aethermind-frontend.onrender.com/callback")
GITHUB_REDIRECT_URI_DEV  = os.getenv("GITHUB_REDIRECT_URI_DEV", "http://127.0.0.1:5000/callback")

# This will be set dynamically per request
GITHUB_REDIRECT_URI = GITHUB_REDIRECT_URI_DEV  # Default for initialization

FERNET_KEY           = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    FERNET_KEY = Fernet.generate_key()
# Ensure key is bytes if it's a string from env
if isinstance(FERNET_KEY, str):
     FERNET_KEY = FERNET_KEY.encode()
cipher = Fernet(FERNET_KEY)

@app.before_request
def log_request_info():
    app.logger.info(f"Incoming Request: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    app.logger.info(f"Outgoing Response: {response.status_code} {request.url}")
    return response

@app.route("/")
async def home():
    return await render_template("index_home.html")

@app.route("/pricing")
async def pricing():
    return await render_template("pricing.html")

@app.route("/documentation")
async def documentation():
    return await render_template("documentation.html")

@app.route("/whitepaper")
async def whitepaper():
    md_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'AETHER_WHITE_PAPER.md')
    with open(md_path, 'r') as f:
        md_content = f.read()
    
    html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    return await render_template("whitepaper.html", content=html_content)

@app.route("/domain/legal")
async def domain_legal():
    return await render_template("domain_legal.html")

@app.route("/domain/medical")
async def domain_medical():
    # TODO: Create domain_medical.html
    return await render_template("domain_legal.html")  # Placeholder

@app.route("/domain/finance")
async def domain_finance():
    # TODO: Create domain_finance.html
    return await render_template("domain_legal.html")  # Placeholder

@app.route("/domain/code")
async def domain_code():
    # TODO: Create domain_code.html
    return await render_template("domain_legal.html")  # Placeholder

@app.route("/domain/research")
async def domain_research():
    # TODO: Create domain_research.html
    return await render_template("domain_legal.html")  # Placeholder

@app.route("/github_login")
async def github_login():
    # Dynamically select redirect URI based on request host
    is_local = "127.0.0.1" in request.host or "localhost" in request.host
    redirect_uri = GITHUB_REDIRECT_URI_DEV if is_local else GITHUB_REDIRECT_URI_PROD
    
    app.logger.info(f"GitHub OAuth: Using redirect URI: {redirect_uri} (local={is_local})")
    
    scope = "read:user repo invite"   # repo scope needed for ToolForge
    url = (f"https://github.com/login/oauth/authorize"
           f"?client_id={GITHUB_CLIENT_ID}&redirect_uri={quote_plus(redirect_uri)}"
           f"&scope={quote_plus(scope)}")
    return redirect(url)

@app.route("/callback")
async def callback():
    code = request.args.get("code")
    if not code:
        return "Missing code", 400
    # exchange code for token
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post("https://github.com/login/oauth/access_token",
                           data={"client_id": GITHUB_CLIENT_ID,
                                 "client_secret": GITHUB_CLIENT_SECRET,
                                 "code": code},
                           headers={"Accept": "application/json"})
            if r.status_code != 200:
                return "OAuth failed", 400
            token = r.json().get("access_token")
            if not token:
                return f"OAuth failed: {r.text}", 400

            # fetch user info
            user_r = await client.get("https://api.github.com/user",
                               headers={"Authorization": f"token {token}"})
            if user_r.status_code != 200:
                return "Failed to fetch user", 400

        github_user = user_r.json()["login"]
        github_url = user_r.json()["html_url"]
        github_avatar = user_r.json().get("avatar_url")
        github_email = user_r.json().get("email")
        
        # encrypt & store token
        encrypted = cipher.encrypt(token.encode()).decode()
        session["github_user"] = github_user
        session["github_url"] = github_url
        session["github_avatar"] = github_avatar
        session["github_email"] = github_email
        session["github_token"]  = encrypted          # stored only in session cookie
        return redirect(url_for("onboarding"))
    except Exception as e:
        app.logger.error(f"OAuth Error: {e}")
        return f"OAuth Error: {e}", 500

@app.route("/onboarding")
async def onboarding():
    if "github_user" not in session:
        return redirect(url_for("home"))
    return await render_template("onboarding.html", github_user=session["github_user"])

@app.route("/create_key", methods=["GET", "POST"])
async def create_key():
    """
    Generate a personal API key for authenticated user.
    GET: Show key creation form
    POST: Generate and return key
    
    Architecture:
    - Backend service keys (PINECONE_API_KEY, RUNPOD_API_KEY) are in .env
    - User personal keys (am_live_XXX) are generated here and hashed in keys.json
    - Each user gets their own key with role-based permissions (FREE/PRO/ENTERPRISE)
    """
    if "github_user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    # Handle GET request - show form
    if request.method == "GET":
        return await render_template("create_key.html", github_user=session["github_user"])
    
    # Handle POST request - generate key
    user_id = session["github_user"]
    
    # Get domain and subscription tier from form or JSON
    if request.is_json:
        data = await request.get_json()
        domain = data.get("domain", "general")
        tier = data.get("tier", "pro")  # Default to pro for now
    else:
        form_data = await request.form
        domain = form_data.get("domain", "general")
        tier = form_data.get("tier", "pro")
    
    session["user_domain"] = domain  # Store in session
    
    # Always generate a unique API key for each user
    # Keys are hashed and stored in Supabase with GitHub info
    from orchestrator.auth_manager_supabase import UserRole
    
    # Get GitHub user info from session
    github_username = user_id  # user_id is the GitHub username
    github_url = session.get("github_url", f"https://github.com/{github_username}")
    
    # Map tier to role
    role_mapping = {
        "free": UserRole.FREE,
        "pro": UserRole.PRO,
        "enterprise": UserRole.ENTERPRISE
    }
    role = role_mapping.get(tier.lower(), UserRole.PRO)
    
    key = auth_mgr.generate_api_key(
        user_id=user_id,
        github_username=github_username,
        github_url=github_url,
        role=role
    )
    app.logger.info(f"Generated new API key for user {user_id} with role {role.value}")
    
    # Register domain preference with backend API (if available)
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        response = httpx.post(
            f"{backend_url}/v1/user/domain",
            json={"user_id": user_id, "domain": domain},
            headers={"Authorization": f"ApiKey {key}"},
            timeout=5.0
        )
        if response.status_code == 200:
            app.logger.info(f"Domain {domain} registered for user {user_id}")
        else:
            app.logger.warning(f"Failed to register domain: {response.text}")
    except Exception as e:
        app.logger.error(f"Could not reach backend to register domain: {e}")
        # Continue anyway - domain will default to general
    
    # Return JSON if requested, otherwise redirect
    if request.is_json or request.headers.get('Accept') == 'application/json':
        return jsonify({
            "api_key": key,
            "domain": domain,
            "user_id": user_id
        })
    else:
        # Pass key to chat page via query-param (client stores in localStorage)
        return redirect(url_for("index", api_key=key, domain=domain))

# --- existing chat page ---
@app.route("/chat")
async def index():
    # if api_key in query string, preload it into the page
    api_key = request.args.get("api_key", "")
    domain = request.args.get("domain", session.get("user_domain", "general"))
    user_id = session.get("github_user")
    
    # Check if user is a pilot user (admin)
    is_pilot = user_id in PILOT_USERS if user_id else False
    
    # Domain-specific welcome messages
    domain_messages = {
        "code": "Ready to build. I'm your Software Development Specialist.",
        "research": "Ready to analyze. I'm your Research & Analysis Specialist.",
        "business": "Ready to strategize. I'm your Business & Strategy Specialist.",
        "legal": "Ready to research. I'm your Legal Research Specialist.",
        "finance": "Ready to model. I'm your Finance & Investment Specialist.",
        "general": "Ready to assist. I'm your Multi-Domain Master."
    }
    
    welcome_msg = domain_messages.get(domain, domain_messages["general"])
    
    # Use the new shell template - minimal, agent-controlled UI
    resp = await render_template("index_shell.html", 
                                  domain=domain, 
                                  welcome_message=welcome_msg,
                                  is_pilot_user=is_pilot,
                                  user_id=user_id)
    
    # inject a tiny script that puts the key and domain into localStorage
    if api_key:
        resp = resp.replace("</head>",
           f'<script>localStorage.setItem("aethermind_api_key","{api_key}");'
           f'localStorage.setItem("aethermind_domain","{domain}");</script></head>')
    return resp

# Legacy route - redirect to new shell
@app.route("/chat/legacy")
async def legacy_chat():
    """Legacy chat UI with all features visible - for debugging"""
    api_key = request.args.get("api_key", "")
    domain = request.args.get("domain", session.get("user_domain", "general"))
    
    domain_messages = {
        "code": "Ready to build. I'm your Software Development Specialist.",
        "research": "Ready to analyze. I'm your Research & Analysis Specialist.",
        "business": "Ready to strategize. I'm your Business & Strategy Specialist.",
        "legal": "Ready to research. I'm your Legal Research Specialist.",
        "finance": "Ready to model. I'm your Finance & Investment Specialist.",
        "general": "Ready to assist. I'm your Multi-Domain Master."
    }
    
    welcome_msg = domain_messages.get(domain, domain_messages["general"])
    
    resp = await render_template("index.html", domain=domain, welcome_message=welcome_msg)
    
    if api_key:
        resp = resp.replace("</head>",
           f'<script>localStorage.setItem("aethermind_api_key","{api_key}");'
           f'localStorage.setItem("aethermind_domain","{domain}");</script></head>')
    return resp

# User profile endpoints
@app.route("/v1/user/profile", methods=["GET"])
async def get_user_profile():
    """Get user profile including onboarding status"""
    api_key = request.headers.get('X-Aether-Key')
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    
    # Verify key and get user
    user_data = auth_mgr.verify_api_key(api_key)
    if not user_data:
        return jsonify({"error": "Invalid API key"}), 401
    
    user_id = user_data.get('user_id')
    
    # Try to get profile from backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{backend_url}/v1/user/profile",
                headers={"X-Aether-Key": api_key},
                timeout=5.0
            )
            if resp.status_code == 200:
                return jsonify(resp.json())
    except:
        pass
    
    # Return basic profile if backend unavailable
    return jsonify({
        "user_id": user_id,
        "onboarded": False,
        "is_pilot": user_id in PILOT_USERS
    })

@app.route("/v1/user/profile", methods=["POST"])
async def save_user_profile():
    """Save user profile from onboarding"""
    api_key = request.headers.get('X-Aether-Key')
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    
    data = await request.get_json()
    
    # Forward to backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{backend_url}/v1/user/profile",
                json=data,
                headers={"X-Aether-Key": api_key},
                timeout=5.0
            )
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logging.error(f"Failed to save profile: {e}")
        return jsonify({"error": "Failed to save profile", "saved_locally": True}), 200

# Background tasks status endpoint
@app.route("/v1/tasks/status", methods=["POST"])
async def get_tasks_status():
    """Get status of background tasks"""
    api_key = request.headers.get('X-Aether-Key')
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    
    data = await request.get_json()
    task_ids = data.get('task_ids', [])
    
    # Forward to backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{backend_url}/v1/tasks/status",
                json={"task_ids": task_ids},
                headers={"X-Aether-Key": api_key},
                timeout=10.0
            )
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logging.error(f"Failed to get task status: {e}")
        return jsonify({"tasks": []}), 200

@app.route("/v1/ingest/multimodal", methods=["POST"])
async def upload_multimodal():
    """
    Proxy endpoint for multimodal file uploads.
    Forwards requests to FastAPI backend at port 8000.
    """
    try:
        # Get API key from header
        api_key = request.headers.get('Aether-Secret-Key') or request.headers.get('X-Aether-Key')
        
        if not api_key:
            return jsonify({"error": "Missing API key"}), 401
        
        # Get the uploaded file (await in Quart)
        files = await request.files
        if 'file' not in files:
            return jsonify({"error": "No file provided"}), 400
        
        file = files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Forward to FastAPI backend
        backend_url = "http://127.0.0.1:8000/v1/ingest/multimodal"
        
        # Read file content (synchronous in Quart)
        file_content = file.read()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Prepare multipart form data
            upload_files = {'file': (file.filename, file_content, file.content_type)}
            headers = {'X-Aether-Key': api_key}
            
            logging.info(f"Forwarding upload to backend: {file.filename}")
            response = await client.post(backend_url, files=upload_files, headers=headers)
            
            # Return the backend response
            return jsonify(response.json()), response.status_code
            
    except httpx.RequestError as e:
        logging.error(f"Backend connection error: {e}")
        return jsonify({"error": "Backend service unavailable"}), 503
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/games")
async def games_index():
    return await render_template("games.html")

@app.route("/games/life-simulator")
async def game_life_simulator():
    return await render_template("game_life_simulator.html")

@app.route("/games/smart-city")
async def game_smart_city():
    return await render_template("game_smart_city.html")

@app.route("/v1/vehicle/control", methods=["POST"])
async def vehicle_control():
    """
    Real-time vehicle control endpoint.
    Uses JEPA (Joint Embedding Predictive Architecture) to detect 'Surprise' (Energy).
    """
    global last_vehicle_state
    
    try:
        data = await request.get_json()
        
        # Telemetry
        front_dist = data.get('sensors', {}).get('front_min', 100)
        speed = data.get('speed', 0)
        
        # 1. ENCODE STATE (Create an embedding vector)
        # We normalize inputs to 0-1 range for the neural geometry
        current_state = np.zeros(VEHICLE_DIM)
        current_state[0] = min(front_dist, 200) / 200.0  # Front sensor
        current_state[1] = (speed + 2) / 6.0             # Speed (-2 to +4 range approx)
        # Add some noise/complexity to simulate high-dim features
        current_state[2] = np.sin(current_state[0] * np.pi) 
        
        # 2. COMPUTE JEPA ENERGY (Prediction Error)
        # Low Energy = Expected State (Routine driving)
        # High Energy = Unexpected State (Sudden obstacle, crash, erratic movement)
        energy = jepa_brain.compute_energy(last_vehicle_state, current_state)
        
        # 3. ONLINE LEARNING (Update World Model)
        # The brain learns to predict the next state
        jepa_brain.update_world_model(last_vehicle_state, current_state, learning_rate=0.05)
        last_vehicle_state = current_state
        
        # 4. DECISION LOGIC
        command = {
            "action": "maintain",
            "reason": f"Energy Low ({energy:.2f})",
            "energy": float(energy)
        }
        
        # JEPA-Driven Response: If the situation is "Surprising" (High Energy), be cautious
        if energy > 0.6:
             command = {
                "action": "slow_down",
                "reason": f"High JEPA Energy ({energy:.2f}) - Surprise Detected",
                "target_speed": 1.0
            }
        
        # Reflexive Safety Layer (Hard overrides for critical safety)
        if front_dist < 35:
            command = {
                "action": "emergency_stop",
                "reason": "CRITICAL_PROXIMITY",
                "brake_force": 1.0
            }
        elif front_dist < 80 and energy > 0.4:
            # Combined Logic: Close AND somewhat surprising
            command = {
                "action": "slow_down",
                "reason": f"Caution: Obstacle + Energy Spike ({energy:.2f})",
                "target_speed": 0.5
            }
            
        return jsonify(command)
        
    except Exception as e:
        app.logger.error(f"Vehicle Control Error: {e}")
        return jsonify({"action": "maintain", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
