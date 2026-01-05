import os
import sys
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE any imports that depend on them
# Load environment variables from parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Ensure we can import orchestrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import logging
import httpx
import uuid
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
from orchestrator.auth_manager_supabase import AuthManagerSupabase, UserRole

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-change-me")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

auth_mgr = AuthManagerSupabase()

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
def home():
    return render_template("index_home.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/domain/legal")
def domain_legal():
    return render_template("domain_legal.html")

@app.route("/domain/medical")
def domain_medical():
    # TODO: Create domain_medical.html
    return render_template("domain_legal.html")  # Placeholder

@app.route("/domain/finance")
def domain_finance():
    # TODO: Create domain_finance.html
    return render_template("domain_legal.html")  # Placeholder

@app.route("/domain/code")
def domain_code():
    # TODO: Create domain_code.html
    return render_template("domain_legal.html")  # Placeholder

@app.route("/domain/research")
def domain_research():
    # TODO: Create domain_research.html
    return render_template("domain_legal.html")  # Placeholder

@app.route("/github_login")
def github_login():
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
def callback():
    code = request.args.get("code")
    if not code:
        return "Missing code", 400
    # exchange code for token
    try:
        r = httpx.post("https://github.com/login/oauth/access_token",
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
        user_r = httpx.get("https://api.github.com/user",
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
def onboarding():
    if "github_user" not in session:
        return redirect(url_for("home"))
    return render_template("onboarding.html", github_user=session["github_user"])

@app.route("/create_key", methods=["POST"])
def create_key():
    """
    Generate a personal API key for authenticated user.
    
    Architecture:
    - Backend service keys (PINECONE_API_KEY, RUNPOD_API_KEY) are in .env
    - User personal keys (am_live_XXX) are generated here and hashed in keys.json
    - Each user gets their own key with role-based permissions (FREE/PRO/ENTERPRISE)
    """
    if "github_user" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session["github_user"]
    
    # Get domain and subscription tier from form or JSON
    if request.is_json:
        data = request.get_json()
        domain = data.get("domain", "general")
        tier = data.get("tier", "pro")  # Default to pro for now
    else:
        domain = request.form.get("domain", "general")
        tier = request.form.get("tier", "pro")
    
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
def index():
    # if api_key in query string, preload it into the page
    api_key = request.args.get("api_key", "")
    domain = request.args.get("domain", session.get("user_domain", "general"))
    
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
    
    resp = render_template("index.html", domain=domain, welcome_message=welcome_msg)
    
    # inject a tiny script that puts the key and domain into localStorage
    if api_key:
        resp = resp.replace("</head>",
           f'<script>localStorage.setItem("aethermind_api_key","{api_key}");'
           f'localStorage.setItem("aethermind_domain","{domain}");</script></head>')
    return resp

if __name__ == '__main__':
    app.run(debug=True)
