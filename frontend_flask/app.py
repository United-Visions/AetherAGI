from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import logging
import os, httpx, uuid
import sys
from urllib.parse import quote_plus
from cryptography.fernet import Fernet

# Ensure we can import orchestrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.auth_manager import AuthManager

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-change-me")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

auth_mgr = AuthManager()

# --- GitHub OAuth ---
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI  = os.getenv("GITHUB_REDIRECT_URI", "http://127.0.0.1:5000/callback")
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

@app.route("/github_login")
def github_login():
    scope = "read:user repo invite"   # repo scope needed for ToolForge
    url = (f"https://github.com/login/oauth/authorize"
           f"?client_id={GITHUB_CLIENT_ID}&redirect_uri={quote_plus(GITHUB_REDIRECT_URI)}"
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
        # encrypt & store token
        encrypted = cipher.encrypt(token.encode()).decode()
        session["github_user"] = github_user
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
    if "github_user" not in session:
        return redirect(url_for("home"))
    user_id = session["github_user"]
    key = auth_mgr.generate_api_key(user_id)   # persistent key
    # pass key to chat page via query-param (client stores in localStorage)
    return redirect(url_for("index", api_key=key))

# --- existing chat page ---
@app.route("/chat")
def index():
    # if api_key in query string, preload it into the page
    api_key = request.args.get("api_key", "")
    resp = render_template("index.html")
    # inject a tiny script that puts the key into localStorage
    if api_key:
        resp = resp.replace("</head>",
           f'<script>localStorage.setItem("aethermind_api_key","{api_key}");</script></head>')
    return resp

if __name__ == '__main__':
    app.run(debug=True)
