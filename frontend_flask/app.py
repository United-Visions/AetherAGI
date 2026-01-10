import os
import sys
import json
import yaml
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE any imports that depend on them
# Load environment variables from parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Ensure we can import orchestrator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quart import Quart, render_template, request, redirect, url_for, session, jsonify
import logging
import httpx
import uuid
import markdown
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
from orchestrator.auth_manager_supabase import AuthManagerSupabase, UserRole
from orchestrator.supabase_client import SupabaseClient
from brain.jepa_aligner import JEPAAligner
from frontend_flask.core.benchmark_manager import BenchmarkManager
import numpy as np
from datetime import datetime

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
supabase = SupabaseClient().client
benchmark_mgr = BenchmarkManager()

# --- AETHER BRAIN (JEPA) ---
# Initialize the Joint Embedding Predictive Architecture for vehicle control
# We use a small dimension (16) for the simple vehicle telemetry vector
VEHICLE_DIM = 16
jepa_brain = JEPAAligner(dimension=VEHICLE_DIM, energy_threshold=0.3)
last_vehicle_state = np.zeros(VEHICLE_DIM)

BENCHMARK_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'benchmarks', 'results'
)

def compute_moving_average(scores, window_size=10):
    """Calculates the moving average of a list of scores."""
    if not scores or window_size <= 0:
        return []
    
    moving_averages = []
    # Use a rolling window to compute the average
    for i in range(len(scores)):
        start_index = max(0, i - window_size + 1)
        window = scores[start_index:i+1]
        window_average = sum(window) / len(window) if window else 0
        moving_averages.append(window_average)
    return moving_averages

def compute_trend_line(chunk_results):
    """Return linear regression stats for benchmark chunks."""
    if not chunk_results:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "equation": "score = 0",
            "direction": "flat",
            "latest_prediction": 0.0
        }

    xs = [entry.get("chunk", idx + 1) for idx, entry in enumerate(chunk_results)]
    ys = [entry.get("score", 0.0) for entry in chunk_results]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x

    direction = "up"
    if slope < -0.001:
        direction = "down"
    elif abs(slope) <= 0.001:
        direction = "flat"

    latest_prediction = slope * xs[-1] + intercept
    return {
        "slope": slope,
        "intercept": intercept,
        "equation": f"score = {slope:.4f} * chunk + {intercept:.4f}",
        "direction": direction,
        "latest_prediction": latest_prediction
    }


def list_benchmark_datasets():
    """Return metadata for available benchmark runs from Supabase."""
    datasets = {}
    
    try:
        if not supabase:
            return datasets
            
        # Fetch distinctive metadata (simulated distinct via limited query logic)
        # Ideally: SELECT DISTINCT family, day_id, variant FROM benchmark_results
        # PostgREST doesn't support SELECT DISTINCT directly easily in client
        # So we fetch recent 1000 rows and aggregate
        response = supabase.table('benchmark_results').select('family, day_id, variant, score, timestamp').order('timestamp', desc=True).limit(2000).execute()
        
        for row in response.data:
            family_key = row.get('family', 'unknown')
            # Fix: Do not default to day_1 to prevent zombie data merging
            day = row.get('day_id') or 'unknown_session'
            variant_key = row.get('variant') or 'gen'
            
            if family_key not in datasets:
                datasets[family_key] = {
                    "display_name": family_key.replace('_', ' ').title(),
                    "runs": []
                }
            
            # Check if this day is already added
            existing_run = next((r for r in datasets[family_key]["runs"] if r['id'] == day), None)
            if not existing_run:
                datasets[family_key]["runs"].append({
                    "id": day,
                    "file_name": day, # Legacy field compat
                    "day": day.replace('day_', ''),
                    "family": family_key,
                    "variant": "Mixed" if "all" in day else variant_key,
                    "overall_average": 0, # Calculated in detailed view
                    "chunks_completed": 0,
                    "timestamp": row.get('timestamp')
                })
        
        # Sort runs
        for family in datasets.values():
             family['runs'].sort(key=lambda r: r.get('id') or '0')
             
    except Exception as e:
        logging.warning(f"Failed to list benchmarks from Supabase: {e}")
        # Fallback to empty if table doesn't exist yet
        pass

    return datasets


def load_benchmark_payload(dataset_slug):
    """Load benchmark data from Supabase for a specific day."""
    if not dataset_slug or not supabase:
        return None

    try:
        # Fetch all chunks for this day
        # dataset_slug corresponds to 'day_id'
        response = supabase.table('benchmark_results').select('*').eq('day_id', dataset_slug).order('chunk_index', desc=False).execute()
        
        if not response.data:
            return None
            
        chunks = response.data
        
        # Calculate aggregations
        scores = [c.get('score', 0) for c in chunks]
        overall_average = sum(scores) / len(scores) if scores else 0
        
        last_chunk = chunks[-1] if chunks else {}
        
        # Group by variant AND model
        scores_by_variant = {}
        model_stats = {} # { "Gemini 2.5": [0.9, 0.95], ... }
        
        for c in chunks:
            v = c.get('variant', 'unknown')
            meta = c.get('metadata') or {}
            # Handle potential string metadata (legacy) vs dict
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            
            model = meta.get('model', 'unknown')
            score = c.get('score', 0)
            
            # Variant Aggregation
            if v not in scores_by_variant:
                scores_by_variant[v] = {
                    "chunks_completed": 0, 
                    "scores": [],
                    "models": {} 
                }
            scores_by_variant[v]["chunks_completed"] += 1
            scores_by_variant[v]["scores"].append(score)
            
            # Model within Variant Aggregation
            if model not in scores_by_variant[v]["models"]:
                 scores_by_variant[v]["models"][model] = []
            scores_by_variant[v]["models"][model].append(score)
            
            # Overall Model Aggregation (for this day)
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(score)

        # Finalize averages
        for v in scores_by_variant:
            v_data = scores_by_variant[v]
            v_scores = v_data["scores"]
            v_data["average"] = sum(v_scores)/len(v_scores) if v_scores else 0
            
            # Finalize model specific averages for this variant
            for m in v_data["models"]:
                m_scores = v_data["models"][m]
                v_data["models"][m] = sum(m_scores)/len(m_scores) if m_scores else 0
                
            del v_data["scores"] # Clean up raw lists from summary
            
        # Finalize overall model stats
        final_model_stats = {}
        for m, m_scores in model_stats.items():
            final_model_stats[m] = sum(m_scores)/len(m_scores) if m_scores else 0
            
        # NEW: Group by Model -> Variant for hierarchy view
        results_by_model = {}
        for c in chunks:
            meta = c.get('metadata') or {}
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            model = meta.get('model', 'unknown')
            variant = c.get('variant', 'unknown')
            
            if model not in results_by_model:
                results_by_model[model] = {
                    "variants": {},
                    "overall_score": 0,
                    "total_chunks": 0,
                    "scores_acc": []
                }
            
            if variant not in results_by_model[model]["variants"]:
                results_by_model[model]["variants"][variant] = {
                    "chunks": [],
                    "latest_chunk": None,
                    "score_sum": 0,
                    "count": 0
                }
                
            # Add to variant bucket
            results_by_model[model]["variants"][variant]["chunks"].append(c)
            results_by_model[model]["variants"][variant]["latest_chunk"] = c # reliant on sort order
            results_by_model[model]["variants"][variant]["score_sum"] += c.get('score', 0)
            results_by_model[model]["variants"][variant]["count"] += 1
            
            # Add to model totals
            results_by_model[model]["scores_acc"].append(c.get('score', 0))
            results_by_model[model]["total_chunks"] += 1
            
        # Calculate Model Averages
        for m in results_by_model:
            m_data = results_by_model[m]
            if m_data["scores_acc"]:
                m_data["overall_score"] = sum(m_data["scores_acc"]) / len(m_data["scores_acc"])
            
            # Calculate Variant Averages within Model
            for v in m_data["variants"]:
                v_data = m_data["variants"][v]
                if v_data["count"] > 0:
                    v_data["average"] = v_data["score_sum"] / v_data["count"]
        
        # Sort results_by_model by overall_score descending
        sorted_results = dict(sorted(results_by_model.items(), key=lambda item: item[1].get('overall_score', 0), reverse=True))

        return {
            "family": chunks[0].get('family', 'gsm'),
            "variant": "Multi-Variant" if len(scores_by_variant) > 1 else chunks[0].get('variant'),
            "available_variants": list(scores_by_variant.keys()),
            "overall_average": overall_average,
            "all_chunk_results": [
                {
                    "chunk": c.get("chunk_index"),
                    "score": c.get("score"),
                    "correct": c.get("correct_count"),
                    "total": c.get("total_questions"),
                    "variant": c.get("variant"), # Critical for coloring
                    "model": (c.get("metadata") or {}).get("model", "unknown") if isinstance(c.get("metadata"), dict) else "unknown",
                    "timestamp": c.get("timestamp"),
                    "total_chunks": c.get("total_chunks")
                }
                for c in chunks
            ],
            "scores_by_variant": scores_by_variant,
            "model_leaderboard": final_model_stats,
            "results_by_model": sorted_results,
            "last_chunk": last_chunk,
            "timestamp": last_chunk.get("timestamp")
        }

    except Exception as e:
        logging.error(f"Failed to load benchmark payload from Supabase: {e}")
        return None


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
    return await render_template("home.html")

@app.route("/pricing")
async def pricing():
    return await render_template("pricing.html")

@app.route("/voice-test")
async def voice_test():
    """Voice synthesis test page"""
    return await render_template("voice_test.html")

@app.route("/documentation")
async def documentation():
    return await render_template("documentation.html")

@app.route("/benchmarks")
async def benchmarks():
    all_datasets = list_benchmark_datasets()
    
    # Determine selected family and day from query params
    selected_family_key = request.args.get("family")
    if not selected_family_key and all_datasets:
        selected_family_key = next(iter(all_datasets))

    selected_day_id = request.args.get("day")
    
    # Get the specific run to display
    selected_run = None
    if selected_family_key in all_datasets:
        family_runs = all_datasets[selected_family_key]['runs']
        if selected_day_id:
            selected_run = next((r for r in family_runs if r['id'] == selected_day_id), None)
        
        # If no specific day is selected or found, default to the first run in the family
        if not selected_run and family_runs:
            selected_run = family_runs[0]
            selected_day_id = selected_run['id']

    payload = load_benchmark_payload(selected_day_id)

    if not payload:
        return await render_template(
            "benchmarks.html",
            error_message=f"No benchmark data available for '{selected_day_id}'.",
            datasets=all_datasets,
            selected_family=selected_family_key,
            selected_day=selected_day_id,
            family="N/A",
            variant="N/A",
            variant_data={},
            overall_average=None,
            last_chunk={},
            chunk_results=[],
            completion_pct=0,
            trend=compute_trend_line([]),
            chunk_labels=[],
            chunk_scores=[],
            moving_average=[],
            trend_line=[],
            best_chunk=None,
            low_chunk=None,
            recent_average=None,
            total_chunks=0,
            timestamp=None,
            supabase_url=os.getenv("SUPABASE_URL") or os.getenv("SB_URL"),
            supabase_key=os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SB_SECRET_KEY")
        )

    all_chunk_results = sorted(payload.get("all_chunk_results", []), key=lambda entry: entry.get("chunk", 0))
    scores_by_variant = payload.get("scores_by_variant", {})
    available_variants = payload.get("available_variants", [])
    model_leaderboard = payload.get("model_leaderboard", {})

    # Determine which variant to show stats for
    requested_variant = request.args.get('view_variant')
    variant_key = "Unknown Variant"
    
    if requested_variant and requested_variant in scores_by_variant:
        variant_key = requested_variant
    elif scores_by_variant:
        # Sort keys to ensure stable default
        variant_key = sorted(list(scores_by_variant.keys()))[0]
        
    variant_data = scores_by_variant.get(variant_key, {})

    # Filter chunk results for visualization based on selection
    # If a specific view is requested, we filter.
    if requested_variant == 'all':
        chunk_results = all_chunk_results
    elif variant_key != "Unknown Variant":
        chunk_results = [c for c in all_chunk_results if c.get('variant') == variant_key]
    else:
        chunk_results = all_chunk_results

    chunk_results = sorted(chunk_results, key=lambda entry: entry.get("chunk", 0))

    best_chunk = max(chunk_results, key=lambda entry: entry.get("score", 0)) if chunk_results else None
    low_chunk = min(chunk_results, key=lambda entry: entry.get("score", 0)) if chunk_results else None
    recent_slice = chunk_results[-5:] if len(chunk_results) >= 5 else chunk_results
    recent_average = (
        sum(entry.get("score", 0) for entry in recent_slice) / len(recent_slice)
        if recent_slice else None
    )

    total_chunks = 0
    if chunk_results:
        total_chunks = (
            chunk_results[-1].get("total_chunks")
            or payload.get("last_chunk", {}).get("total_chunks")
            or variant_data.get("chunks_completed")
            or len(chunk_results)
        )
    else:
        total_chunks = (
            payload.get("last_chunk", {}).get("total_chunks")
            or variant_data.get("chunks_completed")
            or 0
        )

    completion_pct = 0.0
    if chunk_results and total_chunks:
        completion_pct = (chunk_results[-1].get("chunk", 0) / total_chunks) * 100

    trend = compute_trend_line(chunk_results)
    chunk_labels = [f"Chunk {entry.get('chunk', '?')}" for entry in chunk_results]
    chunk_scores = [entry.get("score", 0) for entry in chunk_results]
    moving_average = compute_moving_average(chunk_scores, window_size=10)
    chunk_indices = [entry.get("chunk", idx + 1) for idx, entry in enumerate(chunk_results)]
    trend_line = [trend["slope"] * idx + trend["intercept"] for idx in chunk_indices] if chunk_indices else []

    return await render_template(
        "benchmarks.html",
        family=payload.get("family", "gsm"),
        variant=variant_key,
        variant_data=variant_data,
        available_variants=available_variants,
        scores_by_variant=scores_by_variant,
        model_leaderboard=variant_data.get('models', {}),
        overall_model_stats=model_leaderboard,
        results_by_model=payload.get("results_by_model", {}),
        overall_average=payload.get("overall_average"),
        last_chunk=payload.get("last_chunk", {}),
        chunk_results=chunk_results,
        completion_pct=completion_pct,
        trend=trend,
        chunk_labels=chunk_labels,
        chunk_scores=chunk_scores,
        moving_average=moving_average,
        trend_line=trend_line,
        best_chunk=best_chunk,
        low_chunk=low_chunk,
        recent_average=recent_average,
        total_chunks=total_chunks,
        timestamp=payload.get("timestamp"),
        datasets=all_datasets,
        selected_family=selected_family_key,
        selected_day=selected_day_id,
        error_message=None,
        supabase_url=os.getenv("SUPABASE_URL") or os.getenv("SB_URL"),
        supabase_key=os.getenv("SUPABASE_ANON_KEY") or os.getenv("SB_ANON_KEY")
    )

@app.route("/whitepaper")
async def whitepaper():
    return await render_template("aether_whitepaper.html")

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
    """Legacy chat UI - now redirects to shell"""
    api_key = request.args.get("api_key", "")
    domain = request.args.get("domain", session.get("user_domain", "general"))
    return redirect(url_for("index", api_key=api_key, domain=domain))

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

# Chat completions proxy endpoint
@app.route("/v1/chat/completions", methods=["POST"])
async def chat_completions():
    """Proxy chat completions to backend"""
    api_key = request.headers.get('X-Aether-Key')
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    
    data = await request.get_json()
    
    # Forward to backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{backend_url}/v1/chat/completions",
                json=data,
                headers={"X-Aether-Key": api_key},
                timeout=120.0  # Long timeout for LLM responses
            )
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logging.error(f"Chat completions error: {e}")
        return jsonify({"error": str(e)}), 500

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

# Voice synthesis proxy endpoints
@app.route("/v1/voice/synthesize", methods=["POST", "OPTIONS"])
async def voice_synthesize_proxy():
    """Proxy voice synthesis to backend"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return "", 200
    
    api_key = request.headers.get('X-Api-Key')
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    
    data = await request.get_json()
    
    # Forward to backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{backend_url}/v1/voice/synthesize",
                json=data,
                headers={"X-Api-Key": api_key},
                timeout=60.0
            )
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logging.error(f"Voice synthesis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/v1/voice/voices", methods=["GET"])
async def voice_list_proxy():
    """Proxy voice list to backend"""
    api_key = request.headers.get('X-Api-Key')
    language = request.args.get("language", "en")
    
    # Forward to backend
    try:
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{backend_url}/v1/voice/voices?language={language}",
                headers={"X-Api-Key": api_key} if api_key else {},
                timeout=30.0
            )
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logging.error(f"Voice list error: {e}")
        return jsonify({"error": str(e)}), 500

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

@app.route("/api/benchmarks/start", methods=["POST"])
async def api_benchmarks_start():
    api_key = request.headers.get('X-Aether-Key')
    # Basic auth check or session check
    # if not api_key: return jsonify({"error": "Unauthorized"}), 401
    
    data = await request.get_json()
    day_id = data.get("day_id")
    family = data.get("family", "gsm")
    variants = data.get("variants", ["all"])
    model = data.get("model") # Optional model override
    
    if not day_id:
        return jsonify({"error": "day_id is required"}), 400
        
    result = benchmark_mgr.start_benchmark(day_id, family, variants, model=model)
    return jsonify(result)

@app.route("/api/benchmarks/stop", methods=["POST"])
async def api_benchmarks_stop():
    data = await request.get_json()
    day_id = data.get("day_id")
    run_id = data.get("run_id")
    clear_data = data.get("clear_data", False)
    
    stopped = benchmark_mgr.stop_benchmark(day_id, run_id)
    
    if clear_data and day_id and supabase:
        try:
            # Delete results for this day
            supabase.table('benchmark_results').delete().eq('day_id', day_id).execute()
        except Exception as e:
            app.logger.error(f"Failed to clear data for {day_id}: {e}")
            
    return jsonify({"status": "stopped", "stopped_runs": stopped, "data_cleared": clear_data})

@app.route("/api/benchmarks/clear", methods=["POST"])
async def api_benchmarks_clear():
    data = await request.get_json()
    day_id = data.get("day_id")
    
    if not day_id:
        return jsonify({"error": "day_id is required"}), 400

    if supabase:
        try:
             supabase.table('benchmark_results').delete().eq('day_id', day_id).execute()
             return jsonify({"status": "cleared", "day_id": day_id})
        except Exception as e:
             return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Supabase not connected"}), 503

@app.route("/api/benchmarks/status", methods=["GET"])
async def api_benchmarks_status():
    status = benchmark_mgr.get_active_status()
    # Enrich process list with parsed details
    enriched_runs = []
    for rid in status.get("active_runs", []):
         parts = rid.split('_')
         # ID format: day_id_Variant_Name_timestamp
         # Example: day_1_GSM-Hard_1234567890
         # But variant might have spaces replaced by underscores?
         # start_benchmark does: variant.replace(' ', '_')
         
         # Heuristic parsing
         day_id = parts[0] + "_" + parts[1] if len(parts) > 1 and parts[0] == "day" else "unknown"
         
         enriched_runs.append({
             "run_id": rid,
             "day_id": day_id,
             "raw": rid
         })
         
    status["details"] = enriched_runs
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True)
