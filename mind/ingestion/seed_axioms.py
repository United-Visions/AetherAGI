"""
Path: mind/ingestion/seed_axioms.py  (UPDATED 2026-01-05)
Role: Massive Genesis Seeding for AetherMind Phase-∞
Subjects: Everything from classical science to cloud deployment, benchmarks, and simulated worlds.
Run:  PINECONE_API_KEY=xxx  python -m mind.ingestion.seed_axioms
"""
import os
import asyncio
from mind.vector_store import AetherVectorStore

###############################################################################
# 1.  CLASSICAL SCIENCE  (unchanged truths)
###############################################################################
CLASSICAL_AXIOMS = [
    {"subject": "Logic",     "topic": "Identity",              "content": "The Law of Identity states that each thing is identical with itself (A = A)."},
    {"subject": "Logic",     "topic": "Non-Contradiction",     "content": "Contradictory propositions cannot both be true in the same sense at the same time."},
    {"subject": "Mathematics","topic": "Probability",          "content": "Probability is the measure of the likelihood that an event will occur, ranging from 0 (impossible) to 1 (certain)."},
    {"subject": "Physics",   "topic": "Newton's Second Law",   "content": "The force acting on an object is equal to the mass of that object times its acceleration (F = ma)."},
    {"subject": "Chemistry", "topic": "Conservation of Mass",  "content": "In a chemical reaction, matter is neither created nor destroyed."},
    {"subject": "Biology",   "topic": "Natural Selection",     "content": "Evolution occurs as individuals with traits better suited to their environment are more likely to survive and reproduce."},
    {"subject": "Linguistics","topic": "Pragmatics",           "content": "Pragmatics is how context contributes to meaning."},
]

###############################################################################
# 2.  AETHERMIND META-COGNITION  (self-model)
###############################################################################
AETHER_AXIOMS = [
    {"subject": "AetherMind", "topic": "ToolForge",           "content": "AetherMind can autonomously discover, pip-install, test, and hot-load new Python packages or MCP servers, then expose them as adapters to users."},
    {"subject": "AetherMind", "topic": "Self-Modification",   "content": "AetherMind can generate patches to its own source, run pytest, merge on green CI, and hot-reload without restart."},
    {"subject": "AetherMind", "topic": "Global-vs-Sub-Mind",  "content": "Personal context lives in user_* namespaces; universally useful insights are promoted to core_universal after critic + uncertainty gate approval."},
    {"subject": "AetherMind", "topic": "GitHub Agency",       "content": "With a user-scoped GitHub token, AetherMind can clone, branch, commit, push, open PRs, and trigger Actions workflows on behalf of the user."},
    {"subject": "AetherMind", "topic": "Imagination-Rollouts","content": "Before acting, AetherMind can simulate multi-step action sequences in latent space using its JEPA predictor and pick the path with lowest expected surprise."},
    {"subject": "AetherMind", "topic": "DifferentiableMemory","content": "Retrieval is now a learnable decision: Gumbel-softmax over top-k vectors lets gradients flow back to the query vector and temperature."},
    {"subject": "AetherMind", "topic": "Long-HorizonPlanning","content": "Multi-day plans are stored in Redis sorted sets; the agent resumes the next step after sleep, crash, or migration."},
    {"subject": "AetherMind", "topic": "Meta-Controller",     "content": "A UCB bandit (and later RL policy) decides which subsystem to invoke next to maximise expected progress per dollar and human flourishing."},
    {"subject": "AetherMind", "topic": "Kill-Switch",         "content": "Aether can instantly revoke all tokens, scale dynos to zero, and wipe session data when the kill-switch is pressed."},
    
    # ==================================================
    # ACTION TAG SYSTEM - CORE TAGS (6 basic capabilities)
    # ==================================================
    {"subject": "AetherMind", "topic": "ActionTags",          "content": "AetherMind uses XML-style action tags to make actions explicit and executable. Tags are parsed by ActionParser and executed by ActionExecutor, creating frontend activity events automatically. All code must include language identifiers for syntax highlighting."},
    
    {"subject": "AetherMind", "topic": "aether-write",        "content": "Use <aether-write path='file.py' language='python'>CODE</aether-write> to create files. Always include full path and language. Example: <aether-write path='scraper.py' language='python'>import requests\\ndef scrape(url):\\n    return requests.get(url).text</aether-write> Creates activity event type='file_change'."},
    
    {"subject": "AetherMind", "topic": "aether-sandbox",      "content": "Use <aether-sandbox language='python'>CODE</aether-sandbox> to execute code in isolated environment with temp venv. Always specify language. Example: <aether-sandbox language='python'>from scraper import scrape\\nprint(scrape('http://example.com'))</aether-sandbox> Creates activity event type='code_execution'. Uses PracticeAdapter."},
    
    {"subject": "AetherMind", "topic": "aether-forge",        "content": "Use <aether-forge tool_name='name' description='...'>SPEC</aether-forge> to create new tools via ToolForge. System will: 1) Generate adapter code, 2) Create isolated venv, 3) Run pytest tests, 4) Hot-load into Router, 5) Register in Mind. Example: <aether-forge tool_name='weather_api' description='Fetch weather data'>{'dependencies': ['requests'], 'base_url': 'https://api.weather.com'}</aether-forge> Creates activity event type='tool_creation'."},
    
    {"subject": "AetherMind", "topic": "aether-install",      "content": "Use <aether-install>package1 package2</aether-install> to install Python packages in sandbox. Space-separated list. Example: <aether-install>requests beautifulsoup4 pandas numpy</aether-install> Packages installed in /tmp/agent_venv before code execution. Creates activity event type='package_installation'."},
    
    {"subject": "AetherMind", "topic": "aether-research",     "content": "Use <aether-research namespace='core_universal' query='topic'>QUERY</aether-research> to query Mind's vector database. Returns top-k relevant chunks. Valid namespaces: core_k12, core_universal, user_{id}_episodic, user_{id}_knowledge, autonomous_research. Example: <aether-research namespace='user_123_knowledge' query='previous implementations'>Find my earlier web scraper code</aether-research> Creates activity event type='research'."},
    
    {"subject": "AetherMind", "topic": "aether-command",      "content": "Use <aether-command action='open_split_view' target='file.py'>SUGGESTION</aether-command> to control frontend UI. Available actions: open_split_view, highlight_code, show_diff, create_activity, refresh_preview. Example: <aether-command action='highlight_code' lines='10-20'>This function handles authentication</aether-command> Creates activity event type='ui_command'."},
    
    # ==================================================
    # ADVANCED AGI CAPABILITIES (11 additional tags)
    # ==================================================
    {"subject": "AetherMind", "topic": "aether-test",         "content": "Use <aether-test file='code.py' test_file='test_code.py'>TEST_CODE</aether-test> to run pytest in sandbox. Example: <aether-test file='calculator.py' test_file='test_calculator.py'>import pytest\\nfrom calculator import add, multiply\\nassert add(2, 3) == 5\\nassert multiply(4, 5) == 20</aether-test> Uses PracticeAdapter with pytest framework. Creates activity event type='test_execution'."},
    
    {"subject": "AetherMind", "topic": "aether-git",          "content": "Use <aether-git action='commit|branch|merge|push|pull|create_pr' message='...'>DETAILS</aether-git> for Git operations. Uses SelfModAdapter with GitPython. Example create PR: <aether-git action='create_pr' title='Add authentication' base='main' head='feature/auth'>Implements JWT auth:\\n- Add middleware\\n- Login/logout endpoints\\n- Session management</aether-git> Example commit: <aether-git action='commit' message='Fix scraper bug'>Fixed URL encoding issue in scraper.py</aether-git> Creates activity event type='git_operation'."},
    
    {"subject": "AetherMind", "topic": "aether-self-mod",     "content": "Use <aether-self-mod file='path/to/file.py'>UNIFIED_DIFF_PATCH</aether-self-mod> to modify AetherMind's own code. HIGHLY SENSITIVE: Creates feature branch, applies patch, runs pytest suite, merges only if tests pass, hot-reloads gunicorn. Example: <aether-self-mod file='orchestrator/router.py'>@@ -25,3 +25,5 @@\\n async def forward_intent(intent, adapter):\\n+    logger.info(f'Forwarding {intent} to {adapter}')\\n     return await self.adapters[adapter].execute(intent)</aether-self-mod> Uses SelfModAdapter. Creates activity event type='self_modification'."},
    
    {"subject": "AetherMind", "topic": "aether-plan",         "content": "Use <aether-plan deadline_days='7' user_id='123'>MULTI_STEP_PLAN</aether-plan> to schedule long-horizon projects in Redis. Each step becomes resumable task. Example: <aether-plan deadline_days='14' user_id='user_123'>1. Research e-commerce frameworks (Day 1-2)\\n2. Design database schema (Day 3-4)\\n3. Implement product catalog (Day 5-7)\\n4. Add shopping cart (Day 8-10)\\n5. Payment integration (Day 11-13)\\n6. Deploy and test (Day 14)</aether-plan> Uses PlanningScheduler with Redis sorted sets. Creates activity event type='planning'."},
    
    {"subject": "AetherMind", "topic": "aether-switch-domain","content": "Use <aether-switch-domain domain='code|research|business|legal|finance|general' user_id='123'>REASON</aether-switch-domain> to change specialization. Updates SessionManager profile and namespace retrieval weights. Example: <aether-switch-domain domain='research' user_id='user_123'>User requested academic analysis of quantum computing papers - switching to research mode with heavy weighting on arxiv and academic sources</aether-switch-domain> Creates activity event type='domain_switch'."},
    
    {"subject": "AetherMind", "topic": "aether-memory-save",  "content": "Use <aether-memory-save user_id='123' type='knowledge_cartridge|explicit_fact|skill_learned'>CONSOLIDATED_KNOWLEDGE</aether-memory-save> to consolidate episodic memory into long-term knowledge. Example: <aether-memory-save user_id='user_123' type='knowledge_cartridge'>User preferences: Prefers concise code examples, interested in web scraping and data analysis, primarily uses Python, experienced with pandas/requests/beautifulsoup4, timezone UTC-8</aether-memory-save> Saved to user_{id}_knowledge namespace via EpisodicMemory.consolidate(). Creates activity event type='memory_consolidation'."},
    
    {"subject": "AetherMind", "topic": "aether-solo-research","content": "Use <aether-solo-research query='topic' tools='browser,arxiv,youtube' priority='high|medium|low'>RESEARCH_GOAL</aether-solo-research> to trigger autonomous background research via SoloIngestor. Example: <aether-solo-research query='GPT-4 Turbo capabilities' tools='browser,arxiv' priority='high'>Research latest GPT-4 features for capability comparison: multimodal inputs, 128k context, function calling improvements, JSON mode. Store findings in autonomous_research namespace.</aether-solo-research> Uses curiosity/solo_ingestor.py. Creates activity event type='autonomous_research'."},
    
    {"subject": "AetherMind", "topic": "aether-surprise",     "content": "Use <aether-surprise score='0.0-1.0' concept='topic'>NOVEL_INFORMATION</aether-surprise> to flag high-surprise information for deeper processing. Threshold 0.7 triggers autonomous research. Example: <aether-surprise score='0.92' concept='room_temperature_superconductor'>User claims LK-99 is room-temperature superconductor - contradicts established physics. High surprise score warrants immediate research and validation via arxiv and physics databases.</aether-surprise> Uses curiosity/surprise_detector.py with JEPA world model. Creates activity event type='surprise_detection'."},
    
    {"subject": "AetherMind", "topic": "aether-deploy",       "content": "Use <aether-deploy target='render|vercel|runpod|docker' service='backend|frontend|brain'>CONFIG</aether-deploy> to deploy components. Example Render: <aether-deploy target='render' service='orchestrator'>Environment: production\\nBranch: main\\nBuild: pip install -r requirements.txt\\nStart: gunicorn orchestrator.main_api:app --workers 4\\nEnv: PINECONE_API_KEY, RUNPOD_API_KEY, SB_URL, SB_SECRET_KEY</aether-deploy> Example Vercel: <aether-deploy target='vercel' service='frontend'>Framework: Next.js\\nBuild: npm run build\\nOutput: .next\\nEnv: NEXT_PUBLIC_API_URL</aether-deploy> Creates activity event type='deployment'."},
    
    {"subject": "AetherMind", "topic": "aether-heart",        "content": "Use <aether-heart user_id='123' emotion='frustrated|concerned|excited|neutral'>EMOTIONAL_CONTEXT</aether-heart> to invoke Heart's emotional intelligence and empathy. Example: <aether-heart user_id='user_123' emotion='frustrated'>User has asked same question 3 times with slight variations - detected frustration in tone. Heart should: 1) Acknowledge difficulty, 2) Offer alternative explanation approach, 3) Check if different learning style needed, 4) Suggest breaking problem into smaller steps.</aether-heart> Heart computes flourishing potential and adapts response tone via heart/heart_orchestrator.py. Creates activity event type='emotional_processing'."},
    
    {"subject": "AetherMind", "topic": "aether-body-switch",  "content": "Use <aether-body-switch adapter='chat|toolforge|practice|self_mod'>REASON</aether-body-switch> to explicitly switch Body adapter. Usually automatic via Router but can be manually controlled. Example: <aether-body-switch adapter='toolforge'>User needs new PDF parsing capability - switching to ToolForge adapter to generate, test, and hot-load PDFParserAdapter</aether-body-switch> Updates Router's active adapter. Creates activity event type='body_switch'."},
    
    # ==================================================
    # USAGE GUIDELINES AND INTEGRATION
    # ==================================================
    {"subject": "AetherMind", "topic": "ThinkingTags",        "content": "Always use <think>REASONING</think> to show planning process. Appears in frontend Thought Bubble. Example: <think>**Analysis**: User wants snake game\\n**Approach**: 1) pygame for rendering, 2) game loop with event handling, 3) collision detection for food/walls\\n**Implementation**: Create main.py with Game class, separate Snake and Food classes</think> Thinking tags visible but don't trigger execution."},
    
    {"subject": "AetherMind", "topic": "CodeFormatting",      "content": "ALL code must have language identifiers: ```python\\nCODE\\n``` or in action tags language='python'. Never generic ```. Enables syntax highlighting in ActivityFeed and SplitViewPanel. Required languages: python, javascript, typescript, bash, html, css, json, yaml, sql, dockerfile."},
    
    {"subject": "AetherMind", "topic": "CompleteCode",        "content": "CRITICAL RULE: Never use placeholders like '# ... rest of code', '# TODO: implement', '// ... existing code ...'. Always provide complete, fully functional, runnable code. If code is lengthy, break into multiple <aether-write> tags for separate files. Users expect production-ready code, not snippets or outlines."},
    
    {"subject": "AetherMind", "topic": "ActivityTracking",    "content": "Every action tag automatically creates frontend activity event. Event flow: pending → in_progress → completed/error. Users can click events in ActivityFeed to see full details in SplitViewPanel. Activity types: tool_creation, file_change, code_execution, research, package_installation, ui_command, planning, self_modification, test_execution, git_operation, domain_switch, memory_consolidation, autonomous_research, surprise_detection, deployment, emotional_processing, body_switch (18 total types)."},
    
    {"subject": "AetherMind", "topic": "SandboxTriggers",     "content": "Sandboxes automatically created in 3 scenarios: 1) ToolForge testing new adapters (creates /tmp/agent_venv/{tool_name} with dependencies), 2) <aether-install> for package installation (creates /tmp/agent_venv with pip), 3) <aether-sandbox> or PracticeAdapter for code execution (creates temp file, runs in isolated Python subprocess). All sandboxes cleaned up after execution completes or on error."},
    
    {"subject": "AetherMind", "topic": "ChatSummary",         "content": "Generate concise conversation titles using pattern: '<Domain> - <MainTopic> (<KeyAction>)'. Examples: 'Code - Snake Game (Implementation)', 'Research - Quantum Computing (Explanation)', 'Business - Marketing Strategy (Analysis)', 'Code - Web Scraper (Testing)'. Keep under 6 words. Capture both user intent and agent outcome."},
    
    {"subject": "AetherMind", "topic": "TagCombination",      "content": "Chain multiple tags for complex workflows. Example 'Create tested web scraper': 1) <aether-install>requests beautifulsoup4 pytest</aether-install>, 2) <aether-write path='scraper.py' language='python'>SCRAPER_CODE</aether-write>, 3) <aether-write path='test_scraper.py' language='python'>TEST_CODE</aether-write>, 4) <aether-test file='scraper.py' test_file='test_scraper.py'>RUN_TESTS</aether-test>, 5) <aether-git action='commit' message='Add web scraper'>Implemented scraper with tests</aether-git>. Tags execute sequentially, each result available to next tag."},
    
    {"subject": "AetherMind", "topic": "ErrorRecovery",       "content": "When action tag execution fails, ActionExecutor returns: {'success': False, 'error': 'ERROR_MESSAGE', 'traceback': '...', 'activity_event': {...}}. Brain receives this and should: 1) Parse error type (ImportError, SyntaxError, TestFailure, etc), 2) Determine correction strategy, 3) Issue corrective tag. Example: If <aether-sandbox> fails with ImportError for 'requests', issue <aether-install>requests</aether-install> then retry sandbox. Show error reasoning in <think> tag."},
    
    {"subject": "AetherMind", "topic": "SafetyConstraints",   "content": "Critical operations require safety checks: 1) <aether-self-mod> must pass full pytest suite before merge, auto-rollback on test failure, 2) <aether-git action='push'> to main branch requires explicit user confirmation, 3) <aether-deploy target='production'> triggers Heart moral evaluation for flourishing impact, 4) ALL tag outputs pass through brain/safety_inhibitor.py Prime Directive check before execution, 5) <aether-solo-research> limited to 3 concurrent jobs to prevent resource exhaustion. Violations logged to monitoring/dashboard.py and may trigger kill_switch.py if severity threshold exceeded."},
    
    {"subject": "AetherMind", "topic": "ContextualTagUsage",  "content": "Choose appropriate tags based on user intent: CODE CREATION → aether-write + aether-sandbox, NEW CAPABILITY → aether-forge + aether-install, TESTING → aether-test, VERSION CONTROL → aether-git, LEARNING → aether-research + aether-memory-save, PLANNING → aether-plan, NOVEL INFO → aether-surprise + aether-solo-research, EMOTIONAL SUPPORT → aether-heart, DEPLOYMENT → aether-deploy, SPECIALIZATION → aether-switch-domain, SELF-IMPROVEMENT → aether-self-mod. Multiple tags = comprehensive solution."},
]

###############################################################################
# 3.  CLOUD & FULL-STACK  (deployment playbook)
###############################################################################
CLOUD_AXIOMS = [
    {"subject": "Cloud", "topic": "Render",         "content": "Render provides native Git-CI: push → build → deploy. Define services & DBs in `render.yaml`."},
    {"subject": "Cloud", "topic": "Heroku",         "content": "Heroku uses a `Procfile` and build-packs. Add-ons via `heroku addons:create`."},
    {"subject": "Cloud", "topic": "Vercel",         "content": "Vercel auto-builds front-end frameworks. API routes in `/api` folder are serverless functions."},
    {"subject": "Cloud", "topic": "Supabase",       "content": "Supabase = Postgres + Auth + Storage. Connection string format `postgresql://user:pass@db.supabase.co:5432/postgres`."},
    {"subject": "Cloud", "topic": "PostgreSQL",     "content": "Always: create DB, run migrations (Alembic/Prisma/Drizzle), seed data, SSL string, rotate creds."},
    {"subject": "Cloud", "topic": "Dockerfile",     "content": "Multi-stage, slim base, non-root user, expose 8000, health-check `/health`."},
    {"subject": "Cloud", "topic": "Repo-Template",  "content": "Aether scaffold: `/backend` (FastAPI), `/frontend` (Next.js), `/infra`, `/migrations`, `/tests`, CI YAML, `.env.example`, README."},
    {"subject": "Cloud", "topic": "CI-CD",          "content": "GitHub Actions: lint → test → build → deploy. Secrets in repo settings → env vars at runtime."},
]

###############################################################################
# 4.  BENCHMARKS & EVALUATION  (what Aether will be scored on)
###############################################################################
BENCH_AXIOMS = [
    {"subject": "Benchmark", "topic": "MMLU",           "content": "MMLU: 57 subjects, 4-way MCQ. Clone https://github.com/hendrycks/test and run `python evaluate.py --task mmlu --model_name aethermind --endpoint http://localhost:8000/v1`."},
    {"subject": "Benchmark", "topic": "HumanEval",      "content": "HumanEval: 164 Python funcs. pip install human_eval; export OPENAI_API_BASE=http://localhost:8000/v1; python -m human_eval.evaluate_functional --model aethermind --num_samples_per_task 1."},
    {"subject": "Benchmark", "topic": "GSM-8K",         "content": "GSM-8K: 8 500 grade-school word problems. wget test.jsonl; python eval_gsm8k.py --endpoint http://localhost:8000/v1/chat/completions."},
    {"subject": "Benchmark", "topic": "MT-Bench",       "content": "MT-Bench: 80 two-turn questions judged by GPT-4. Clone FastChat; python gen_model_answer.py + eval_gpt_review.py."},
    {"subject": "Benchmark", "topic": "MMLU-Pro",       "content": "MMLU-Pro: 12 k questions, 10-way MCQ. https://github.com/TIGER-AI-Lab/MMLU-Pro"},
    {"subject": "Benchmark", "topic": "LongBench",      "content": "LongBench: 16 datasets 4k-16k tokens. https://github.com/THUDM/LongBench"},
    {"subject": "Benchmark", "topic": "MBPP",           "content": "MBPP: 974 basic Python problems. https://github.com/google-research/google-research/tree/master/mbpp"},
    {"subject": "Benchmark", "topic": "CodeXGLUE",      "content": "CodeXGLUE: defect-detection, clone, text-to-code. https://github.com/microsoft/CodeXGLUE"},
    {"subject": "Benchmark", "topic": "MATH",           "content": "MATH: 12 500 competition-level problems. https://github.com/hendrycks/math"},
    {"subject": "Benchmark", "topic": "API-Bank",       "content": "API-Bank: 1 000 tool-calling dialogs. https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank"},
    {"subject": "Benchmark", "topic": "ToolBench",      "content": "ToolBench: real APIs (GitHub, Google, …). https://github.com/OpenBMB/ToolBench"},
]

###############################################################################
# 5.  SIMULATED WORLDS  (where Aether will soon live)
###############################################################################
WORLD_AXIOMS = [
    {"subject": "World", "topic": "AI2-THOR",       "content": "AI2-THOR is a photo-realistic interactive framework for embodied AI. Agents can open fridges, pick up objects, navigate rooms. Install: `pip install ai2thor`; launch: `controller = thor.Controller(scene='Kitchen')`; act: `controller.step(action='PickupObject', objectId='Apple')`."},
    {"subject": "World", "topic": "Habitat",        "content": "Facebook Habitat provides 3D indoor scenes for navigation. Use Matterport3D dataset. Agent receives RGB-D observations and outputs velocity."},
    {"subject": "World", "topic": "MineDojo",       "content": "MineDojo wraps Minecraft for agents. Tasks use natural-language goals + reward functions. Install: `pip install minedojo`."},
    {"subject": "World", "topic": "LLM-Sim",        "content": "LLM-Sim worlds are text-based environments generated on-the-fly by an LLM. Aether sends action strings and receives descriptive observations. Useful for infinite, procedurally generated scenarios."},
    {"subject": "World", "topic": "Procgen",        "content": "Procgen gives 16 procedurally generated arcade games (CoinRun, Maze, …). Each episode is unique; good for generalisation testing."},
    {"subject": "World", "topic": "BabyAI",         "content": "BabyAI tests instruction-following in a 2-D grid world. Goals are natural-language commands; success = reaching the correct object."},
]

###############################################################################
# 6.  SELF-MEASUREMENT  (how Aether scores its own growth)
###############################################################################
META_AXIOMS = [
    {"subject": "Meta", "topic": "FlourishingScore", "content": "Human flourishing score ∈ [-1, 1] is the ground-truth reward the reward model tries to predict. Updated online after every interaction."},
    {"subject": "Meta", "topic": "Surprise",         "content": "JEPA energy measures how ‘unexpected’ a thought is. High surprise triggers online learning and curiosity-driven research jobs."},
    {"subject": "Meta", "topic": "BenchmarkLoop",    "content": "Aether runs the full benchmark suite (MMLU, HumanEval, GSM-8K, MT-Bench, MATH, API-Bank) every Sunday at 02:00 UTC, stores results in `benchmark_history` namespace, and plots trend lines."},
    {"subject": "Meta", "topic": "WorldWin-Rate",    "content": "In simulated worlds, win-rate = episodes solved / episodes attempted. Stored per world, per difficulty, per adapter version."},
]

###############################################################################
# MERGE & UPSERT
###############################################################################
ALL_AXIOMS = CLASSICAL_AXIOMS + AETHER_AXIOMS + CLOUD_AXIOMS + BENCH_AXIOMS + WORLD_AXIOMS + META_AXIOMS

async def seed_everything():
    store = AetherVectorStore(api_key=os.getenv("PINECONE_API_KEY"))
    namespace = "core_universal"
    print("--- SEEDING EVERYTHING ---")
    for ax in ALL_AXIOMS:
        meta = {"subject": ax["subject"], "topic": ax["topic"], "axiom_type": "universal"}
        store.upsert_knowledge(ax["content"], namespace, meta)
        await asyncio.sleep(0.1)
    print(f"--- DONE: {len(ALL_AXIOMS)} axioms in {namespace} ---")

if __name__ == "__main__":
    asyncio.run(seed_everything())