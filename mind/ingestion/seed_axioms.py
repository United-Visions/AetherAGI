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