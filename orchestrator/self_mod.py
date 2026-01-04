"""
Path: orchestrator/self_mod.py
Safe self-modification pipeline.
"""
import os, subprocess, tempfile, shutil, git, json, uuid
from body.adapter_base import BodyAdapter
from loguru import logger

REPO_PATH = "/app"           # container path
TEST_CMD  = " -m pytest tests/ -x --tb=short"

class SelfModAdapter(BodyAdapter):
    async def execute(self, intent: str) -> str:
        """
        Intent JSON: {"file": "orchestrator/router.py", "patch": "@@ -10,3 +10,4 @@ ..."}
        """
        spec = json.loads(intent)
        file    = spec["file"]
        patch   = spec["patch"]

        branch = f"agent-{uuid.uuid4().hex[:8]}"
        try:
            repo = git.Repo(REPO_PATH)
            repo.git.checkout('-b', branch)

            # apply patch
            with tempfile.NamedTemporaryFile("w", suffix=".patch") as f:
                f.write(patch)
                f.flush()
                subprocess.check_call(["git", "apply", f.name], cwd=REPO_PATH)

            # tests
            result = subprocess.run(TEST_CMD.split(), cwd=REPO_PATH,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                repo.git.add(update=True)
                repo.index.commit(f"agent self-mod {branch}")
                repo.git.checkout("main")
                repo.git.merge(branch)
                # hot-reload gunicorn
                subprocess.call(["kill", "-HUP", "1"])
                return "self-mod success – tests pass & merged"
            else:
                repo.git.checkout("main")
                repo.git.branch("-D", branch)
                return f"self-mod rejected – tests failed:\n{result.stdout}\n{result.stderr}"
        except Exception as e:
            logger.exception("self-mod crash")
            return f"self-mod crash: {e}"

    async def add_tool_and_commit(self, name: str, code: str, tests: str) -> str:
        """Agent calls this after ToolForge.generate + test pass"""
        patch = f'--- /dev/null\\n+++ b/body/adapters/{name}.py\\n@@ -0,0 +1 @@\\n+{code}'
        test_patch = f'--- /dev/null\\n+++ b/tests/agent_generated/test_{name}.py\\n@@ -0,0 +1 @@\\n+{tests}'
        full_patch = patch + "\\n" + test_patch
        return await self.execute(json.dumps({"file": f"body/adapters/{name}.py", "patch": full_patch}))
