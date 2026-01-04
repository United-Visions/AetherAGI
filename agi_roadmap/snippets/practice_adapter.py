
snippets/practice_adapter.py


"""
Path: body/adapters/practice_adapter.py
Executes generated procedures and returns *outcomes* as text.
"""
import json, asyncio, tempfile, subprocess, traceback
from ..adapter_base import BodyAdapter
from loguru import logger

class PracticeAdapter(BodyAdapter):
    async def execute(self, intent: str) -> str:
        """
        Intent JSON: {"language": ""|"bash", "code": "...", "tests": ["assert f(3)==9"]}
        """
        try:
            spec = json.loads(intent)
            lang = spec["language"]
            code = spec["code"]
            tests = spec.get("tests", [])
            if lang == "":
                return await self._run_(code, tests)
            if lang == "bash":
                return await self._run_bash(code)
            return f"PracticeAdapter: unknown language {lang}"
        except Exception as e:
            logger.exception("practice crash")
            return f"practice crash: {traceback.format_exc()}"

    async def _run_(self, code: str, tests: list) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n# --- tests ---\n")
            for t in tests:
                f.write(t + "\n")
            f.flush()
            proc = await asyncio.create_subprocess_exec(
                "", f.name, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"

    async def _run_bash(self, script: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            script, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"


#Wire in router.py


if settings.practice_adapter:
    from body.adapters.practice_adapter import PracticeAdapter
    self.adapters["practice"] = PracticeAdapter()