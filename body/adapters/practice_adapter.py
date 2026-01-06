"""
Path: body/adapters/practice_adapter.py
Executes generated procedures and returns *outcomes* as text.
"""
import json, asyncio, tempfile, subprocess, traceback, sys, os
from ..adapter_base import BodyAdapter
from loguru import logger

class PracticeAdapter(BodyAdapter):
    async def execute(self, intent: str) -> str:
        """
        Intent JSON: {"language": "python"|"bash", "code": "...", "tests": ["assert f(3)==9"]}
        """
        logger.info(f"PracticeAdapter executing: {intent[:200]}...")
        try:
            spec = json.loads(intent)
            lang = spec["language"]
            code = spec["code"]
            tests = spec.get("tests", [])
            
            logger.debug(f"Language: {lang}, Code length: {len(code)} bytes")
            
            if lang == "python":
                result = await self._run_python(code, tests)
                logger.info(f"Python execution complete: {len(result)} bytes output")
                return result
            if lang == "bash":
                result = await self._run_bash(code)
                logger.info(f"Bash execution complete: {len(result)} bytes output")
                return result
            
            logger.warning(f"Unknown language: {lang}")
            return f"PracticeAdapter: unknown language {lang}"
        except Exception as e:
            logger.exception("PracticeAdapter crashed during execution")
            return f"practice crash: {traceback.format_exc()}"

    async def _run_python(self, code: str, tests: list) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n# --- tests ---\n")
            for t in tests:
                f.write(t + "\n")
            f.flush()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, f.name, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    async def _run_bash(self, script: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            script, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
