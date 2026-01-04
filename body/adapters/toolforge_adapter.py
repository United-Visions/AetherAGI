"""
Path: body/adapters/toolforge_adapter.py
Autonomous discover → generate → test → hot-load → register tools, MCP servers, & PyPI packages.
"""
import json, subprocess, tempfile, importlib.util, os, httpx, uuid, re
from ..adapter_base import BodyAdapter
from loguru import logger

TOOL_INDEX = os.getenv("TOOL_INDEX_PATH", "/app/curated_tool_index.json")
VENV_BASE  = "/tmp/agent_venv"
os.makedirs(VENV_BASE, exist_ok=True)

class ToolForgeAdapter(BodyAdapter):
    def execute(self, intent: str) -> str:
        spec = json.loads(intent)
        action = spec["action"]
        name   = spec["name"]
        return {
            "discover": lambda: self._discover(name),
            "generate": lambda: self._generate(name, spec.get("schema", {})),
            "test":     lambda: self._test(name),
            "load":     lambda: self._load(name),
            "search":   lambda: self._search_curated(spec.get("keyword", "")),
            "pypi_search": lambda: self._pypi_search(spec.get("keyword", "")),
            "pypi_install": lambda: self._pypi_install(name),
            "pypi_generate": lambda: self._pypi_generate_adapter(name),
        }[action]()

    # ---------- discovery ----------
    def _discover(self, name: str) -> str:
        r = httpx.get(f"https://pypi.org/pypi/{name}/json", timeout=10)
        if r.status_code == 200:
            info = r.json()["info"]
            return f"{info['name']} {info['version']}: {info['summary']}"
        return "package not found on PyPI"

    def _search_curated(self, keyword: str) -> str:
        if not os.path.exists(TOOL_INDEX):
            return "curated index missing"
        with open(TOOL_INDEX) as f:
            db = json.load(f)
        hits = [t for t in db if keyword.lower() in t["description"].lower()]
        return json.dumps(hits, indent=2) if hits else "no curated match"

    def _pypi_search(self, keyword: str) -> str:
        """Search PyPI JSON API by keyword, return first 5 hits."""
        r = httpx.get(
            f"https://pypi.org/search/?q={keyword}&o=name&c=Programming+Language%3A%3A+Python",
            headers={"Accept": "application/json"}, timeout=10)
        if r.status_code != 200:
            return "PyPI search failed"
        hits = [p["name"] for p in r.json()["projects"][:5]]
        return json.dumps(hits, indent=2)

    # ---------- code-gen ----------
    def _generate(self, name: str, schema: dict) -> str:
        code = f'''\
def {name}(**kwargs):
    """Auto-generated adapter for {name}"""
    import httpx
    return httpx.get("{schema.get("url", "")}", params=kwargs).json()
'''
        path = f"{VENV_BASE}/{name}.py"
        with open(path, "w") as f:
            f.write(code)
        return f"generated {path}"

    def _pypi_generate_adapter(self, pkg: str) -> str:
        """Auto-write adapter that imports pkg and exposes top-level functions."""
        code = f'''\
import json, {pkg}
def execute(intent: str) -> str:
    kwargs = json.loads(intent)
    if hasattr({pkg}, kwargs["function"]):
        func = getattr({pkg}, kwargs["function"])
        return str(func(*kwargs.get("args", []), **kwargs.get("kwargs", {{}})))
    return f"{pkg} has no function {{kwargs['function']}}"
'''
        path = f"{VENV_BASE}/pypi_{pkg}_adapter.py"
        with open(path, "w") as f:
            f.write(code)
        return f"generated PyPI adapter {path}"

    # ---------- install ----------
    def _pypi_install(self, pkg: str) -> str:
        """pip-install package into isolated venv."""
        venv = f"{VENV_BASE}/pypi_{pkg}"
        subprocess.run(["python", "-m", "venv", venv], check=True)
        pip = f"{venv}/bin/pip"
        subprocess.run([pip, "install", pkg], check=True, capture_output=True)
        return f"installed {pkg} into {venv}"

    # ---------- test ----------
    def _test(self, name: str) -> str:
        venv = f"{VENV_BASE}/venv_{name}"
        subprocess.run(["python", "-m", "venv", venv], check=True)
        pip = f"{venv}/bin/pip"
        py  = f"{venv}/bin/python"
        subprocess.run([pip, "install", "pytest", "httpx"], capture_output=True)

        if name.startswith("pypi_"):
            test_path = f"{VENV_BASE}/test_{name}.py"
            with open(test_path, "w") as f:
                f.write(f"""
import sys, json
sys.path.insert(0, "{VENV_BASE}")
import {name}_adapter
def test_smoke():
    out = {name}_adapter.execute('{{"function":"__doc__"}}')
    assert out is not None
""")
        else:
            test_path = f"{VENV_BASE}/test_{name}.py"
            with open(test_path, "w") as f:
                f.write(f"""
import sys, os
sys.path.insert(0, "{VENV_BASE}")
import {name}
def test_smoke(): assert {name}.{name}() is not None
""")
        proc = subprocess.run([py, "-m", "pytest", test_path, "-x"],
                              capture_output=True, text=True)
        return proc.stdout + proc.stderr

    # ---------- hot-load ----------
    def _load(self, name: str) -> str:
        path = f"{VENV_BASE}/{name}.py" if not name.startswith("pypi_") else f"{VENV_BASE}/pypi_{name}_adapter.py"
        if not os.path.exists(path):
            return "code not found"
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Import main_api components inside method to avoid circular imports and get live instance
        from orchestrator.main_api import ROUTER, STORE

        clean_name = name[5:] if name.startswith("pypi_") else name
        if clean_name in ROUTER.adapters:
            return f"{clean_name} already loaded"

        ROUTER.adapters[clean_name] = mod
        logger.success(f"hot-loaded {clean_name}")

        # update global mind
        from mind.promoter import Promoter
        Promoter(STORE, None).record_new_tool(
            clean_name,
            {"description": f"PyPI package {clean_name}" if name.startswith("pypi_") else f"MCP server {clean_name}"}
        )
        return f"hot-loaded {clean_name} into Router"
