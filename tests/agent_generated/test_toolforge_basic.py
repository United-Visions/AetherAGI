
import pytest
import os
import json
from body.adapters.toolforge_adapter import ToolForgeAdapter

# Mock objects to avoid full environment dependency during test
class MockRouter:
    def __init__(self):
        self.adapters = {}

class MockStore:
    pass

class MockPromoter:
    def __init__(self, store, gate):
        pass
    def record_new_tool(self, name, metadata):
        pass

# We need to patch the imports inside ToolForgeAdapter._load to use our mocks
# Since the imports happen inside the function, we can patch sys.modules or mock the function.
# A simpler way for this unit test is to mock the imports in sys.modules *before* the function runs,
# but since the import is inside the function, we can rely on `unittest.mock.patch`.

from unittest.mock import MagicMock, patch

@pytest.fixture
def adapter():
    return ToolForgeAdapter()

def test_discover_pypi(adapter):
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "info": {"name": "testpkg", "version": "1.0.0", "summary": "A test package"}
        }
        result = adapter.execute(json.dumps({"action": "discover", "name": "testpkg"}))
        assert "testpkg 1.0.0: A test package" in result

def test_generate_code(adapter):
    name = "test_tool"
    schema = {"url": "http://example.com"}
    result = adapter.execute(json.dumps({
        "action": "generate",
        "name": name,
        "schema": schema
    }))
    assert f"generated /tmp/agent_venv/{name}.py" in result
    assert os.path.exists(f"/tmp/agent_venv/{name}.py")

# Clean up
def teardown_module(module):
    if os.path.exists("/tmp/agent_venv/test_tool.py"):
        os.remove("/tmp/agent_venv/test_tool.py")
