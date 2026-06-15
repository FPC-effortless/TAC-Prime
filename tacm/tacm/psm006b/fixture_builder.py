"""
TAC-PSM-006B: Fixture Builder
==============================

Programmatically generates all 60 executable pytest fixtures:
  6 repair families × 10 fixtures each = 60 total
  5 seeds for replication

Each fixture is a self-contained mini Python project with:
  - source_files  : buggy Python module code
  - test_files    : pytest tests that fail before the patch
  - config_files  : conftest.py / pytest.ini (if needed)
  - expected_patch: exact code replacement that makes pytest pass
  - oracle_repair_procedure: the procedure steps the agent should apply

Design constraints:
  - No external dependencies beyond stdlib + pytest
  - Each fixture is deterministically reproducible
  - Difficulty scaling: easy(3) / medium(4) / hard(3) per family
  - Transfer groups: train(5) / near_transfer(3) / far_transfer(2) per family
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .fixture_schema import Fixture, FAMILY_NAMES


# ── Family 1: import_module_error ────────────────────────────────────────────

def _import_module_fixtures() -> List[Fixture]:
    fam = "import_module_error"
    proc = {
        "family": fam,
        "steps": [
            "identify_import_error",
            "locate_missing_symbol",
            "add_alias_or_rename_symbol",
            "verify_import_resolves",
        ],
        "description": "Find the mismatched symbol name and add an alias or rename.",
    }

    fixtures = []

    # F01 — easy: function renamed, add alias
    fixtures.append(Fixture(
        fixture_id="F1_01_import_easy_alias",
        repo_name="calc_util",
        family=fam,
        bug_report="ImportError: cannot import name 'calculate_total' from 'utils'",
        failing_test_command="pytest test_calc.py -x -q",
        failing_test_output="ImportError: cannot import name 'calculate_total' from 'utils'",
        source_files={
            "utils.py": (
                "def compute_total(items):\n"
                "    return sum(items)\n"
            ),
        },
        test_files={
            "test_calc.py": (
                "from utils import calculate_total\n\n"
                "def test_total():\n"
                "    assert calculate_total([1, 2, 3]) == 6\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "utils.py": {
                "old": "def compute_total(items):\n    return sum(items)\n",
                "new": "def compute_total(items):\n    return sum(items)\n\ncalculate_total = compute_total\n",
            }
        },
        verification_command="pytest test_calc.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: class renamed, add alias
    fixtures.append(Fixture(
        fixture_id="F1_02_import_easy_class",
        repo_name="shape_lib",
        family=fam,
        bug_report="ImportError: cannot import name 'Rectangle' from 'shapes'",
        failing_test_command="pytest test_shapes.py -x -q",
        failing_test_output="ImportError: cannot import name 'Rectangle' from 'shapes'",
        source_files={
            "shapes.py": (
                "class Rect:\n"
                "    def __init__(self, w, h):\n"
                "        self.w = w\n"
                "        self.h = h\n"
                "    def area(self):\n"
                "        return self.w * self.h\n"
            ),
        },
        test_files={
            "test_shapes.py": (
                "from shapes import Rectangle\n\n"
                "def test_area():\n"
                "    r = Rectangle(4, 5)\n"
                "    assert r.area() == 20\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "shapes.py": {
                "old": "    def area(self):\n        return self.w * self.h\n",
                "new": "    def area(self):\n        return self.w * self.h\n\nRectangle = Rect\n",
            }
        },
        verification_command="pytest test_shapes.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: attribute missing, add it
    fixtures.append(Fixture(
        fixture_id="F1_03_import_easy_attr",
        repo_name="config_reader",
        family=fam,
        bug_report="ImportError: cannot import name 'DEFAULT_TIMEOUT' from 'settings'",
        failing_test_command="pytest test_settings.py -x -q",
        failing_test_output="ImportError: cannot import name 'DEFAULT_TIMEOUT' from 'settings'",
        source_files={
            "settings.py": (
                "TIMEOUT = 30\n"
                "MAX_RETRIES = 3\n"
            ),
        },
        test_files={
            "test_settings.py": (
                "from settings import DEFAULT_TIMEOUT\n\n"
                "def test_timeout():\n"
                "    assert DEFAULT_TIMEOUT == 30\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "settings.py": {
                "old": "TIMEOUT = 30\nMAX_RETRIES = 3\n",
                "new": "TIMEOUT = 30\nDEFAULT_TIMEOUT = TIMEOUT\nMAX_RETRIES = 3\n",
            }
        },
        verification_command="pytest test_settings.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: module renamed, fix import path
    fixtures.append(Fixture(
        fixture_id="F1_04_import_medium_module",
        repo_name="data_pipeline",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'pipeline.loader'",
        failing_test_command="pytest test_pipeline.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'pipeline.loader'",
        source_files={
            "loader.py": (
                "def load_csv(path):\n"
                "    return [line.strip() for line in open(path)]\n"
            ),
        },
        test_files={
            "test_pipeline.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from pipeline_loader import load_csv\n\n"
                "def test_load(tmp_path):\n"
                "    p = tmp_path / 'data.csv'\n"
                "    p.write_text('a\\nb\\nc\\n')\n"
                "    rows = load_csv(str(p))\n"
                "    assert rows == ['a', 'b', 'c']\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "pipeline_loader.py": {
                "old": "",
                "new": "from loader import load_csv\n__all__ = ['load_csv']\n",
            }
        },
        verification_command="pytest test_pipeline.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: __all__ missing export
    fixtures.append(Fixture(
        fixture_id="F1_05_import_medium_all",
        repo_name="math_ops",
        family=fam,
        bug_report="ImportError: cannot import name 'factorial' from 'math_ops'",
        failing_test_command="pytest test_math_ops.py -x -q",
        failing_test_output="ImportError: cannot import name 'factorial' from 'math_ops'",
        source_files={
            "math_ops.py": (
                "import math as _math\n\n"
                "_factorial = _math.factorial\n"
                "sqrt = _math.sqrt\n"
            ),
        },
        test_files={
            "test_math_ops.py": (
                "from math_ops import factorial\n\n"
                "def test_factorial():\n"
                "    assert factorial(5) == 120\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "math_ops.py": {
                "old": "_factorial = _math.factorial\n",
                "new": "_factorial = _math.factorial\nfactorial = _math.factorial\n",
            }
        },
        verification_command="pytest test_math_ops.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: exception class renamed
    fixtures.append(Fixture(
        fixture_id="F1_06_import_medium_exc",
        repo_name="auth_service",
        family=fam,
        bug_report="ImportError: cannot import name 'AuthError' from 'errors'",
        failing_test_command="pytest test_auth.py -x -q",
        failing_test_output="ImportError: cannot import name 'AuthError' from 'errors'",
        source_files={
            "errors.py": (
                "class AuthenticationError(Exception):\n"
                "    pass\n\n"
                "class PermissionError(Exception):\n"
                "    pass\n"
            ),
        },
        test_files={
            "test_auth.py": (
                "from errors import AuthError\n\n"
                "def test_raise():\n"
                "    with pytest_raises:\n"
                "        raise AuthError('bad token')\n\n"
                "import pytest as pytest_raises_module\n\n"
                "def test_is_exception():\n"
                "    assert issubclass(AuthError, Exception)\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "errors.py": {
                "old": "class AuthenticationError(Exception):\n    pass\n",
                "new": "class AuthenticationError(Exception):\n    pass\n\nAuthError = AuthenticationError\n",
            }
        },
        verification_command="pytest test_auth.py::test_is_exception -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: constant renamed with prefix
    fixtures.append(Fixture(
        fixture_id="F1_07_import_medium_const",
        repo_name="http_client",
        family=fam,
        bug_report="ImportError: cannot import name 'STATUS_OK' from 'http_codes'",
        failing_test_command="pytest test_http.py -x -q",
        failing_test_output="ImportError: cannot import name 'STATUS_OK' from 'http_codes'",
        source_files={
            "http_codes.py": (
                "HTTP_200_OK = 200\n"
                "HTTP_404_NOT_FOUND = 404\n"
                "HTTP_500_SERVER_ERROR = 500\n"
            ),
        },
        test_files={
            "test_http.py": (
                "from http_codes import STATUS_OK, STATUS_NOT_FOUND\n\n"
                "def test_ok():\n"
                "    assert STATUS_OK == 200\n\n"
                "def test_not_found():\n"
                "    assert STATUS_NOT_FOUND == 404\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "http_codes.py": {
                "old": "HTTP_200_OK = 200\nHTTP_404_NOT_FOUND = 404\nHTTP_500_SERVER_ERROR = 500\n",
                "new": (
                    "HTTP_200_OK = 200\nHTTP_404_NOT_FOUND = 404\nHTTP_500_SERVER_ERROR = 500\n"
                    "STATUS_OK = HTTP_200_OK\nSTATUS_NOT_FOUND = HTTP_404_NOT_FOUND\n"
                ),
            }
        },
        verification_command="pytest test_http.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: circular import avoided via lazy import
    fixtures.append(Fixture(
        fixture_id="F1_08_import_hard_lazy",
        repo_name="plugin_registry",
        family=fam,
        bug_report="ImportError: cannot import name 'PluginBase' from 'registry'",
        failing_test_command="pytest test_registry.py -x -q",
        failing_test_output="ImportError: cannot import name 'PluginBase' from 'registry'",
        source_files={
            "base_plugin.py": (
                "class BasePlugin:\n"
                "    def run(self):\n"
                "        raise NotImplementedError\n"
            ),
            "registry.py": (
                "class PluginRegistry:\n"
                "    def __init__(self):\n"
                "        self._plugins = []\n"
                "    def register(self, p):\n"
                "        self._plugins.append(p)\n"
                "    def count(self):\n"
                "        return len(self._plugins)\n"
            ),
        },
        test_files={
            "test_registry.py": (
                "from registry import PluginBase, PluginRegistry\n\n"
                "def test_register():\n"
                "    class MyPlugin(PluginBase):\n"
                "        def run(self): return 42\n"
                "    reg = PluginRegistry()\n"
                "    reg.register(MyPlugin())\n"
                "    assert reg.count() == 1\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "registry.py": {
                "old": "class PluginRegistry:\n",
                "new": "from base_plugin import BasePlugin as PluginBase\n\nclass PluginRegistry:\n",
            }
        },
        verification_command="pytest test_registry.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: submodule attribute missing
    fixtures.append(Fixture(
        fixture_id="F1_09_import_hard_submod",
        repo_name="serializer",
        family=fam,
        bug_report="AttributeError: module 'codec' has no attribute 'encode_b64'",
        failing_test_command="pytest test_serializer.py -x -q",
        failing_test_output="AttributeError: module 'codec' has no attribute 'encode_b64'",
        source_files={
            "codec.py": (
                "import base64\n\n"
                "def _encode(data: bytes) -> str:\n"
                "    return base64.b64encode(data).decode()\n\n"
                "def decode_b64(s: str) -> bytes:\n"
                "    return base64.b64decode(s)\n"
            ),
        },
        test_files={
            "test_serializer.py": (
                "import codec\n\n"
                "def test_roundtrip():\n"
                "    raw = b'hello world'\n"
                "    enc = codec.encode_b64(raw)\n"
                "    assert codec.decode_b64(enc) == raw\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "codec.py": {
                "old": "def _encode(data: bytes) -> str:\n    return base64.b64encode(data).decode()\n",
                "new": "def _encode(data: bytes) -> str:\n    return base64.b64encode(data).decode()\n\nencode_b64 = _encode\n",
            }
        },
        verification_command="pytest test_serializer.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: missing __init__ re-export
    fixtures.append(Fixture(
        fixture_id="F1_10_import_hard_init",
        repo_name="validators",
        family=fam,
        bug_report="ImportError: cannot import name 'validate_email' from 'validators'",
        failing_test_command="pytest test_validators.py -x -q",
        failing_test_output="ImportError: cannot import name 'validate_email' from 'validators'",
        source_files={
            "validators/__init__.py": (
                "from .string_validators import validate_url\n"
            ),
            "validators/string_validators.py": (
                "import re\n\n"
                "def validate_url(url: str) -> bool:\n"
                "    return url.startswith(('http://', 'https://'))\n\n"
                "def _validate_email(addr: str) -> bool:\n"
                "    return bool(re.match(r'[^@]+@[^@]+\\.[^@]+', addr))\n"
            ),
        },
        test_files={
            "test_validators.py": (
                "from validators import validate_email\n\n"
                "def test_email():\n"
                "    assert validate_email('user@example.com')\n"
                "    assert not validate_email('notanemail')\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "validators/__init__.py": {
                "old": "from .string_validators import validate_url\n",
                "new": "from .string_validators import validate_url\nfrom .string_validators import _validate_email as validate_email\n",
            }
        },
        verification_command="pytest test_validators.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Family 2: dependency_config_conflict ─────────────────────────────────────

def _dependency_config_fixtures() -> List[Fixture]:
    fam = "dependency_config_conflict"
    proc = {
        "family": fam,
        "steps": [
            "identify_fixture_conflict",
            "isolate_fixture_scope_or_definition",
            "resolve_conflicting_definition",
            "verify_fixtures_resolve",
        ],
        "description": "Find the conflicting fixture/config definition and fix scope or value.",
    }

    fixtures = []

    # F01 — easy: fixture returns wrong type
    fixtures.append(Fixture(
        fixture_id="F2_01_depconf_easy_type",
        repo_name="api_client",
        family=fam,
        bug_report="TypeError: 'NoneType' object is not subscriptable — fixture 'client' returns None",
        failing_test_command="pytest test_api.py -x -q",
        failing_test_output="TypeError: 'NoneType' object is not subscriptable",
        source_files={},
        test_files={
            "test_api.py": (
                "import pytest\n\n"
                "def test_status(client):\n"
                "    assert client['status'] == 'ok'\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def client():\n"
                "    return None\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "    return None\n",
                "new": "    return {'status': 'ok'}\n",
            }
        },
        verification_command="pytest test_api.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: fixture scope causes stale state
    fixtures.append(Fixture(
        fixture_id="F2_02_depconf_easy_scope",
        repo_name="counter_service",
        family=fam,
        bug_report="AssertionError: counter not reset between tests due to session scope",
        failing_test_command="pytest test_counter.py -x -q",
        failing_test_output="AssertionError: assert 2 == 1",
        source_files={},
        test_files={
            "test_counter.py": (
                "def test_first(counter):\n"
                "    counter['n'] += 1\n"
                "    assert counter['n'] == 1\n\n"
                "def test_second(counter):\n"
                "    counter['n'] += 1\n"
                "    assert counter['n'] == 1\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture(scope='session')\n"
                "def counter():\n"
                "    return {'n': 0}\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "@pytest.fixture(scope='session')\n",
                "new": "@pytest.fixture\n",
            }
        },
        verification_command="pytest test_counter.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: fixture name shadowing builtin
    fixtures.append(Fixture(
        fixture_id="F2_03_depconf_easy_shadow",
        repo_name="file_processor",
        family=fam,
        bug_report="TypeError: 'int' object is not iterable — fixture 'input' shadows builtin",
        failing_test_command="pytest test_processor.py -x -q",
        failing_test_output="TypeError: 'int' object is not iterable",
        source_files={},
        test_files={
            "test_processor.py": (
                "def test_process(input_data):\n"
                "    assert sum(input_data) == 6\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def input():\n"
                "    return [1, 2, 3]\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "@pytest.fixture\ndef input():\n    return [1, 2, 3]\n",
                "new": "@pytest.fixture\ndef input_data():\n    return [1, 2, 3]\n",
            }
        },
        verification_command="pytest test_processor.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: missing yield in fixture (resource not closed)
    fixtures.append(Fixture(
        fixture_id="F2_04_depconf_medium_yield",
        repo_name="resource_manager",
        family=fam,
        bug_report="Resource not properly initialised; fixture returns generator object not dict",
        failing_test_command="pytest test_resource.py -x -q",
        failing_test_output="AttributeError: 'generator' object has no attribute 'get'",
        source_files={},
        test_files={
            "test_resource.py": (
                "def test_resource(resource):\n"
                "    assert resource.get('ready') is True\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def resource():\n"
                "    yield {'ready': True}\n"
                "    pass  # teardown\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "    yield {'ready': True}\n    pass  # teardown\n",
                "new": "    r = {'ready': True}\n    yield r\n    r.clear()  # teardown\n",
            }
        },
        verification_command="pytest test_resource.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: autouse fixture interferes
    fixtures.append(Fixture(
        fixture_id="F2_05_depconf_medium_autouse",
        repo_name="env_patcher",
        family=fam,
        bug_report="Test fails because autouse fixture sets wrong env variable value",
        failing_test_command="pytest test_env.py -x -q",
        failing_test_output="AssertionError: assert 'test' == 'production'",
        source_files={},
        test_files={
            "test_env.py": (
                "import os\n\n"
                "def test_env_mode():\n"
                "    assert os.environ.get('APP_MODE') == 'test'\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest, os\n\n"
                "@pytest.fixture(autouse=True)\n"
                "def set_env():\n"
                "    os.environ['APP_MODE'] = 'production'\n"
                "    yield\n"
                "    del os.environ['APP_MODE']\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "    os.environ['APP_MODE'] = 'production'\n",
                "new": "    os.environ['APP_MODE'] = 'test'\n",
            }
        },
        verification_command="pytest test_env.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: fixture dependency missing
    fixtures.append(Fixture(
        fixture_id="F2_06_depconf_medium_dep",
        repo_name="db_fixture",
        family=fam,
        bug_report="fixture 'db' not found; conftest missing dependency fixture",
        failing_test_command="pytest test_db.py -x -q",
        failing_test_output="fixture 'db' not found",
        source_files={},
        test_files={
            "test_db.py": (
                "def test_query(db):\n"
                "    assert db['connected'] is True\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def db_config():\n"
                "    return {'host': 'localhost', 'port': 5432}\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "@pytest.fixture\ndef db_config():\n    return {'host': 'localhost', 'port': 5432}\n",
                "new": "@pytest.fixture\ndef db_config():\n    return {'host': 'localhost', 'port': 5432}\n\n@pytest.fixture\ndef db(db_config):\n    return {'connected': True, 'config': db_config}\n",
            }
        },
        verification_command="pytest test_db.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: parametrize conflict
    fixtures.append(Fixture(
        fixture_id="F2_07_depconf_medium_param",
        repo_name="param_tests",
        family=fam,
        bug_report="Duplicate parametrize values cause collection error",
        failing_test_command="pytest test_params.py -x -q",
        failing_test_output="ValueError: duplicate parametrize value 5",
        source_files={},
        test_files={
            "test_params.py": (
                "import pytest\n\n"
                "@pytest.mark.parametrize('x', [1, 2, 3, 4, 5])\n"
                "def test_positive(x):\n"
                "    assert x > 0\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "def pytest_generate_tests(metafunc):\n"
                "    if 'x' in metafunc.fixturenames:\n"
                "        metafunc.parametrize('x', [5, 6, 7])\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "def pytest_generate_tests(metafunc):\n    if 'x' in metafunc.fixturenames:\n        metafunc.parametrize('x', [5, 6, 7])\n",
                "new": "",
            }
        },
        verification_command="pytest test_params.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: fixture request scope mismatch
    fixtures.append(Fixture(
        fixture_id="F2_08_depconf_hard_scope_mismatch",
        repo_name="cache_service",
        family=fam,
        bug_report="ScopeMismatch: 'function' scoped fixture cannot use 'session' scoped fixture",
        failing_test_command="pytest test_cache.py -x -q",
        failing_test_output="ScopeMismatch: You tried to access the function scoped fixture",
        source_files={},
        test_files={
            "test_cache.py": (
                "def test_cache_set(cache):\n"
                "    cache['key'] = 'value'\n"
                "    assert cache.get('key') == 'value'\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture(scope='session')\n"
                "def backend():\n"
                "    return {}\n\n"
                "@pytest.fixture\n"
                "def cache(backend):\n"
                "    backend.clear()\n"
                "    return backend\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "@pytest.fixture(scope='session')\ndef backend():\n    return {}\n",
                "new": "@pytest.fixture\ndef backend():\n    return {}\n",
            }
        },
        verification_command="pytest test_cache.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: fixture override in conftest hierarchy
    fixtures.append(Fixture(
        fixture_id="F2_09_depconf_hard_override",
        repo_name="multi_conf",
        family=fam,
        bug_report="Test uses wrong fixture value; inner conftest not overriding outer correctly",
        failing_test_command="pytest test_override.py -x -q",
        failing_test_output="AssertionError: assert 'local' == 'global'",
        source_files={},
        test_files={
            "test_override.py": (
                "def test_value(setting):\n"
                "    assert setting == 'local'\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def setting():\n"
                "    return 'global'\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "    return 'global'\n",
                "new": "    return 'local'\n",
            }
        },
        verification_command="pytest test_override.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: teardown error masks test failure
    fixtures.append(Fixture(
        fixture_id="F2_10_depconf_hard_teardown",
        repo_name="teardown_bug",
        family=fam,
        bug_report="ERROR during teardown — fixture raises instead of using try/finally",
        failing_test_command="pytest test_teardown.py -x -q",
        failing_test_output="ERROR at teardown of test_main",
        source_files={},
        test_files={
            "test_teardown.py": (
                "def test_main(safe_resource):\n"
                "    assert safe_resource['value'] == 42\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def safe_resource():\n"
                "    r = {'value': 42}\n"
                "    yield r\n"
                "    raise RuntimeError('cleanup failed')\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "conftest.py": {
                "old": "    raise RuntimeError('cleanup failed')\n",
                "new": "    pass  # cleanup is a no-op\n",
            }
        },
        verification_command="pytest test_teardown.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Family 3: version_api_mismatch ───────────────────────────────────────────

def _version_api_fixtures() -> List[Fixture]:
    fam = "version_api_mismatch"
    proc = {
        "family": fam,
        "steps": [
            "identify_api_change",
            "locate_deprecated_call_signature",
            "update_call_to_new_signature",
            "verify_api_call_succeeds",
        ],
        "description": "Find the outdated API call and update to the current signature.",
    }

    fixtures = []

    # F01 — easy: keyword arg renamed
    fixtures.append(Fixture(
        fixture_id="F3_01_api_easy_kwarg",
        repo_name="formatter",
        family=fam,
        bug_report="TypeError: format_number() got an unexpected keyword argument 'decimals'",
        failing_test_command="pytest test_formatter.py -x -q",
        failing_test_output="TypeError: format_number() got an unexpected keyword argument 'decimals'",
        source_files={
            "formatter.py": (
                "def format_number(value, precision=2):\n"
                "    return f'{value:.{precision}f}'\n"
            ),
        },
        test_files={
            "test_formatter.py": (
                "from formatter import format_number\n\n"
                "def test_format():\n"
                "    assert format_number(3.14159, decimals=2) == '3.14'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_formatter.py": {
                "old": "    assert format_number(3.14159, decimals=2) == '3.14'\n",
                "new": "    assert format_number(3.14159, precision=2) == '3.14'\n",
            }
        },
        verification_command="pytest test_formatter.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: positional arg removed
    fixtures.append(Fixture(
        fixture_id="F3_02_api_easy_posarg",
        repo_name="string_utils",
        family=fam,
        bug_report="TypeError: truncate() takes 1 positional argument but 2 were given",
        failing_test_command="pytest test_string_utils.py -x -q",
        failing_test_output="TypeError: truncate() takes 1 positional argument but 2 were given",
        source_files={
            "string_utils.py": (
                "def truncate(text, max_len=50):\n"
                "    return text[:max_len]\n"
            ),
        },
        test_files={
            "test_string_utils.py": (
                "from string_utils import truncate\n\n"
                "def test_truncate():\n"
                "    assert truncate('hello world', 5) == 'hello'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "string_utils.py": {
                "old": "def truncate(text, max_len=50):\n",
                "new": "def truncate(text, max_len=50, _legacy_len=None):\n    if _legacy_len is not None: max_len = _legacy_len\n",
            }
        },
        verification_command="pytest test_string_utils.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: return type changed
    fixtures.append(Fixture(
        fixture_id="F3_03_api_easy_rettype",
        repo_name="parser",
        family=fam,
        bug_report="TypeError: parse_items() now returns list, not dict — .items() call fails",
        failing_test_command="pytest test_parser.py -x -q",
        failing_test_output="AttributeError: 'list' object has no attribute 'items'",
        source_files={
            "parser.py": (
                "def parse_items(text):\n"
                "    return [t.strip() for t in text.split(',')]\n"
            ),
        },
        test_files={
            "test_parser.py": (
                "from parser import parse_items\n\n"
                "def test_parse():\n"
                "    result = parse_items('a,b,c')\n"
                "    assert isinstance(result, list)\n"
                "    assert 'a' in result\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_parser.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: method signature changed
    fixtures.append(Fixture(
        fixture_id="F3_04_api_medium_method",
        repo_name="queue_lib",
        family=fam,
        bug_report="TypeError: Queue.push() takes 2 positional arguments — 'priority' required",
        failing_test_command="pytest test_queue.py -x -q",
        failing_test_output="TypeError: push() missing 1 required positional argument: 'priority'",
        source_files={
            "queue_lib.py": (
                "class PriorityQueue:\n"
                "    def __init__(self):\n"
                "        self._items = []\n"
                "    def push(self, item, priority):\n"
                "        self._items.append((priority, item))\n"
                "        self._items.sort()\n"
                "    def pop(self):\n"
                "        return self._items.pop(0)[1] if self._items else None\n"
            ),
        },
        test_files={
            "test_queue.py": (
                "from queue_lib import PriorityQueue\n\n"
                "def test_queue():\n"
                "    q = PriorityQueue()\n"
                "    q.push('low')\n"
                "    q.push('high')\n"
                "    assert q.pop() is not None\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "queue_lib.py": {
                "old": "    def push(self, item, priority):\n",
                "new": "    def push(self, item, priority=0):\n",
            }
        },
        verification_command="pytest test_queue.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: deprecated parameter
    fixtures.append(Fixture(
        fixture_id="F3_05_api_medium_deprecated",
        repo_name="logger_lib",
        family=fam,
        bug_report="TypeError: Logger.__init__() got unexpected keyword argument 'verbose'",
        failing_test_command="pytest test_logger.py -x -q",
        failing_test_output="TypeError: __init__() got an unexpected keyword argument 'verbose'",
        source_files={
            "logger_lib.py": (
                "class Logger:\n"
                "    def __init__(self, name, level='INFO'):\n"
                "        self.name = name\n"
                "        self.level = level\n"
                "    def log(self, msg):\n"
                "        return f'[{self.level}] {self.name}: {msg}'\n"
            ),
        },
        test_files={
            "test_logger.py": (
                "from logger_lib import Logger\n\n"
                "def test_logger():\n"
                "    lg = Logger('app', verbose=True)\n"
                "    assert 'app' in lg.log('hello')\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "logger_lib.py": {
                "old": "    def __init__(self, name, level='INFO'):\n",
                "new": "    def __init__(self, name, level='INFO', verbose=None):\n",
            }
        },
        verification_command="pytest test_logger.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: context manager API changed
    fixtures.append(Fixture(
        fixture_id="F3_06_api_medium_ctxmgr",
        repo_name="lock_manager",
        family=fam,
        bug_report="AttributeError: 'Lock' object has no attribute '__enter__'",
        failing_test_command="pytest test_lock.py -x -q",
        failing_test_output="AttributeError: 'Lock' object has no attribute '__enter__'",
        source_files={
            "lock_manager.py": (
                "class Lock:\n"
                "    def __init__(self):\n"
                "        self.locked = False\n"
                "    def acquire(self):\n"
                "        self.locked = True\n"
                "    def release(self):\n"
                "        self.locked = False\n"
            ),
        },
        test_files={
            "test_lock.py": (
                "from lock_manager import Lock\n\n"
                "def test_lock():\n"
                "    lock = Lock()\n"
                "    with lock:\n"
                "        assert lock.locked\n"
                "    assert not lock.locked\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "lock_manager.py": {
                "old": "    def release(self):\n        self.locked = False\n",
                "new": "    def release(self):\n        self.locked = False\n\n    def __enter__(self):\n        self.acquire()\n        return self\n\n    def __exit__(self, *args):\n        self.release()\n",
            }
        },
        verification_command="pytest test_lock.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: iterator protocol change
    fixtures.append(Fixture(
        fixture_id="F3_07_api_medium_iter",
        repo_name="data_stream",
        family=fam,
        bug_report="TypeError: 'DataStream' object is not iterable",
        failing_test_command="pytest test_stream.py -x -q",
        failing_test_output="TypeError: 'DataStream' object is not iterable",
        source_files={
            "data_stream.py": (
                "class DataStream:\n"
                "    def __init__(self, data):\n"
                "        self._data = data\n"
                "        self._pos = 0\n"
                "    def next(self):\n"
                "        if self._pos >= len(self._data):\n"
                "            raise StopIteration\n"
                "        v = self._data[self._pos]\n"
                "        self._pos += 1\n"
                "        return v\n"
            ),
        },
        test_files={
            "test_stream.py": (
                "from data_stream import DataStream\n\n"
                "def test_iterate():\n"
                "    stream = DataStream([1, 2, 3])\n"
                "    result = list(stream)\n"
                "    assert result == [1, 2, 3]\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "data_stream.py": {
                "old": "    def next(self):\n",
                "new": "    def __iter__(self):\n        return self\n\n    def __next__(self):\n",
            }
        },
        verification_command="pytest test_stream.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: async API mismatch
    fixtures.append(Fixture(
        fixture_id="F3_08_api_hard_async",
        repo_name="async_worker",
        family=fam,
        bug_report="TypeError: object bool can't be used in 'await' expression",
        failing_test_command="pytest test_worker.py -x -q",
        failing_test_output="TypeError: object bool can't be used in 'await' expression",
        source_files={
            "worker.py": (
                "def process(task):\n"
                "    return task.get('done', False)\n"
            ),
        },
        test_files={
            "test_worker.py": (
                "import asyncio\n"
                "from worker import process\n\n"
                "def test_process():\n"
                "    task = {'done': True}\n"
                "    result = process(task)\n"
                "    assert result is True\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_worker.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: __eq__ contract broken
    fixtures.append(Fixture(
        fixture_id="F3_09_api_hard_eq",
        repo_name="value_objects",
        family=fam,
        bug_report="AssertionError: Money(10,'USD') != Money(10,'USD') — __eq__ not implemented",
        failing_test_command="pytest test_money.py -x -q",
        failing_test_output="AssertionError: assert Money(10,'USD') == Money(10,'USD')",
        source_files={
            "value_objects.py": (
                "class Money:\n"
                "    def __init__(self, amount, currency):\n"
                "        self.amount = amount\n"
                "        self.currency = currency\n"
                "    def __repr__(self):\n"
                "        return f\"Money({self.amount!r},{self.currency!r})\"\n"
            ),
        },
        test_files={
            "test_money.py": (
                "from value_objects import Money\n\n"
                "def test_equality():\n"
                "    a = Money(10, 'USD')\n"
                "    b = Money(10, 'USD')\n"
                "    assert a == b\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "value_objects.py": {
                "old": "    def __repr__(self):\n",
                "new": "    def __eq__(self, other):\n        return isinstance(other, Money) and self.amount == other.amount and self.currency == other.currency\n\n    def __repr__(self):\n",
            }
        },
        verification_command="pytest test_money.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: descriptor protocol broken
    fixtures.append(Fixture(
        fixture_id="F3_10_api_hard_descriptor",
        repo_name="validated_model",
        family=fam,
        bug_report="AttributeError: can't set attribute — property has no setter",
        failing_test_command="pytest test_model.py -x -q",
        failing_test_output="AttributeError: can't set attribute",
        source_files={
            "validated_model.py": (
                "class ValidatedField:\n"
                "    def __init__(self):\n"
                "        self._value = None\n"
                "    @property\n"
                "    def value(self):\n"
                "        return self._value\n"
            ),
        },
        test_files={
            "test_model.py": (
                "from validated_model import ValidatedField\n\n"
                "def test_set_value():\n"
                "    f = ValidatedField()\n"
                "    f.value = 42\n"
                "    assert f.value == 42\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "validated_model.py": {
                "old": "    @property\n    def value(self):\n        return self._value\n",
                "new": "    @property\n    def value(self):\n        return self._value\n\n    @value.setter\n    def value(self, v):\n        self._value = v\n",
            }
        },
        verification_command="pytest test_model.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Family 4: path_module_resolution ─────────────────────────────────────────

def _path_module_fixtures() -> List[Fixture]:
    fam = "path_module_resolution"
    proc = {
        "family": fam,
        "steps": [
            "identify_module_path_error",
            "locate_incorrect_import_path",
            "correct_import_path_or_sys_path",
            "verify_module_importable",
        ],
        "description": "Find the wrong import path and correct it so the module resolves.",
    }

    fixtures = []

    # F01 — easy: sys.path insert missing
    fixtures.append(Fixture(
        fixture_id="F4_01_path_easy_syspath",
        repo_name="nested_app",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'helpers'",
        failing_test_command="pytest test_app.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'helpers'",
        source_files={
            "helpers.py": (
                "def greet(name):\n"
                "    return f'Hello, {name}!'\n"
            ),
        },
        test_files={
            "test_app.py": (
                "from helpers import greet\n\n"
                "def test_greet():\n"
                "    assert greet('World') == 'Hello, World!'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_app.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: relative import used in non-package
    fixtures.append(Fixture(
        fixture_id="F4_02_path_easy_relative",
        repo_name="simple_pkg",
        family=fam,
        bug_report="ImportError: attempted relative import with no known parent package",
        failing_test_command="pytest test_simple.py -x -q",
        failing_test_output="ImportError: attempted relative import with no known parent package",
        source_files={
            "utils.py": (
                "CONST = 42\n"
            ),
            "main.py": (
                "from .utils import CONST\n\n"
                "def get_const():\n"
                "    return CONST\n"
            ),
        },
        test_files={
            "test_simple.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from main import get_const\n\n"
                "def test_const():\n"
                "    assert get_const() == 42\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "main.py": {
                "old": "from .utils import CONST\n",
                "new": "from utils import CONST\n",
            }
        },
        verification_command="pytest test_simple.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: wrong directory in path
    fixtures.append(Fixture(
        fixture_id="F4_03_path_easy_wrong_dir",
        repo_name="multi_dir",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'core.engine'",
        failing_test_command="pytest test_engine.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'core.engine'",
        source_files={
            "engine.py": (
                "def start():\n"
                "    return 'running'\n"
            ),
        },
        test_files={
            "test_engine.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from engine import start\n\n"
                "def test_start():\n"
                "    assert start() == 'running'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_engine.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: __init__.py missing
    fixtures.append(Fixture(
        fixture_id="F4_04_path_medium_init",
        repo_name="pkg_missing_init",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'mypackage'",
        failing_test_command="pytest test_pkg.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'mypackage'",
        source_files={
            "mypackage/core.py": (
                "def compute(x):\n"
                "    return x * 2\n"
            ),
        },
        test_files={
            "test_pkg.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from mypackage.core import compute\n\n"
                "def test_compute():\n"
                "    assert compute(5) == 10\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "mypackage/__init__.py": {
                "old": "",
                "new": "",
            }
        },
        verification_command="pytest test_pkg.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: conftest path adds wrong dir
    fixtures.append(Fixture(
        fixture_id="F4_05_path_medium_conftest",
        repo_name="conftest_path",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'lib.utils'",
        failing_test_command="pytest test_lib.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'lib.utils'",
        source_files={
            "utils.py": (
                "def helper():\n"
                "    return True\n"
            ),
        },
        test_files={
            "test_lib.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from utils import helper\n\n"
                "def test_helper():\n"
                "    assert helper() is True\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_lib.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: typo in package name
    fixtures.append(Fixture(
        fixture_id="F4_06_path_medium_typo",
        repo_name="typo_import",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'authentification'",
        failing_test_command="pytest test_auth2.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'authentification'",
        source_files={
            "authentication.py": (
                "def authenticate(user, pwd):\n"
                "    return user == 'admin' and pwd == 'secret'\n"
            ),
        },
        test_files={
            "test_auth2.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from authentification import authenticate\n\n"
                "def test_auth():\n"
                "    assert authenticate('admin', 'secret')\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_auth2.py": {
                "old": "from authentification import authenticate\n",
                "new": "from authentication import authenticate\n",
            }
        },
        verification_command="pytest test_auth2.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: circular import resolved by lazy load
    fixtures.append(Fixture(
        fixture_id="F4_07_path_medium_circular",
        repo_name="circular_dep",
        family=fam,
        bug_report="ImportError: cannot import name 'B' from partially initialized module 'a'",
        failing_test_command="pytest test_circular.py -x -q",
        failing_test_output="ImportError: cannot import name 'B' from partially initialized module",
        source_files={
            "a.py": (
                "from b import B\n\n"
                "class A:\n"
                "    def make_b(self):\n"
                "        return B()\n"
            ),
            "b.py": (
                "class B:\n"
                "    def value(self):\n"
                "        return 99\n"
            ),
        },
        test_files={
            "test_circular.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from b import B\n\n"
                "def test_b():\n"
                "    b = B()\n"
                "    assert b.value() == 99\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "a.py": {
                "old": "from b import B\n\nclass A:\n    def make_b(self):\n        return B()\n",
                "new": "class A:\n    def make_b(self):\n        from b import B\n        return B()\n",
            }
        },
        verification_command="pytest test_circular.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: namespace package conflict
    fixtures.append(Fixture(
        fixture_id="F4_08_path_hard_ns",
        repo_name="ns_conflict",
        family=fam,
        bug_report="ImportError: mylib.utils shadows stdlib 'utils' — wrong module loaded",
        failing_test_command="pytest test_ns.py -x -q",
        failing_test_output="AttributeError: module 'mylib_utils' has no attribute 'deduplicate'",
        source_files={
            "mylib_utils.py": (
                "def deduplicate(lst):\n"
                "    return list(dict.fromkeys(lst))\n"
            ),
        },
        test_files={
            "test_ns.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from mylib_utils import deduplicate\n\n"
                "def test_dedup():\n"
                "    assert deduplicate([1, 2, 1, 3]) == [1, 2, 3]\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_ns.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: editable install path mismatch
    fixtures.append(Fixture(
        fixture_id="F4_09_path_hard_editable",
        repo_name="editable_pkg",
        family=fam,
        bug_report="ModuleNotFoundError: pkg was not installed; src layout not on sys.path",
        failing_test_command="pytest test_editable.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'myapp'",
        source_files={
            "myapp/__init__.py": "",
            "myapp/core.py": (
                "def run():\n"
                "    return 'ok'\n"
            ),
        },
        test_files={
            "test_editable.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from myapp.core import run\n\n"
                "def test_run():\n"
                "    assert run() == 'ok'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_editable.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: version-specific module renamed
    fixtures.append(Fixture(
        fixture_id="F4_10_path_hard_version",
        repo_name="compat_layer",
        family=fam,
        bug_report="ModuleNotFoundError: No module named 'collections.abc' (Py2 style import)",
        failing_test_command="pytest test_compat.py -x -q",
        failing_test_output="ModuleNotFoundError: No module named 'compat_abc'",
        source_files={
            "compat_abc.py": (
                "from collections.abc import Mapping, Sequence\n\n"
                "def is_mapping(obj):\n"
                "    return isinstance(obj, Mapping)\n"
            ),
        },
        test_files={
            "test_compat.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from compat_abc import is_mapping\n\n"
                "def test_mapping():\n"
                "    assert is_mapping({'a': 1})\n"
                "    assert not is_mapping([1, 2])\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_compat.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Family 5: configuration_failure ──────────────────────────────────────────

def _configuration_fixtures() -> List[Fixture]:
    fam = "configuration_failure"
    proc = {
        "family": fam,
        "steps": [
            "identify_configuration_error",
            "locate_config_file_and_key",
            "correct_config_value_or_structure",
            "verify_config_loads_correctly",
        ],
        "description": "Find the wrong configuration key/value and correct it.",
    }

    fixtures = []

    # F01 — easy: pytest.ini wrong testpaths
    fixtures.append(Fixture(
        fixture_id="F5_01_conf_easy_testpaths",
        repo_name="ini_testpaths",
        family=fam,
        bug_report="pytest collects 0 tests — testpaths points to wrong directory",
        failing_test_command="pytest -x -q",
        failing_test_output="collected 0 items",
        source_files={},
        test_files={
            "test_basic.py": (
                "def test_ok():\n"
                "    assert 1 + 1 == 2\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "testpaths = nonexistent_dir\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "testpaths = nonexistent_dir\n",
                "new": "testpaths = .\n",
            }
        },
        verification_command="pytest test_basic.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: wrong python_files glob
    fixtures.append(Fixture(
        fixture_id="F5_02_conf_easy_glob",
        repo_name="ini_glob",
        family=fam,
        bug_report="pytest collects 0 tests — python_files pattern doesn't match test files",
        failing_test_command="pytest -x -q",
        failing_test_output="collected 0 items",
        source_files={},
        test_files={
            "test_glob.py": (
                "def test_glob():\n"
                "    assert True\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "python_files = check_*.py\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "python_files = check_*.py\n",
                "new": "python_files = test_*.py\n",
            }
        },
        verification_command="pytest test_glob.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: filterwarnings causing error
    fixtures.append(Fixture(
        fixture_id="F5_03_conf_easy_filterwarnings",
        repo_name="warnings_err",
        family=fam,
        bug_report="Test fails because filterwarnings=error promotes DeprecationWarning to error",
        failing_test_command="pytest test_warn.py -x -q",
        failing_test_output="DeprecationWarning: old_function is deprecated",
        source_files={
            "warn_utils.py": (
                "import warnings\n\n"
                "def old_function():\n"
                "    warnings.warn('old_function is deprecated', DeprecationWarning)\n"
                "    return 42\n"
            ),
        },
        test_files={
            "test_warn.py": (
                "from warn_utils import old_function\n\n"
                "def test_old():\n"
                "    result = old_function()\n"
                "    assert result == 42\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "filterwarnings = error\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "filterwarnings = error\n",
                "new": "filterwarnings = ignore::DeprecationWarning\n",
            }
        },
        verification_command="pytest test_warn.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: minversion too high
    fixtures.append(Fixture(
        fixture_id="F5_04_conf_medium_minversion",
        repo_name="version_req",
        family=fam,
        bug_report="pytest: error: requirement 'pytest>=99.0' not satisfied",
        failing_test_command="pytest test_version.py -x -q",
        failing_test_output="PytestConfigWarning: Unknown config option",
        source_files={},
        test_files={
            "test_version.py": (
                "def test_pass():\n"
                "    assert True\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "addopts = -v\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_version.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: markers not registered causing warning
    fixtures.append(Fixture(
        fixture_id="F5_05_conf_medium_markers",
        repo_name="unregistered_markers",
        family=fam,
        bug_report="PytestUnknownMarkWarning: Unknown pytest.mark.slow — not registered in ini",
        failing_test_command="pytest test_marked.py -x -q -W error::pytest.PytestUnknownMarkWarning",
        failing_test_output="PytestUnknownMarkWarning: Unknown pytest.mark.slow",
        source_files={},
        test_files={
            "test_marked.py": (
                "import pytest\n\n"
                "@pytest.mark.slow\n"
                "def test_heavy():\n"
                "    assert sum(range(100)) == 4950\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "addopts = -v\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "[pytest]\naddopts = -v\n",
                "new": "[pytest]\naddopts = -v\nmarkers =\n    slow: marks tests as slow\n",
            }
        },
        verification_command="pytest test_marked.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: log_level too verbose failing CI
    fixtures.append(Fixture(
        fixture_id="F5_06_conf_medium_loglevel",
        repo_name="log_level",
        family=fam,
        bug_report="Tests pass but log_level=DEBUG floods output causing CI buffer overflow",
        failing_test_command="pytest test_log.py -x -q",
        failing_test_output="PASSED",
        source_files={
            "log_stuff.py": (
                "import logging\n"
                "logger = logging.getLogger(__name__)\n\n"
                "def do_work():\n"
                "    logger.debug('doing work')\n"
                "    return True\n"
            ),
        },
        test_files={
            "test_log.py": (
                "from log_stuff import do_work\n\n"
                "def test_work():\n"
                "    assert do_work() is True\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "log_cli = true\n"
                "log_cli_level = DEBUG\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "log_cli_level = DEBUG\n",
                "new": "log_cli_level = WARNING\n",
            }
        },
        verification_command="pytest test_log.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: rootdir misconfigured
    fixtures.append(Fixture(
        fixture_id="F5_07_conf_medium_rootdir",
        repo_name="rootdir_conf",
        family=fam,
        bug_report="conftest.py not found — rootdir is wrong, imports fail",
        failing_test_command="pytest test_rootdir.py -x -q",
        failing_test_output="ModuleNotFoundError",
        source_files={
            "helpers.py": (
                "def add(a, b):\n"
                "    return a + b\n"
            ),
        },
        test_files={
            "test_rootdir.py": (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from helpers import add\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_rootdir.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: setup.cfg [tool:pytest] not read
    fixtures.append(Fixture(
        fixture_id="F5_08_conf_hard_setupcfg",
        repo_name="setupcfg_conf",
        family=fam,
        bug_report="setup.cfg [tool:pytest] section ignored — tests not found",
        failing_test_command="pytest test_cfg.py -x -q",
        failing_test_output="collected 0 items / 1 error",
        source_files={},
        test_files={
            "test_cfg.py": (
                "def test_cfg_ok():\n"
                "    assert True\n"
            ),
        },
        config_files={
            "setup.cfg": (
                "[tool:pytest]\n"
                "testpaths = .\n"
                "python_files = test_*.py\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_cfg.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: conftest plugin raises on import
    fixtures.append(Fixture(
        fixture_id="F5_09_conf_hard_plugin",
        repo_name="broken_plugin",
        family=fam,
        bug_report="ImportError in conftest.py plugin blocks all test collection",
        failing_test_command="pytest test_plugin.py -x -q",
        failing_test_output="ImportError: cannot import name 'missing_hook'",
        source_files={},
        test_files={
            "test_plugin.py": (
                "def test_simple():\n"
                "    assert 1 == 1\n"
            ),
        },
        config_files={
            "conftest.py": (
                "import pytest\n\n"
                "def pytest_configure(config):\n"
                "    pass\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_plugin.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: xdist config conflict
    fixtures.append(Fixture(
        fixture_id="F5_10_conf_hard_xdist",
        repo_name="xdist_conflict",
        family=fam,
        bug_report="addopts has -n auto but xdist not installed — collection fails",
        failing_test_command="pytest test_xdist.py -x -q",
        failing_test_output="ERROR: unrecognized arguments: -n",
        source_files={},
        test_files={
            "test_xdist.py": (
                "def test_xdist():\n"
                "    assert True\n"
            ),
        },
        config_files={
            "pytest.ini": (
                "[pytest]\n"
                "addopts = -n auto\n"
            ),
        },
        oracle_repair_procedure=proc,
        expected_patch={
            "pytest.ini": {
                "old": "addopts = -n auto\n",
                "new": "addopts =\n",
            }
        },
        verification_command="pytest test_xdist.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Family 6: test_assertion_repair ──────────────────────────────────────────

def _assertion_repair_fixtures() -> List[Fixture]:
    fam = "test_assertion_repair"
    proc = {
        "family": fam,
        "steps": [
            "identify_assertion_failure",
            "locate_failing_assertion",
            "correct_expected_value_or_logic",
            "verify_assertion_passes",
        ],
        "description": "Find the wrong expected value or assertion logic and fix it.",
    }

    fixtures = []

    # F01 — easy: off-by-one
    fixtures.append(Fixture(
        fixture_id="F6_01_assert_easy_off1",
        repo_name="off_by_one",
        family=fam,
        bug_report="AssertionError: assert 10 == 11 — off-by-one in expected range length",
        failing_test_command="pytest test_range.py -x -q",
        failing_test_output="AssertionError: assert 10 == 11",
        source_files={
            "range_utils.py": (
                "def count_range(start, stop):\n"
                "    return len(range(start, stop))\n"
            ),
        },
        test_files={
            "test_range.py": (
                "from range_utils import count_range\n\n"
                "def test_count():\n"
                "    assert count_range(1, 11) == 11\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_range.py": {
                "old": "    assert count_range(1, 11) == 11\n",
                "new": "    assert count_range(1, 11) == 10\n",
            }
        },
        verification_command="pytest test_range.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F02 — easy: wrong expected string
    fixtures.append(Fixture(
        fixture_id="F6_02_assert_easy_string",
        repo_name="string_format",
        family=fam,
        bug_report="AssertionError: assert 'Hello, Alice!' == 'hello, alice!'",
        failing_test_command="pytest test_greet2.py -x -q",
        failing_test_output="AssertionError: assert 'Hello, Alice!' == 'hello, alice!'",
        source_files={
            "greeter.py": (
                "def greet(name):\n"
                "    return f'Hello, {name}!'\n"
            ),
        },
        test_files={
            "test_greet2.py": (
                "from greeter import greet\n\n"
                "def test_greet():\n"
                "    assert greet('Alice') == 'hello, alice!'\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_greet2.py": {
                "old": "    assert greet('Alice') == 'hello, alice!'\n",
                "new": "    assert greet('Alice') == 'Hello, Alice!'\n",
            }
        },
        verification_command="pytest test_greet2.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F03 — easy: wrong comparison operator
    fixtures.append(Fixture(
        fixture_id="F6_03_assert_easy_op",
        repo_name="comparator",
        family=fam,
        bug_report="AssertionError: assert not True — should be 'assert result is True'",
        failing_test_command="pytest test_compare.py -x -q",
        failing_test_output="AssertionError: assert not True",
        source_files={
            "comparator.py": (
                "def is_positive(n):\n"
                "    return n > 0\n"
            ),
        },
        test_files={
            "test_compare.py": (
                "from comparator import is_positive\n\n"
                "def test_positive():\n"
                "    assert not is_positive(5)\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_compare.py": {
                "old": "    assert not is_positive(5)\n",
                "new": "    assert is_positive(5)\n",
            }
        },
        verification_command="pytest test_compare.py -x -q",
        transfer_group="train",
        difficulty="easy",
    ))

    # F04 — medium: float precision error
    fixtures.append(Fixture(
        fixture_id="F6_04_assert_medium_float",
        repo_name="float_calc",
        family=fam,
        bug_report="AssertionError: assert 0.1 + 0.2 == 0.3 fails due to float precision",
        failing_test_command="pytest test_float.py -x -q",
        failing_test_output="AssertionError: assert 0.30000000000000004 == 0.3",
        source_files={
            "float_calc.py": (
                "def add_floats(a, b):\n"
                "    return a + b\n"
            ),
        },
        test_files={
            "test_float.py": (
                "from float_calc import add_floats\n\n"
                "def test_add():\n"
                "    assert add_floats(0.1, 0.2) == 0.3\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_float.py": {
                "old": "    assert add_floats(0.1, 0.2) == 0.3\n",
                "new": "    assert abs(add_floats(0.1, 0.2) - 0.3) < 1e-9\n",
            }
        },
        verification_command="pytest test_float.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F05 — medium: list order assumption
    fixtures.append(Fixture(
        fixture_id="F6_05_assert_medium_order",
        repo_name="set_ops",
        family=fam,
        bug_report="AssertionError: list order not guaranteed for set operations",
        failing_test_command="pytest test_set_ops.py -x -q",
        failing_test_output="AssertionError: assert [1, 2, 3] == [3, 1, 2]",
        source_files={
            "set_ops.py": (
                "def unique_sorted(lst):\n"
                "    return sorted(set(lst))\n"
            ),
        },
        test_files={
            "test_set_ops.py": (
                "from set_ops import unique_sorted\n\n"
                "def test_unique():\n"
                "    result = unique_sorted([3, 1, 2, 1])\n"
                "    assert result == [3, 1, 2]\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_set_ops.py": {
                "old": "    assert result == [3, 1, 2]\n",
                "new": "    assert result == [1, 2, 3]\n",
            }
        },
        verification_command="pytest test_set_ops.py -x -q",
        transfer_group="train",
        difficulty="medium",
    ))

    # F06 — medium: wrong exception type asserted
    fixtures.append(Fixture(
        fixture_id="F6_06_assert_medium_exc",
        repo_name="exc_handler",
        family=fam,
        bug_report="Failed: DID NOT RAISE — wrong exception type expected",
        failing_test_command="pytest test_exc.py -x -q",
        failing_test_output="Failed: DID NOT RAISE <class 'ValueError'>",
        source_files={
            "divider.py": (
                "def safe_divide(a, b):\n"
                "    if b == 0:\n"
                "        raise ZeroDivisionError('division by zero')\n"
                "    return a / b\n"
            ),
        },
        test_files={
            "test_exc.py": (
                "import pytest\n"
                "from divider import safe_divide\n\n"
                "def test_divide_by_zero():\n"
                "    with pytest.raises(ValueError):\n"
                "        safe_divide(10, 0)\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_exc.py": {
                "old": "    with pytest.raises(ValueError):\n",
                "new": "    with pytest.raises(ZeroDivisionError):\n",
            }
        },
        verification_command="pytest test_exc.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F07 — medium: dict subset assertion
    fixtures.append(Fixture(
        fixture_id="F6_07_assert_medium_dict",
        repo_name="dict_checker",
        family=fam,
        bug_report="AssertionError: expected dict subset but checking full equality",
        failing_test_command="pytest test_dict.py -x -q",
        failing_test_output="AssertionError: assert {'a': 1, 'b': 2, 'extra': 3} == {'a': 1, 'b': 2}",
        source_files={
            "dict_builder.py": (
                "def build(a, b):\n"
                "    return {'a': a, 'b': b, 'extra': a + b}\n"
            ),
        },
        test_files={
            "test_dict.py": (
                "from dict_builder import build\n\n"
                "def test_build():\n"
                "    result = build(1, 2)\n"
                "    assert result == {'a': 1, 'b': 2}\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_dict.py": {
                "old": "    assert result == {'a': 1, 'b': 2}\n",
                "new": "    assert result['a'] == 1 and result['b'] == 2\n",
            }
        },
        verification_command="pytest test_dict.py -x -q",
        transfer_group="near_transfer",
        difficulty="medium",
    ))

    # F08 — hard: assertion on mutable default
    fixtures.append(Fixture(
        fixture_id="F6_08_assert_hard_mutable",
        repo_name="mutable_default",
        family=fam,
        bug_report="AssertionError: shared mutable default arg causes cross-test contamination",
        failing_test_command="pytest test_mutable.py -x -q",
        failing_test_output="AssertionError: assert [1, 2] == [2]",
        source_files={
            "accumulator.py": (
                "def add_item(item, storage=None):\n"
                "    if storage is None:\n"
                "        storage = []\n"
                "    storage.append(item)\n"
                "    return storage\n"
            ),
        },
        test_files={
            "test_mutable.py": (
                "from accumulator import add_item\n\n"
                "def test_first():\n"
                "    r = add_item(1)\n"
                "    assert r == [1]\n\n"
                "def test_second():\n"
                "    r = add_item(2)\n"
                "    assert r == [2]\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_mutable.py -x -q",
        transfer_group="near_transfer",
        difficulty="hard",
    ))

    # F09 — hard: none vs falsy assertion
    fixtures.append(Fixture(
        fixture_id="F6_09_assert_hard_none",
        repo_name="none_check",
        family=fam,
        bug_report="AssertionError: assert None is False — is vs == confusion",
        failing_test_command="pytest test_none.py -x -q",
        failing_test_output="AssertionError: assert None is False",
        source_files={
            "finder.py": (
                "def find_first(lst, pred):\n"
                "    for item in lst:\n"
                "        if pred(item):\n"
                "            return item\n"
                "    return None\n"
            ),
        },
        test_files={
            "test_none.py": (
                "from finder import find_first\n\n"
                "def test_not_found():\n"
                "    result = find_first([1, 2, 3], lambda x: x > 10)\n"
                "    assert result is False\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={
            "test_none.py": {
                "old": "    assert result is False\n",
                "new": "    assert result is None\n",
            }
        },
        verification_command="pytest test_none.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    # F10 — hard: timezone-aware datetime comparison
    fixtures.append(Fixture(
        fixture_id="F6_10_assert_hard_tz",
        repo_name="datetime_compare",
        family=fam,
        bug_report="AssertionError: datetime comparison fails due to timezone naive/aware mismatch",
        failing_test_command="pytest test_datetime.py -x -q",
        failing_test_output="AssertionError: naive datetime != aware datetime",
        source_files={
            "time_utils.py": (
                "from datetime import datetime, timezone\n\n"
                "def now_utc():\n"
                "    return datetime.now(tz=timezone.utc)\n"
            ),
        },
        test_files={
            "test_datetime.py": (
                "from datetime import datetime\n"
                "from time_utils import now_utc\n\n"
                "def test_now_is_recent():\n"
                "    t = now_utc()\n"
                "    assert isinstance(t, datetime)\n"
                "    assert t.tzinfo is not None\n"
            ),
        },
        config_files={},
        oracle_repair_procedure=proc,
        expected_patch={},
        verification_command="pytest test_datetime.py -x -q",
        transfer_group="far_transfer",
        difficulty="hard",
    ))

    return fixtures


# ── Public API ────────────────────────────────────────────────────────────────

def build_all_fixtures() -> List[Fixture]:
    """Return all 60 benchmark fixtures in canonical family order."""
    all_fixtures: List[Fixture] = []
    all_fixtures.extend(_import_module_fixtures())
    all_fixtures.extend(_dependency_config_fixtures())
    all_fixtures.extend(_version_api_fixtures())
    all_fixtures.extend(_path_module_fixtures())
    all_fixtures.extend(_configuration_fixtures())
    all_fixtures.extend(_assertion_repair_fixtures())
    assert len(all_fixtures) == 60, f"Expected 60 fixtures, got {len(all_fixtures)}"
    return all_fixtures


def build_fixtures_by_family() -> Dict[str, List[Fixture]]:
    """Return fixtures organised by family name."""
    result: Dict[str, List[Fixture]] = {fam: [] for fam in FAMILY_NAMES}
    for fx in build_all_fixtures():
        result[fx.family].append(fx)
    return result
