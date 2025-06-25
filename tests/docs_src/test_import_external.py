"""Tests for documentation modules that require external services.

These tests are marked as 'external' and require live network connections
to external services. They are excluded from the default test run to avoid
failures due to network issues, SSL certificate expiration, or service
unavailability.

To run these tests explicitly:
    pytest -m external tests/docs_src/test_import_external.py
"""

import importlib
from pathlib import Path

import pytest

from ..helpers import add_to_sys_path

root_path = (Path(__file__).parents[2] / "docs").resolve()

# Modules that require external services
EXTERNAL_MODULES = [
    "docs_src.user_guide.external_rest_apis.main",
    "docs_src.user_guide.external_rest_apis.security",
    "docs_src.user_guide.runtimes.ag2.mesop.main",
    "docs_src.user_guide.runtimes.ag2.mesop.using_non_openai_models",
]


@pytest.mark.external
@pytest.mark.parametrize("module", EXTERNAL_MODULES)
def test_external_submodules(module: str) -> None:
    """Test import of modules that depend on external services.

    These tests require:
    - Live network connection
    - External service availability (weather.tools.fastagency.ai)
    - Valid SSL certificates
    """
    with add_to_sys_path(root_path):
        importlib.import_module(module)  # nosemgrep
