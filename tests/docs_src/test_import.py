import importlib
import sys
from pathlib import Path

import pytest

from fastagency.exceptions import FastAgencyCLIPythonVersionError

from ..helpers import add_to_sys_path, list_submodules

root_path = (Path(__file__).parents[2] / "docs").resolve()
module_name = "docs_src"

# Constants for module paths
MESOP_AUTH_MODULES = {
    "docs_src.user_guide.ui.mesop.main_mesop_firebase_auth",
    "docs_src.user_guide.ui.mesop.main_mesop_basic_auth",
}

MESOP_NON_AUTH_MODULES = {
    "docs_src.user_guide.ui.mesop.main_mesop",
}

MESOP_EXCLUDED_MODULES = {
    "docs_src.tutorials.giphy",
    "docs_src.tutorials.whatsapp",
    "docs_src.user_guide.runtimes.ag2.mesop",
    "docs_src.user_guide.dependency_injection",
}

EXTERNAL_MODULES = {
    "docs_src.user_guide.external_rest_apis.main",
    "docs_src.user_guide.external_rest_apis.security",
}

# Mock Environment variables for Mesop Auth testing
MOCK_ENV_VARS: dict[str, str] = {
    "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/credentials.json",
}


def test_list_submodules() -> None:
    # Specify the name of the module you want to inspect

    # Get the list of submodules for the specified module
    submodules = list_submodules(module_name, include_path=root_path)

    assert len(submodules) > 0
    assert "docs_src" in submodules
    assert "docs_src.getting_started" in submodules


@pytest.mark.parametrize("module", list_submodules(module_name, include_path=root_path))
def test_submodules(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip external modules - they are tested separately in test_import_external.py
    if module in EXTERNAL_MODULES:
        pytest.skip(
            f"External module {module} is tested separately with 'external' marker"
        )

    with add_to_sys_path(root_path):
        if sys.version_info >= (3, 10):
            if module in MESOP_AUTH_MODULES or module in MESOP_NON_AUTH_MODULES:
                if module in MESOP_AUTH_MODULES:
                    # Ensure required environment variables are set mocked
                    for key, value in MOCK_ENV_VARS.items():
                        monkeypatch.setenv(key, value)
                importlib.import_module(module)  # nosemgrep
                return

        else:
            # Python < 3.10 handling
            if module in MESOP_AUTH_MODULES or module in MESOP_NON_AUTH_MODULES:
                with pytest.raises(
                    ModuleNotFoundError,
                    match="No module named 'mesop'",
                ):
                    importlib.import_module(module)  # nosemgrep
                return
            elif (
                module.startswith("docs_src.user_guide.ui.mesop")
                or module in MESOP_EXCLUDED_MODULES
            ):
                pass
            elif "mesop" in module or "tutorials.giphy.simple_main" in module:
                with pytest.raises(
                    FastAgencyCLIPythonVersionError,
                    match="Mesop requires Python 3.10 or higher",
                ):
                    importlib.import_module(module)  # nosemgrep
                return

        importlib.import_module(module)  # nosemgrep
