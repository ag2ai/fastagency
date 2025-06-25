import pytest
from typer.testing import CliRunner

from fastagency.cli import app

from .....conftest import InputMock
from ....helpers import skip_internal_server_error

runner = CliRunner()

INPUT_MESSAGE = (
    "Search for information about AG2 (formerly AutoGen) and summarize the results,"
)


@pytest.mark.openai
@pytest.mark.external
@pytest.mark.slow
@skip_internal_server_error
def test_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", InputMock([INPUT_MESSAGE]))

    result = runner.invoke(
        app,
        [
            "run",
            "docs/docs_src/user_guide/runtimes/ag2/websurfer_tool.py",
            "--single-run",
        ],
    )
    assert result.exit_code == 0
    # assert INPUT_MESSAGE in result.stdout
    assert "Workflow (to User) [workflow_completed]" in result.stdout
