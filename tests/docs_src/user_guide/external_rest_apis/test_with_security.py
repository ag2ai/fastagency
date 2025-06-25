import pytest
from typer.testing import CliRunner

from fastagency.cli import app

from ....conftest import InputMock
from ...helpers import skip_internal_server_error

runner = CliRunner()

INPUT_MESSAGE = "Get me hourly forecast for Zagreb city"


@pytest.mark.openai
@pytest.mark.external
@pytest.mark.slow
@skip_internal_server_error
def test_cli_with_security(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", InputMock([INPUT_MESSAGE]))

    result = runner.invoke(
        app,
        [
            "run",
            "docs/docs_src/user_guide/external_rest_apis/security.py",
            "--single-run",
        ],
    )
    assert result.exit_code == 0
    assert INPUT_MESSAGE in result.stdout
    assert "get_hourly_weather_hourly_get" in result.stdout
