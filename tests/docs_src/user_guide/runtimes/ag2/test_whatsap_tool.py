from unittest.mock import MagicMock

import pytest
import requests
from typer.testing import CliRunner

from fastagency.cli import app

from .....conftest import InputMock
from ....helpers import skip_internal_server_error

runner = CliRunner()

INPUT_MESSAGE = "Send 'Hi!' to 123456789"


@pytest.mark.skip("TODO: Make test not reliant on LLM calls")
@pytest.mark.openai
@skip_internal_server_error
def test_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", InputMock([INPUT_MESSAGE]))

    mock = MagicMock()

    monkeypatch.setattr(requests, "post", mock)

    result = runner.invoke(
        app,
        [
            "run",
            "docs/docs_src/user_guide/runtimes/ag2/whatsapp_tool.py",
            "--single-run",
        ],
    )

    mock.assert_called_once()

    assert mock.call_args.args[0] == "https://api.infobip.com/whatsapp/1/message/text"

    json = mock.call_args.kwargs["json"]

    assert "from" in json
    assert "to" in json
    assert "content" in json

    assert result.exit_code == 0
    assert "Workflow -> User [workflow_completed]" in result.stdout
