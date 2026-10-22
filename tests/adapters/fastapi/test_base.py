from types import SimpleNamespace
from unittest.mock import patch

from fastagency.adapters.fastapi.base import FastAPIProvider


def test_workflow_discovery_uses_configured_route_path() -> None:
    provider = FastAPIProvider("http://synthetic.local/")
    requested_urls: list[str] = []

    def mock_get(url: str, timeout: int) -> SimpleNamespace:
        requested_urls.append(url)
        if url == "http://synthetic.local/fastagency/discovery":
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"name": "synthetic_workflow", "description": "fixture"}
                ],
            )
        return SimpleNamespace(status_code=404, json=lambda: {"detail": "Not Found"})

    with patch("fastagency.adapters.fastapi.base.requests.get", side_effect=mock_get):
        assert provider.names == ["synthetic_workflow"]

    assert requested_urls == ["http://synthetic.local/fastagency/discovery"]
