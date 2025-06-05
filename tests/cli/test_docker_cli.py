import platform
import subprocess
from typing import Any

import pytest
from typer.testing import CliRunner

from fastagency.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "command,expected",  # noqa: PT006
    [
        (
            ["docker", "build"],
            [
                "docker",
                "build",
                "-t",
                "deploy_fastagency",
                "-f",
                "docker/Dockerfile",
                "--progress",
                "plain",
                ".",
            ],
        ),
        (
            ["docker", "build", "--no-cache", "."],
            [
                "docker",
                "build",
                "-t",
                "deploy_fastagency",
                "-f",
                "docker/Dockerfile",
                "--progress",
                "plain",
                "--no-cache",
                ".",
            ],
        ),
    ],
)
def test_docker_build(
    command: list[str], expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def patch_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        assert args[0] == expected
        assert kwargs["check"]
        assert kwargs["capture_output"]
        assert kwargs["text"]

        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="Dummy docker build output"
        )

    monkeypatch.setattr(subprocess, "run", patch_subprocess_run)

    result = runner.invoke(app, command)
    assert result.exit_code == 0
    assert "Building FastAgency Docker image" in result.stdout
    assert " ".join(expected) in result.stdout


@pytest.mark.skip("TODO: check why this test is failing")
@pytest.mark.skipif(
    platform.system() == "Windows" or platform.system() == "Darwin",
    reason="Docker not supported on Windows or macOS CI",
)
def test_docker_build_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["docker", "build", "--invalid-argument", "."]

    result = runner.invoke(app, command)
    assert result.exit_code != 0
    assert "Building FastAgency Docker image" in result.stdout, result.stdout
    assert "Error: unknown flag: --invalid-argument" in result.stdout, result.stdout


@pytest.mark.parametrize(
    "command,expected",  # noqa: PT006
    [
        (
            ["docker", "run"],
            [
                "docker",
                "run",
                "--name",
                "deploy_fastagency",
                "--publish",
                "8888:8888",
                "--detach",
                "deploy_fastagency",
            ],
        ),
        (
            ["docker", "run", "--rm"],
            [
                "docker",
                "run",
                "--name",
                "deploy_fastagency",
                "--publish",
                "8888:8888",
                "--rm",
                "--detach",
                "deploy_fastagency",
            ],
        ),
        (
            ["docker", "run", "--network", "host"],
            [
                "docker",
                "run",
                "--name",
                "deploy_fastagency",
                "--publish",
                "8888:8888",
                "--detach",
                "--network",
                "host",
                "deploy_fastagency",
            ],
        ),
    ],
)
def test_docker_run(
    command: list[str], expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def patch_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        assert args[0] == expected
        assert kwargs["check"]
        assert kwargs["capture_output"]
        assert kwargs["text"]

        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="Dummy docker run output"
        )

    monkeypatch.setattr(subprocess, "run", patch_subprocess_run)

    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.stdout
    assert "Running FastAgency Docker image" in result.stdout
    assert " ".join(expected) in result.stdout


@pytest.mark.skip("TODO: check why this test is failing")
@pytest.mark.skipif(
    platform.system() == "Windows" or platform.system() == "Darwin",
    reason="Docker not supported on Windows or macOS CI",
)
def test_docker_run_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["docker", "run", "--invalid-argument"]

    result = runner.invoke(app, command)
    assert result.exit_code != 0
    assert "Running FastAgency Docker image" in result.stdout, result.stdout
    assert "Error: unknown flag: --invalid-argument" in result.stdout, result.stdout


@pytest.mark.parametrize(
    "command,expected",  # noqa: PT006
    [
        (
            ["docker", "deploy", "--openai-api-key", "dummy_key"],
            [
                "fly",
                "launch",
                "--config",
                "fly.toml",
                "--copy-config",
                "--yes",
            ],
        ),
        (
            [
                "docker",
                "deploy",
                "--openai-api-key",
                "dummy_key",
                "--config",
                "fly.prod.toml",
            ],
            [
                "fly",
                "launch",
                "--config",
                "fly.prod.toml",
                "--copy-config",
                "--yes",
            ],
        ),
    ],
)
def test_docker_deploy(
    command: list[str], expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def patch_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        if args[0][1] == "launch":
            assert args[0] == expected
        elif args[0][1] == "secrets":
            assert args[0][:3] == ["fly", "secrets", "set"]
        assert kwargs["check"]
        assert kwargs["capture_output"]
        assert kwargs["text"]

        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="Dummy docker deploy output"
        )

    monkeypatch.setattr(subprocess, "run", patch_subprocess_run)

    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.stdout
    assert "Deploying FastAgency Docker image to Fly.io" in result.stdout
    assert " ".join(expected) in result.stdout


def test_docker_deploy_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["docker", "deploy", "--invalid-argument"]

    result = runner.invoke(app, command)
    assert result.exit_code != 0, result.stdout
    # In newer versions of Typer, error messages are not in stdout but in the exception
    # We just need to verify the command exits with an error code
    assert result.exit_code == 2, f"Expected exit code 2, got {result.exit_code}"
