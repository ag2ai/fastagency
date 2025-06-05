import subprocess  # nosec B404
from logging import getLogger
from typing import Annotated, Optional

import typer

from .logging import setup_logging

docker_app = typer.Typer(rich_markup_mode="rich")


setup_logging()
logger = getLogger(__name__)


@docker_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Build a Docker image for the FastAgency app",
)
def build(
    build_context: Annotated[
        str,
        typer.Argument(
            ...,
            help="Docker build context",
        ),
    ] = ".",
    *,
    file: Annotated[
        str,
        typer.Option(
            "--file",
            "-f",
            help="Name of the Dockerfile",
        ),
    ] = "docker/Dockerfile",
    tag: Annotated[
        str,
        typer.Option(
            "--tag",
            "-t",
            help='Name and optionally a tag (format: "name:tag")',
        ),
    ] = "deploy_fastagency",
    progress: Annotated[
        str,
        typer.Option(
            "--progress",
            help="Set type of progress output (auto, plain, tty, rawjson).",
        ),
    ] = "plain",
    ctx: typer.Context,
) -> None:
    command = [
        "docker",
        "build",
        "-t",
        tag,
        "-f",
        file,
        "--progress",
        progress,
        build_context,
    ]
    command += ctx.args
    typer.echo(
        f"Building FastAgency Docker image with the command: {' '.join(command)}"
    )
    try:
        # Run the docker build command
        result = subprocess.run(  # nosec B603
            command, check=True, capture_output=True, text=True
        )
        typer.echo(result.stdout)
        typer.echo(f"Image '{tag}' built successfully!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error: {e.stderr}", err=True)
        raise typer.Exit(code=1) from e


@docker_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run a Docker container for the FastAgency app",
)
def run(
    image: Annotated[
        str,
        typer.Argument(
            ...,
            help="The Docker image to run",
        ),
    ] = "deploy_fastagency",
    *,
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Assign a name to the container",
        ),
    ] = "deploy_fastagency",
    env: Annotated[
        Optional[list[str]],
        typer.Option(
            "--env",
            "-e",
            help="Set environment variables",
            show_default=False,
        ),
    ] = None,
    publish: Annotated[
        Optional[list[str]],
        typer.Option(
            "--publish",
            "-p",
            help="Publish a container's port(s) to the host",
            show_default=False,
        ),
    ] = None,
    remove: Annotated[
        bool,
        typer.Option(
            "--rm",
            help="Automatically remove the container and its associated anonymous volumes when it exits",
            is_flag=True,
        ),
    ] = False,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            "-d",
            help="Run container in background and print container ID",
            is_flag=True,
        ),
    ] = True,
    network: Annotated[
        Optional[str],
        typer.Option(
            "--network",
            help="Connect a container to a network",
            show_default=False,
        ),
    ] = None,
    ctx: typer.Context,
) -> None:
    # Construct the docker run command using the provided options
    command = ["docker", "run", "--name", name]

    if env:
        for env_var in env:
            command.extend(["--env", env_var])

    if publish:
        for port in publish:
            command.extend(["--publish", port])
        if "8888:8888" not in publish:
            command.extend(["--publish", "8888:8888"])
    else:
        command.extend(["--publish", "8888:8888"])

    if remove:
        command.append("--rm")

    if detach:
        command.append("--detach")

    if network:
        command.extend(["--network", network])

    command += ctx.args
    command.append(image)

    try:
        typer.echo(
            f"Running FastAgency Docker image with the command: {' '.join(command)}"
        )
        # Run the docker command
        result = subprocess.run(  # nosec B603
            command, check=True, capture_output=True, text=True
        )
        typer.echo(result.stdout)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error: {e.stderr}", err=True)
        raise typer.Exit(code=1) from e


@docker_app.command(
    context_settings={"allow_extra_args": False, "ignore_unknown_options": False},
    help="Deploy the Docker container for the FastAgency app to Fly.io",
)
def deploy(
    config_file: Annotated[
        str,
        typer.Argument(
            ...,
            help="The Fly.io configuration file",
        ),
    ] = "fly.toml",
    *,
    openai_api_key: Annotated[
        str,
        typer.Option(
            "--openai-api-key",
            help="OpenAI API key",
            envvar="OPENAI_API_KEY",
            show_default=False,
        ),
    ],
    # ctx: typer.Context,
) -> None:
    launch_command = [
        "fly",
        "launch",
        "--config",
        config_file,
        "--copy-config",
        "--yes",
    ]
    # launch_command += ctx.args

    set_secret_command = ["fly", "secrets", "set", "OPENAI_API_KEY=" + openai_api_key]
    try:
        typer.echo(
            f"Deploying FastAgency Docker image to Fly.io with the command: {' '.join(launch_command)}"
        )
        # Run the fly deploy command
        deploy_result = subprocess.run(  # nosec B603
            launch_command, check=True, capture_output=True, text=True
        )
        typer.echo(deploy_result.stdout)

        typer.echo(
            f"Setting OpenAI API key with the command: {' '.join(set_secret_command)}"
        )
        # Run the fly secrets set command
        set_secret_result = subprocess.run(  # nosec B603
            set_secret_command, check=True, capture_output=True, text=True
        )
        typer.echo(set_secret_result.stdout)
        typer.echo("Deployed FastAgency Docker image to Fly.io successfully!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error: {e.stderr}", err=True)
        raise typer.Exit(code=1) from e
