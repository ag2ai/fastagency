![alt text](https://github.com/ag2ai/fastagency/blob/main/docs/docs/assets/img/FA-Secondary-LOGO.jpg?raw=true)


# FastAgency


<b>The fastest way to bring multi-agent workflows to production.</b>


---

<p align="center">
  <a href="https://github.com/ag2ai/fastagency/actions/workflows/pipeline.yaml" target="_blank">
    <img src="https://github.com/ag2ai/fastagency/actions/workflows/pipeline.yaml/badge.svg?branch=main" alt="Test Passing"/>
  </a>

  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/fastagency" target="_blank">
      <img src="https://coverage-badge.samuelcolvin.workers.dev/ag2ai/fastagency.svg" alt="Coverage">
  </a>

  <a href="https://www.pepy.tech/projects/fastagency" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/fastagency?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/fastagency" target="_blank">
    <img src="https://img.shields.io/pypi/v/fastagency?label=PyPI" alt="Package version">
  </a>

  <a href="https://pypi.org/project/fastagency" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/fastagency.svg" alt="Supported Python versions">
  </a>

  <br/>

  <a href="https://github.com/ag2ai/fastagency/actions/workflows/codeql.yml" target="_blank">
    <img src="https://github.com/ag2ai/fastagency/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
  </a>

  <a href="https://github.com/ag2ai/fastagency/actions/workflows/dependency-review.yaml" target="_blank">
    <img src="https://github.com/ag2ai/fastagency/actions/workflows/dependency-review.yaml/badge.svg" alt="Dependency Review">
  </a>

  <a href="https://github.com/ag2ai/fastagency/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/ag2ai/fastagency.png" alt="License">
  </a>

  <a href="https://github.com/ag2ai/fastagency/blob/main/CODE_OF_CONDUCT.md" target="_blank">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Code of Conduct">
  </a>

  <a href="https://discord.gg/kJjSGWrknU" target="_blank">
      <img alt="Discord" src="https://img.shields.io/discord/1247409549158121512?logo=discord">
  </a>
</p>

---

## What is FastAgency?

For start, FastAgency is not yet another agentic AI framework. There are many such
frameworks available today, the most popular open-source one is [**AG2** (formerly AutoGen)](https://github.com/ag2ai/ag2). FastAgency provides you with a unified programming interface for deploying agentic workflows written in AG2 agentic framework in both development and productional settings. With only a few lines of code, you can create a web chat application or REST API service interacting with agents of your choice. If you need to scale-up your workloads, FastAgency can help you deploy a fully distributed system using internal message brokers coordinating multiple machines in multiple datacenters with just a few lines of code changed from your local development setup.

**FastAgency** is an open-source framework designed to accelerate the transition from prototype to production for multi-agent AI workflows. For developers who use the AG2 (formerly AutoGen) framework, FastAgency enables you to seamlessly scale Jupyter notebook prototypes into a fully functional, production-ready applications. With multi-framework support, a unified programming interface, and powerful API integration capabilities, FastAgency streamlines the deployment process, saving time and effort while maintaining flexibility and performance.

Whether you're orchestrating complex AI agents or integrating external APIs into workflows, FastAgency provides the tools necessary to quickly transition from concept to production, reducing development cycles and allowing you to focus on optimizing your multi-agent systems.

## Key Features

- [**Unified Programming Interface Across UIs**](user-guide/ui/index.md): FastAgency features a **common programming interface** that enables you to develop your core workflows once and reuse them across various user interfaces without rewriting code. This includes support for both **console-based applications** via `ConsoleUI` and **web-based applications** via `MesopUI`. Whether you need a command-line tool or a fully interactive web app, FastAgency allows you to deploy the same underlying workflows across environments, saving development time and ensuring consistency.

- [**Seamless External API Integration**](user-guide/api/index.md): One of FastAgency's standout features is its ability to easily integrate external APIs into your agent workflows. With just a **few lines of code**, you can import an OpenAPI specification, and in only one more line, you can connect it to your agents. This dramatically simplifies the process of enhancing AI agents with real-time data, external processing, or third-party services. For example, you can easily integrate a weather API to provide dynamic, real-time weather updates to your users, making your application more interactive and useful with minimal effort.

- [**Tester Class for Continuous Integration**](user-guide/testing/index.md): FastAgency also provides a **Tester Class** that enables developers to write and execute tests for their multi-agent workflows. This feature is crucial for maintaining the reliability and robustness of your application, allowing you to automatically verify agent behavior and interactions. The Tester Class is designed to integrate smoothly with **continuous integration (CI)** pipelines, helping you catch bugs early and ensure that your workflows remain functional as they scale into production.

- [**Command-Line Interface (CLI) for Orchestration**](user-guide/cli/index.md): FastAgency includes a powerful **command-line interface (CLI)** for orchestrating and managing multi-agent applications directly from the terminal. The CLI allows developers to quickly run workflows, pass parameters, and monitor agent interactions without needing a full GUI. This is especially useful for automating deployments and integrating workflows into broader DevOps pipelines, enabling developers to maintain control and flexibility in how they manage AI-driven applications.

## Why FastAgency?

FastAgency bridges the gap between rapid prototyping and production-ready deployment, empowering developers to bring their multi-agent systems to life quickly and efficiently. By integrating familiar frameworks like AG2 (formerly AutoGen), providing powerful API integration, and offering robust CI testing tools, FastAgency reduces the complexity and overhead typically associated with deploying AI agents in real-world applications.

Whether you’re building interactive console tools, developing fully-featured web apps, or orchestrating large-scale multi-agent systems, FastAgency is built to help you deploy faster, more reliably, and with greater flexibility.

### Supported Runtimes

Currently, the only supported runtime is [**AG2** (formerly AutoGen)](https://github.com/ag2ai/ag2).

### Supported User Interfaces

FastAgency currently supports workflows defined using AG2 (formerly AutoGen) and provides options for different types of applications:

- **Console**: Use the [**`ConsoleUI`**](fastagency/ui/console/console.py) interface for command-line based interaction. This is ideal for developing and testing workflows in a text-based environment.

- [**Mesop**](https://mesop-dev.github.io/mesop/): Utilize [**`MesopUI`**](fastagency/ui/mesop/mesop.py) for web-based applications. This interface is suitable for creating web applications with a user-friendly interface.

### Supported Network Adapters

FastAgency can use chainable network adapters that can be used to easily create
scalable, production ready architectures for serving your workflows. Currently, we
support the following network adapters:

- [**REST API**](https://en.wikipedia.org/wiki/REST) via [**FastAPI**](https://fastapi.tiangolo.com/): Use the [**`FastAPIAdapter`**](../api/fastagency/adapters/fastapi/FastAPIAdapter.md) to serve your workflow using [**FastAPI**](https://fastapi.tiangolo.com/) server. This setup allows you to work your workflows in multiple workers and serve them using the highly extensible and stable ASGI server.

- [**NATS.io**](https://nats.io/) via [**FastStream**](https://github.com/ag2ai/faststream): Utilize the [**`NatsAdapter`**](../api/fastagency/adapters/nats/NatsAdapter.md) to use [**NATS.io MQ**](https://nats.io/) message broker for highly-scalable, production-ready setup. This interface is suitable for setups in VPN-s or, in combination with the [**`FastAPIAdapter`**](../api/fastagency/adapters/fastapi/FastAPIAdapter.md) to serve public workflows in an authenticated, secure manner.


## Quick Start

### Project setup

We **strongly recommend** using [**Cookiecutter**](../cookiecutter/index.md) for setting up a FastAgency project. It creates the project folder structure, default workflow, automatically installs all the necessary requirements, and creates a [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) that can be used with [Visual Studio Code](https://code.visualstudio.com/) for development.

1. Install Cookiecutter with the following command:
    ```console
    pip install cookiecutter
    ```

2. Run the `cookiecutter` command:
    ```console
    cookiecutter https://github.com/ag2ai/cookiecutter-fastagency.git
    ```

3. Assuming that you used the default values, you should get the following output:
    ```console
    [1/4] project_name (My FastAgency App):
    [2/4] project_slug (my_fastagency_app):
    [3/4] Select app_type
        1 - fastapi+mesop
        2 - mesop
        3 - nats+fastapi+mesop
        Choose from [1/2/3] (1): 1
    [4/4] Select python_version
        1 - 3.12
        2 - 3.11
        3 - 3.10
        Choose from [1/2/3] (1):
    [5/5] Select authentication
        1 - none
        2 - google
        Choose from [1/2] (1):
    ```

4. To run LLM-based applications, you need an API key for the LLM used. The most commonly used LLM is [OpenAI](https://platform.openai.com/docs/models). To use it, create an [OpenAI API Key](https://openai.com/index/openai-api/) and set it as an environment variable in the terminal using the following command:

    ```console
    export OPENAI_API_KEY=openai_api_key_here
    ```

5. Open the generated project in [Visual Studio Code](https://code.visualstudio.com/) with the following command:
    ```console
    code my_fastagency_app
    ```

6. Once the project is opened, you will get the following option to reopen it in a devcontainer:

    <img src="https://fastagency.ai/0.3/user-guide/getting-started/images/reopen-in-container.png" width="600" class="center">

7. After reopening the project in devcontainer, you can verify that the setup is correct by running the provided tests with the following command:

    ```console
    pytest -s
    ```

    You should get the following output if everything is correctly setup.
    ```console
    =================================== test session starts ===================================
    platform linux -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0
    rootdir: /workspaces/my_fastagency_app
    configfile: pyproject.toml
    plugins: asyncio-0.24.0, anyio-4.6.2.post1
    asyncio: mode=Mode.STRICT, default_loop_scope=None
    collected 1 item

    tests/test_workflow.py .                                                            [100%]

    ==================================== 1 passed in 1.02s ====================================
    ```
-----

### Workflow Development

#### Define the Workflow

You need to define the workflow that your application will use. This is where you specify how the agents interact and what they do. Here's a simple example of a workflow definition as it is generated by the cookie cutter under `my_fastagency_app/workflow.py`:

```python
import os
from typing import Any

from autogen import ConversableAgent, LLMConfig
from fastagency import UI
from fastagency.runtimes.ag2 import Workflow

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
)

wf = Workflow()


@wf.register(name="simple_learning", description="Student and teacher learning chat")  # type: ignore[misc]
def simple_workflow(ui: UI, params: dict[str, Any]) -> str:
    initial_message = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="I can help you learn about mathematics. What subject you would like to explore?",
    )

    with llm_config:
      student_agent = ConversableAgent(
          name="Student_Agent",
          system_message="You are a student willing to learn.",
      )
      teacher_agent = ConversableAgent(
          name="Teacher_Agent",
          system_message="You are a math teacher.",
      )

    response = student_agent.run(
        teacher_agent,
        message=initial_message,
        summary_method="reflection_with_llm",
        max_turns=3,
    )

    return ui.process(response)
```

This code snippet sets up a simple learning chat between a student and a teacher. It defines the agents and how they should interact and specify how the conversation should be summarized.

#### Run and Debug the Workflow

To ensure that the workflow we have defined is working properly, we can test it locally using MesopUI. The code below can be found under `my_fastagency_app/local/main_mesop.py` and imports the defined workflow and sets up MesopUI.

You can run the Mesop application locally with the following command on Linux and MacOS:

```console
gunicorn my_fastagency_app.local.main_mesop:app
```

On Windows, please use the following command:
```console
waitress-serve --listen=0.0.0.0:8000 my_fastagency_app.local.main_mesop:app
```

Open the MesopUI URL [http://localhost:8000](http://localhost:8000) in your browser. You can now use the graphical user interface to start, run, test and debug the autogen workflow manually.

![Initial message](https://fastagency.ai/latest/user-guide/getting-started/images/chat-init.png)


## Deployment

### Building the Docker Image

If you created the project using Cookiecutter, then building the Docker image is as simple as running the provided script, as shown below:

```console
./scripts/build_docker.sh
```

### Running the Docker Image

Similarly, running the Docker container is as simple as running the provided script, as shown below:

```console
./scripts/run_docker.sh
```

### Deploying to Fly.io

If you created the project using Cookiecutter, there are built-in scripts to deploy your workflow to [**Fly.io**](https://fly.io/). In Fly.io, the application namespace is global, so the application name you chose might already be taken. To check your application's name availability and to reserve it, you can run the following script:

```console
./scripts/register_to_fly_io.sh
```

Once you have reserved your application name, you can test whether you can deploy your application to Fly.io using the following script:

```console
./scripts/deploy_to_fly_io.sh
```

This is only for testing purposes. You should deploy using [**GitHub Actions**](https://github.com/features/actions){target="_blank"} as explained below.

Cookiecutter generated all the necessary files to deploy your application to Fly.io using [**GitHub Actions**](https://github.com/features/actions). Simply push your code to your github repository's **main** branch and GitHub Actions will automatically deploy your application to Fly.io. For this, you need to set the following secrets in your GitHub repository:

- `FLY_API_TOKEN`
- `OPENAI_API_KEY`

To learn how to create keys and add them as secrets, use the following links:

  - [**Creating a Fly.io API token**](https://fly.io/docs/security/tokens/#manage-tokens-in-the-dashboard)
  - [**Creating an OpenAI API key**](https://platform.openai.com/api-keys)
  - [**Adding secrets to your GitHub repository**](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository)

## Future Plans

We are actively working on expanding FastAgency’s capabilities. In addition to supporting AG2 (formerly AutoGen), we plan to integrate support for other frameworks, other network providers and other UI frameworks.
---

## ⭐⭐⭐ Stay in touch ⭐⭐⭐

Stay up to date with new features and integrations by following our documentation and community updates on our [**Discord server**](https://discord.gg/kJjSGWrknU). FastAgency is continually evolving to support new frameworks, APIs, and deployment strategies, ensuring you remain at the forefront of AI-driven development.

Last but not least, show us your support by giving a star to our [**GitHub repository**](https://github.com/ag2ai/fastagency/).


Your support helps us to stay in touch with you and encourages us to
continue developing and improving the framework. Thank you for your
support!

---

## Contributors

Thanks to all of these amazing people who made the project better!

<a href="https://github.com/ag2ai/fastagency/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=ag2ai/fastagency"/>
</a>
