import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import pytest
from autogen import LLMConfig
from autogen.agentchat import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from openai import InternalServerError

from fastagency.api.openapi import OpenAPI
from fastagency.base import UI
from fastagency.runtimes.ag2 import Workflow
from fastagency.runtimes.ag2.ag2 import _findall, _match
from fastagency.ui.console import ConsoleUI
from tests.conftest import InputMock

from ...docs_src.helpers import skip_internal_server_error


class TestPatternMatching:
    def test_end_of_message(self) -> None:
        chunk = "\n--------------------------------------------------------------------------------\n"
        assert _match("end_of_message", chunk)

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[31m\n>>>>>>>> USING AUTO REPLY...\x1b[0m\n",
            "\n>>>>>>>> USING AUTO REPLY...\n",
        ],
    )
    def test_auto_reply(self, chunk: str) -> None:
        assert _match("auto_reply", chunk)

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[33mUser_Proxy\x1b[0m (to Weatherman):\n\n",
            "User_Proxy (to Weatherman):\n\n",
        ],
    )
    def test_sender_recipient(self, chunk: str) -> None:
        assert _match("sender_recipient", chunk)
        sender, recipient = _findall("sender_recipient", chunk)

        assert sender == "User_Proxy"
        assert recipient == "Weatherman"

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[32m***** Suggested tool call (call_HNs2kuTywlvatTY5WHzMLfDL): get_daily_weather_daily_get *****\x1b[0m\n",
            "***** Suggested tool call (call_HNs2kuTywlvatTY5WHzMLfDL): get_daily_weather_daily_get *****\n",
        ],
    )
    def test_suggested_function_call(self, chunk: str) -> None:
        assert _match("suggested_function_call", chunk)

        call_id, function_name = _findall("suggested_function_call", chunk)

        assert call_id == "call_HNs2kuTywlvatTY5WHzMLfDL"
        assert function_name == "get_daily_weather_daily_get"

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[32m**********************************************************************************************\x1b[0m\n",
            "**********************************************************************************************\n",
        ],
    )
    def test_stars(self, chunk: str) -> None:
        assert _match("stars", chunk)

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[35m\n>>>>>>>> EXECUTING FUNCTION get_daily_weather_daily_get...\x1b[0m\n",
            "\n>>>>>>>> EXECUTING FUNCTION get_daily_weather_daily_get...\n",
        ],
    )
    def test_function_call_execution(self, chunk: str) -> None:
        assert _match("function_call_execution", chunk)

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[32m***** Response from calling tool (call_HNs2kuTywlvatTY5WHzMLfDL) *****\x1b[0m\n",
            "***** Response from calling tool (call_HNs2kuTywlvatTY5WHzMLfDL) *****\n",
        ],
    )
    def test_response_from_calling_tool(self, chunk: str) -> None:
        assert _match("response_from_calling_tool", chunk)

        call_id = _findall("response_from_calling_tool", chunk)
        assert call_id == "call_HNs2kuTywlvatTY5WHzMLfDL"  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize(
        "chunk",
        [
            "\x1b[31m\n>>>>>>>> NO HUMAN INPUT RECEIVED.\x1b[0m",
            "\n>>>>>>>> NO HUMAN INPUT RECEIVED.",
        ],
    )
    def test_no_human_input_received(self, chunk: str) -> None:
        assert _match("no_human_input_received", chunk)

    def test_user_interrupted(self) -> None:
        chunk = "USER INTERRUPTED\n"
        assert _match("user_interrupted", chunk)

    def test_arguments(self) -> None:
        chunk = 'Arguments: \n{"city": "Zagreb"}\n'
        assert _match("arguments", chunk)

        arguments = _findall("arguments", chunk)

        assert arguments == '{"city": "Zagreb"}'  # type: ignore[comparison-overlap]

    def test_auto_reply_input(self) -> None:
        prompt = "Replying as User_Proxy. Provide feedback to Weatherman. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "

        assert _match("auto_reply_input", prompt)

        sender, recipient = _findall("auto_reply_input", prompt)
        assert sender == "User_Proxy"
        assert recipient == "Weatherman"

    @pytest.mark.parametrize(
        "chunk",
        [
            "\u001b[32m\nNext speaker: Claude\n\u001b[0m",
            "Next speaker: Claude",
        ],
    )
    def test_next_speaker(self, chunk: str) -> None:
        assert _match("next_speaker", chunk), _match("next_speaker", chunk)


@pytest.mark.openai
@pytest.mark.xfail(strict=False, raises=InternalServerError)
@pytest.mark.parametrize(
    "agent_class",
    [
        ConversableAgent,
        MultimodalConversableAgent,
    ],
)
@skip_internal_server_error
def test_simple(
    openai_gpt4o_mini_llm_config: LLMConfig, agent_class: type[ConversableAgent]
) -> None:
    wf = Workflow()

    @wf.register(
        name="simple_learning", description="Student and teacher learning chat"
    )
    def simple_workflow(ui: UI, params: dict[str, Any]) -> str:
        initial_message = "What is triangle inequality?"

        student_agent = ConversableAgent(
            name="Student_Agent",
            system_message="You are a student willing to learn.",
            llm_config=openai_gpt4o_mini_llm_config,
        )
        teacher_agent = agent_class(
            name="Teacher_Agent",
            system_message="You are a math teacher.",
            llm_config=openai_gpt4o_mini_llm_config,
        )

        response = student_agent.run(
            teacher_agent,
            message=initial_message,
            summary_method="reflection_with_llm",
            max_turns=3,
        )

        return ui.process(response)  # type: ignore[no-any-return]

    name = "simple_learning"

    ui = ConsoleUI().create_workflow_ui(workflow_uuid=uuid4().hex)

    ui.workflow_started(
        sender="workflow",
        recipient="User",
        name=name,
    )

    result = wf.run(
        name=name,
        ui=ui,
    )

    ui.workflow_completed(
        sender="workflow",
        recipient="User",
        result=result,
    )

    assert "triangle inequality" in result.lower()


@pytest.mark.openai
@pytest.mark.xfail(strict=False, raises=InternalServerError)
@pytest.mark.parametrize(
    "agent_class",
    [
        ConversableAgent,
        MultimodalConversableAgent,
    ],
)
@skip_internal_server_error
def test_simple_async(
    openai_gpt4o_mini_llm_config: LLMConfig, agent_class: type[ConversableAgent]
) -> None:
    async def async_function() -> str:
        # Simulate some asynchronous work
        await asyncio.sleep(1)
        return "Async function completed!"

    wf = Workflow()

    @wf.register(  # type: ignore[type-var]
        name="simple_learning", description="Student and teacher learning chat"
    )
    async def simple_workflow(ui: UI, params: dict[str, Any]) -> str:
        initial_message = "What is triangle inequality?"

        student_agent = ConversableAgent(
            name="Student_Agent",
            system_message="You are a student willing to learn.",
            llm_config=openai_gpt4o_mini_llm_config,
        )
        teacher_agent = agent_class(
            name="Teacher_Agent",
            system_message="You are a math teacher.",
            llm_config=openai_gpt4o_mini_llm_config,
        )

        async_func_response = await async_function()
        assert async_func_response == "Async function completed!"

        response = await student_agent.a_run(
            teacher_agent,
            message=initial_message,
            summary_method="reflection_with_llm",
            max_turns=3,
        )

        return await ui.async_process(response)  # type: ignore[no-any-return]

    name = "simple_learning"

    ui = ConsoleUI().create_workflow_ui(workflow_uuid=uuid4().hex)

    ui.workflow_started(
        sender="workflow",
        recipient="User",
        name=name,
    )

    result = wf.run(
        name=name,
        ui=ui,
    )

    ui.workflow_completed(
        sender="workflow",
        recipient="User",
        result=result,
    )

    assert "triangle inequality" in result.lower()


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Test fails on macOS due to docker usage settings"
)
def test_register_api(openai_gpt4o_mini_llm_config: LLMConfig) -> None:
    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="ALWAYS",
    )
    assistant = ConversableAgent(
        name="Teacher_Agent",
        system_message="You are a math teacher.",
        llm_config=openai_gpt4o_mini_llm_config,
    )
    json_path = (
        Path(__file__).parents[2] / "api" / "openapi" / "templates" / "openapi.json"
    )
    assert json_path.exists()
    openapi_json = json_path.read_text()
    client = OpenAPI.create(openapi_json=openapi_json)

    wf = Workflow()
    function_to_register = "update_item_items__item_id__ships__ship__put"
    wf.register_api(
        api=client,
        callers=assistant,
        executors=user_proxy,
        functions=function_to_register,
    )

    tools = assistant.llm_config["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == function_to_register


@pytest.mark.openai
@pytest.mark.xfail(strict=False, raises=InternalServerError)
class TestWorkflowWithHumanInputAlways:
    @pytest.fixture
    def wf(self, openai_gpt4o_mini_llm_config: LLMConfig) -> Workflow:
        wf = Workflow()

        @wf.register(
            name="test_workflow",
            description="Test of user proxy with human input mode set to always",
        )
        def workflow(ui: UI, params: dict[str, Any]) -> str:
            initial_message = ui.text_input(
                sender="Workflow",
                recipient="User",
                prompt="I can help you learn about mathematics. What subject you would like to explore?",
            )
            user_proxy = UserProxyAgent(
                name="User_Proxy",
                human_input_mode="ALWAYS",
            )
            assistant = ConversableAgent(
                name="Teacher_Agent",
                system_message="You are a math teacher.",
                llm_config=openai_gpt4o_mini_llm_config,
            )

            @user_proxy.register_for_execution()  # type: ignore[misc]
            @assistant.register_for_llm(description="Get weather information")  # type: ignore[misc]
            def get_weather_info(
                city: Annotated[
                    str, "city for which the weather information is requested"
                ],
            ) -> str:
                return "The weather in Zagreb right now is heavy rain."

            response = user_proxy.run(
                assistant,
                message=initial_message,
                summary_method="reflection_with_llm",
                max_turns=5,
            )

            return ui.process(response)  # type: ignore[no-any-return]

        return wf

    # @pytest.mark.skip(reason="Test fails on macOS due to docker usage settings")
    @pytest.mark.parametrize("response", ["", "Reject"])
    def test(
        self, wf: Workflow, response: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", InputMock([response] * 7))

        result = wf.run(
            name="test_workflow",
            ui=ConsoleUI().create_workflow_ui(workflow_uuid=uuid4().hex),
            initial_message="What is the weather in Zagreb right now?",
        )

        assert result is not None
