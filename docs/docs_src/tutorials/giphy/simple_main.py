import os
from typing import Any

from autogen import ConversableAgent, LLMConfig, UserProxyAgent

from fastagency import UI, FastAgency
from fastagency.api.openapi import OpenAPI
from fastagency.api.openapi.security import APIKeyQuery
from fastagency.runtimes.ag2 import Workflow
from fastagency.ui.mesop import MesopUI

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
)

giphy_api_key = os.getenv("GIPHY_API_KEY", "")
openapi_url="https://raw.githubusercontent.com/ag2ai/fastagency/refs/heads/main/examples/openapi/giphy_openapi.json"
giphy_api = OpenAPI.create(openapi_url=openapi_url)
giphy_api.set_security_params(APIKeyQuery.Parameters(value=giphy_api_key))


wf = Workflow()

@wf.register(name="giphy_chat", description="Giphy chat")
def giphy_workflow(
    ui: UI, params: dict[str, Any]
) -> str:
    def is_termination_msg(msg: dict[str, Any]) -> bool:
        return msg["content"] is not None and "TERMINATE" in msg["content"]

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="ALWAYS",
        is_termination_msg=is_termination_msg,
    )

    system_message="""You are a helper agent that communicates with the user and
        Giphy API. When using API calls, always limit the response size to 5 items.
        When finished, write 'TERMINATE' to end the conversation."""

    with llm_config:
        giphy_agent = ConversableAgent(
            name="Giphy_Agent",
            system_message=system_message,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
        )

    wf.register_api(
        api=giphy_api,
        callers=giphy_agent,
        executors=user_proxy,
        functions=["random_gif", "search_gifs", "trending_gifs"],
    )

    initial_message = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="I can help you find images related to a certain subject. What kind of images would you like to find?",
    )

    response = user_proxy.run(
        giphy_agent,
        message=initial_message,
        summary_method="reflection_with_llm",
        max_turns=10,
    )

    return ui.process(response)  # type: ignore[no-any-return]


app = FastAgency(provider=wf, ui=MesopUI(), title="Giphy chat")
