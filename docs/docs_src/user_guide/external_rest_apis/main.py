import os

from autogen import UserProxyAgent
from autogen.agentchat import ConversableAgent

from fastagency import FastAgency
from fastagency import UI
from fastagency.ui.console import ConsoleUI
from fastagency.runtime.autogen.base import AutoGenWorkflows

from fastagency.api.openapi import OpenAPI


llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.0,
}

WEATHER_OPENAPI_URL = "https://weather.tools.fastagency.ai/openapi.json"

wf = AutoGenWorkflows()


@wf.register(name="simple_weather", description="Weather chat")
def weather_workflow(ui: UI, initial_message: str, session_id: str) -> str:

    weather_api = OpenAPI.create(openapi_url=WEATHER_OPENAPI_URL)

    user_agent = UserProxyAgent(
        name="User_Agent",
        system_message="You are a user agent",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    weather_agent = ConversableAgent(
        name="Weather_Agent",
        system_message="You are a weather agent",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    weather_api.register_for_llm(weather_agent)
    weather_api.register_for_execution(user_agent)

    chat_result = user_agent.initiate_chat(
        weather_agent,
        message=initial_message,
        summary_method="reflection_with_llm",
        max_turns=3,
    )

    return chat_result.summary  # type: ignore[no-any-return]


app = FastAgency(wf=wf, ui=ConsoleUI())
