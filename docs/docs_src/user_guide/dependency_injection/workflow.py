import os
from typing import Annotated, Any

from autogen import UserProxyAgent, register_function, ConversableAgent, LLMConfig
from fastagency import UI
from fastagency.api.dependency_injection import inject_params
from fastagency.runtimes.ag2 import Workflow

account_ballace_dict = {
    ("alice", "password123"): 100,
    ("bob", "password456"): 200,
    ("charlie", "password789"): 300,
}


def get_balance(
    username: Annotated[str, "Username"],
    password: Annotated[str, "Password"],
) -> str:
    if (username, password) not in account_ballace_dict:
        return "Invalid username or password"
    return f"Your balance is {account_ballace_dict[(username, password)]}$"


llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
)

wf = Workflow()


@wf.register(name="bank_chat", description="Bank chat")  # type: ignore[misc]
def bank_workflow(ui: UI, params: dict[str, str]) -> str:
    username = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="Enter your username:",
    )
    password = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="Enter your password:",
    )

    with llm_config:
        user_agent = UserProxyAgent(
            name="User_Agent",
            system_message="You are a user agent",
            human_input_mode="NEVER",
        )
        banker_agent = ConversableAgent(
            name="Banker_Agent",
            system_message="You are a banker agent",
            human_input_mode="NEVER",
        )

    ctx: dict[str, Any] = {
        "username": username,
        "password": password,
    }
    get_balance_with_params = inject_params(get_balance, ctx)
    register_function(
        f=get_balance_with_params,
        caller=banker_agent,
        executor=user_agent,
        description="Get balance",
    )

    response = user_agent.run(
        banker_agent,
        message="We need to get user's balance.",
        summary_method="reflection_with_llm",
        max_turns=3,
    )

    return ui.process(response)  # type: ignore[no-any-return]
