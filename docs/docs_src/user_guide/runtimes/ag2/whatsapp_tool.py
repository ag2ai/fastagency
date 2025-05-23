import os
from typing import Any

from autogen import UserProxyAgent
from autogen import ConversableAgent, LLMConfig

from fastagency import UI, FastAgency
from fastagency.runtimes.ag2 import Workflow
from fastagency.runtimes.ag2.tools import WhatsAppTool
from fastagency.ui.console import ConsoleUI

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
)

wf = Workflow()


@wf.register(name="simple_whatsapp", description="WhatsApp chat")  # type: ignore[type-var]
def whatsapp_workflow(ui: UI, params: dict[str, Any]) -> str:
    def is_termination_msg(msg: dict[str, Any]) -> bool:
        return msg["content"] is not None and "TERMINATE" in msg["content"]

    initial_message = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="I can help you with sending a message over whatsapp, what would you like to send?",
    )

    with llm_config:
        user_agent = UserProxyAgent(
            name="User_Agent",
            system_message="You are a user agent, when the message is successfully sent, you can end the conversation by sending 'TERMINATE'",
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
        )
        assistant_agent = ConversableAgent(
            name="Assistant_Agent",
            system_message="You are a useful assistant for sending messages to whatsapp, use 447860099299 as your (sender) number.",
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
        )

    whatsapp = WhatsAppTool(
        whatsapp_api_key=os.getenv("WHATSAPP_API_KEY", ""),
    )

    whatsapp.register(
        caller=assistant_agent,
        executor=user_agent,
    )

    response = user_agent.run(
        assistant_agent,
        message=initial_message,
        summary_method="reflection_with_llm",
        max_turns=5,
    )

    return ui.process(response)  # type: ignore[no-any-return]


app = FastAgency(provider=wf, ui=ConsoleUI())
