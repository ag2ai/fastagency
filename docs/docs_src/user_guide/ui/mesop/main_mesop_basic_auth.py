import os
from typing import Any

import mesop as me
from autogen import ConversableAgent, LLMConfig

from fastagency import UI, FastAgency
from fastagency.runtimes.ag2 import Workflow
from fastagency.ui.mesop import MesopUI
from fastagency.ui.mesop.auth.basic_auth import BasicAuth
from fastagency.ui.mesop.styles import (
    MesopHomePageStyles,
    MesopMessagesStyles,
    MesopSingleChoiceInnerStyles,
)

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
)

wf = Workflow()


@wf.register(name="simple_learning", description="Student and teacher learning chat")
def simple_workflow(
    ui: UI, params: dict[str, Any]
) -> str:
    initial_message = ui.text_input(
        sender="Workflow",
        recipient="User",
        prompt="What do you want to learn today?",
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
        max_turns=5,
    )

    return ui.process(response)  # type: ignore[no-any-return]


security_policy=me.SecurityPolicy(allowed_iframe_parents=["https://acme.com"], allowed_script_srcs=["https://cdn.jsdelivr.net"])

styles=MesopHomePageStyles(
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
    ],
    root=me.Style(
        background="#e7f2ff",
        height="100%",
        font_family="Inter",
        display="flex",
        flex_direction="row",
    ),
    message=MesopMessagesStyles(
        single_choice_inner=MesopSingleChoiceInnerStyles(
            disabled_button=me.Style(
                margin=me.Margin.symmetric(horizontal=8),
                padding=me.Padding.all(16),
                border_radius=8,
                background="#64b5f6",
                color="#fff",
                font_size=16,
            ),
        )
    ),
)

# Initialize auth with username and password
auth = BasicAuth(
    # TODO: Replace `allowed_users` with the desired usernames and their
    # bcrypt-hashed passwords. One way to generate bcrypt-hashed passwords
    # is by using online tools such as https://bcrypt.online
    allowed_users={
        "harish": "$2y$10$4aH/.C.WritjZAYskA0Dq.htlFDJTa49UuxSVUlp9JCa2K3PgUkaG",  # nosemgrep
        "davor@ag2.ai": "$2y$10$Yz9GuF/bWmRFmnXFkauOwePT/U.VSUHdpMOX7GPB8GiklJE4HJZmG"  # nosemgrep
    }
)

ui = MesopUI(security_policy=security_policy, styles=styles, auth=auth)

app = FastAgency(provider=wf, ui=ui, title="Learning Chat")
