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

    return ui.process(response)  # type: ignore[no-any-return]
