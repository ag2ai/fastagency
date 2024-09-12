import json
from collections.abc import Iterable, Iterator
from typing import Optional
from uuid import uuid4

import mesop as me

from fastagency.core.base import AskingMessage, WorkflowCompleted
from fastagency.core.io.mesop.base import MesopMessage
from fastagency.core.io.mesop.components.inputs import input_user_feedback
from fastagency.core.io.mesop.send_prompt import send_user_feedback_to_autogen

from ...base import (
    IOMessage,
    IOMessageVisitor,
    MultipleChoice,
    SystemMessage,
    TextInput,
    TextMessage,
)
from .components.ui_common import darken_hex_color
from .data_model import Conversation, ConversationMessage, State


def consume_responses(responses: Iterable[MesopMessage]) -> Iterator[None]:
    for message in responses:
        state = me.state(State)
        handle_message(state, message)
        yield
        me.scroll_into_view(key="end_of_messages")
        yield
    yield


def message_box(message: ConversationMessage, read_only: bool) -> None:
    io_message_dict = json.loads(message.io_message_json)
    level = message.level
    conversation_id = message.conversation_id
    io_message = IOMessage.create(**io_message_dict)
    visitor = MesopGUIMessageVisitor(level, conversation_id, message, read_only)
    visitor.process_message(io_message)


def handle_message(state: State, message: MesopMessage) -> None:
    conversation = state.conversation
    messages = conversation.messages
    level = message.conversation.level
    conversation_id = message.conversation.id
    io_message = message.io_message
    message_dict = io_message.model_dump()
    message_json = json.dumps(message_dict)
    conversation_message = ConversationMessage(
        level=level,
        conversation_id=conversation_id,
        io_message_json=message_json,
        feedback=[],
    )
    messages.append(conversation_message)
    conversation.messages = list(messages)
    if isinstance(io_message, AskingMessage):
        conversation.waiting_for_feedback = True
        conversation.completed = False
    if isinstance(io_message, WorkflowCompleted):
        conversation.completed = True
        conversation.waiting_for_feedback = False
        if not conversation.is_from_the_past:
            uuid: str = uuid4().hex
            becomme_past = Conversation(
                id=uuid,
                title=conversation.title,
                messages=conversation.messages,
                completed=True,
                is_from_the_past=True,
                waiting_for_feedback=False,
            )
            state.past_conversations.insert(0, becomme_past)


class MesopGUIMessageVisitor(IOMessageVisitor):
    def __init__(
        self,
        level: int,
        conversation_id: str,
        conversation_message: ConversationMessage,
        read_only: bool = False,
    ) -> None:
        """Initialize the MesopGUIMessageVisitor object.

        Args:
            level (int): The level of the message.
            conversation_id (str): The ID of the conversation.
            conversation_message (ConversationMessage): Conversation message that wraps the visited io_message
            read_only (bool): Input messages are disabled in read only mode
        """
        self._level = level
        self._conversation_id = conversation_id
        self._readonly = read_only
        self._conversation_message = conversation_message

    def _has_feedback(self) -> bool:
        return len(self._conversation_message.feedback) > 0

    def _provide_feedback(self, feedback: str) -> Iterator[None]:
        state = me.state(State)
        conversation = state.conversation
        conversation.feedback = ""
        conversation.waiting_for_feedback = False
        yield
        me.scroll_into_view(key="end_of_messages")
        yield
        responses = send_user_feedback_to_autogen(feedback)
        yield from consume_responses(responses)

    def visit_default(self, message: IOMessage) -> None:
        base_color = "#aff"
        with me.box(
            style=me.Style(
                background=base_color,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            self._header(message, base_color)
            me.markdown(message.type)

    def visit_text_message(self, message: TextMessage) -> None:
        base_color = "#fff"
        with me.box(
            style=me.Style(
                background=base_color,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            self._header(message, base_color, title="Text message")
            me.markdown(message.body)

    def visit_system_message(self, message: SystemMessage) -> None:
        base_color = "#bff"
        with me.box(
            style=me.Style(
                background=base_color,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            self._header(message, base_color, title="System Message")
            me.markdown(json.dumps(message.message, indent=2))

    def visit_text_input(self, message: TextInput) -> str:
        def on_input(ev: me.RadioChangeEvent) -> Iterator[None]:
            state = me.state(State)
            feedback = state.conversation.feedback
            self._conversation_message.feedback = [feedback]
            yield from self._provide_feedback(feedback)

        base_color = "#dff"
        prompt = message.prompt if message.prompt else "Please enter a value"
        if message.suggestions:
            suggestions = ",".join(suggestion for suggestion in message.suggestions)
            prompt += "\n" + suggestions

        with me.box(
            style=me.Style(
                background=base_color,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            self._header(message, base_color, title="Input requested")
            me.markdown(prompt)
            input_user_feedback(
                on_input, disabled=self._readonly or self._has_feedback()
            )
        return ""

    def visit_multiple_choice(self, message: MultipleChoice) -> str:
        def on_change(ev: me.RadioChangeEvent) -> Iterator[None]:
            feedback = ev.value
            self._conversation_message.feedback = [feedback]
            yield from self._provide_feedback(feedback)

        base_color = "#dff"
        prompt = message.prompt if message.prompt else "Please enter a value"
        if message.choices:
            options = map(
                lambda choice: me.RadioOption(
                    label=(choice if choice != message.default else choice + " *"),
                    value=choice,
                ),
                message.choices,
            )
        if self._has_feedback():
            pre_selected = {"value": self._conversation_message.feedback[0]}
        else:
            pre_selected = {}
        with me.box(
            style=me.Style(
                background=base_color,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            self._header(message, base_color, title="Input requested")
            me.text(prompt)
            me.radio(
                on_change=on_change,
                disabled=self._readonly or self._has_feedback(),
                options=options,
                style=me.Style(display="flex", flex_direction="column"),
                **pre_selected,
            )
        return ""

    def process_message(self, message: IOMessage) -> Optional[str]:
        return self.visit(message)

    def _header(
        self, message: IOMessage, base_color: str, title: Optional[str] = None
    ) -> None:
        you_want_it_darker = darken_hex_color(base_color, 0.8)
        with me.box(
            style=me.Style(
                background=you_want_it_darker,
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin.symmetric(vertical=16),
            )
        ):
            h = title if title else message.type
            h += f" from: {message.sender}, to:{message.recipient}"
            if message.auto_reply:
                h += " (auto-reply)"
            me.markdown(h)
