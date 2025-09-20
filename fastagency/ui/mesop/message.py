import json
import random
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Callable, Optional, Union
from uuid import UUID, uuid4

import mesop as me
import mesop.labs as mel

from fastagency.helpers import jsonify_string

from ...logging import get_logger
from ...messages import (
    AskingMessage,
    Error,
    FunctionCallExecution,
    IOMessage,
    KeepAlive,
    MessageProcessorMixin,
    MultipleChoice,
    SuggestedFunctionCall,
    SystemMessage,
    TextInput,
    TextMessage,
    WorkflowCompleted,
)
from ...runtimes.ag2.ag2 import create_ag2_event
from .components.inputs import input_text
from .data_model import Conversation, ConversationMessage, State
from .mesop import MesopMessage
from .send_prompt import get_more_messages, send_user_feedback_to_autogen
from .styles import MesopHomePageStyles, MesopMessageStyles
from .timer import wakeup_component

if TYPE_CHECKING:
    from autogen.events.agent_events import (
        ErrorEvent,
        RunCompletionEvent,
        TextEvent,
        UsingAutoReplyEvent,
    )

logger = get_logger(__name__)


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        return json.JSONEncoder.default(self, obj)


def consume_responses(responses: Iterable[MesopMessage]) -> Iterator[None]:
    for message in responses:
        state = me.state(State)
        handle_message(state, message)
        yield
        if not isinstance(message.io_message, KeepAlive):
            me.scroll_into_view(key="end_of_messages")
        yield
    yield


def handle_message(state: State, message: MesopMessage) -> None:
    conversation = state.conversation
    messages = conversation.messages
    level = message.conversation.level
    conversation_id = message.conversation.id
    io_message = message.io_message
    message_dict = io_message.model_dump()
    message_json = json.dumps(message_dict, cls=UUIDEncoder)
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
                title=find_suitable_title(conversation),
                messages=conversation.messages,
                completed=True,
                is_from_the_past=True,
                waiting_for_feedback=False,
            )
            state.past_conversations.insert(0, becomme_past)


def find_suitable_title(conversation: Conversation) -> str:
    messages = conversation.messages
    first_input = next(filter(lambda m: m.feedback_completed, messages), None)
    if first_input and first_input.feedback:
        return first_input.feedback[0]

    state = me.state(State)
    return f"The conversation that shall remain unnamed ({random.randint(0, 100)}-{len(state.past_conversations)})"  # nosec: B311


def message_box(
    message: ConversationMessage, read_only: bool, *, styles: MesopHomePageStyles
) -> None:
    io_message_dict = json.loads(message.io_message_json)
    level = message.level
    conversation_id = message.conversation_id
    try:
        io_message = IOMessage.create(**io_message_dict)
    except Exception:
        io_message = create_ag2_event(
            **io_message_dict,
        )

    visitor = MesopGUIMessageVisitor(level, conversation_id, message, styles, read_only)
    visitor.process_message(io_message)


class MesopGUIMessageVisitor(MessageProcessorMixin):
    def __init__(
        self,
        level: int,
        conversation_id: str,
        conversation_message: ConversationMessage,
        styles: MesopHomePageStyles,
        read_only: bool = False,
    ) -> None:
        """Initialize the MesopGUIMessageVisitor object.

        Args:
            level (int): The level of the message.
            conversation_id (str): The ID of the conversation.
            conversation_message (ConversationMessage): Conversation message that wraps the visited io_message
            styles (MesopHomePageStyles): Styles for the message
            read_only (bool): Input messages are disabled in read only mode
        """
        self._level = level
        self._conversation_id = conversation_id
        self._readonly = read_only
        self._conversation_message = conversation_message
        self._styles = styles

    def _has_feedback(self) -> bool:
        return len(self._conversation_message.feedback) > 0

    def _is_completed(self) -> bool:
        return self._conversation_message.feedback_completed

    def _provide_feedback(self, feedback: str) -> Iterator[None]:
        logger.debug(f"MesopGUIMessageVisitor._provide_feedback({feedback=})")
        state = me.state(State)
        conversation = state.conversation
        conversation.feedback = ""
        conversation.waiting_for_feedback = False
        yield
        me.scroll_into_view(key="end_of_messages")
        yield
        responses = send_user_feedback_to_autogen(feedback)
        yield from consume_responses(responses)

    def _render_content(self, content: str, msg_md_style: me.Style) -> None:
        content = jsonify_string(content)
        me.markdown(content, style=msg_md_style)

    def visit_default(
        self,
        message: IOMessage,
        *,
        content: Optional[str] = None,
        style: Optional[MesopMessageStyles] = None,
        error: Optional[bool] = False,
        inner_callback: Optional[Callable[..., None]] = None,
        scrollable: Optional[bool] = False,
    ) -> None:
        # logger.info(f"visit_default: {message=}")
        style = style or self._styles.message.default
        title = message.type.replace("_", " ").capitalize()
        title = "[Error] " + title if error else title
        with me.box(style=style.box or self._styles.message.default.box):
            self._header(
                message,
                title=title,
                box_style=style.header_box,
                md_style=style.header_md,
            )

            if isinstance(message, IOMessage):
                content = content or json.dumps(
                    message.model_dump()["content"], cls=UUIDEncoder
                )
            else:
                content = (
                    message.content.content
                    if hasattr(message, "content")
                    and hasattr(message.content, "content")
                    else ""
                )
            content = self._body_to_str(content)

            self._render_content(
                content,
                msg_md_style=style.scrollable_md
                or self._styles.message.default.scrollable_md
                if scrollable
                else style.md or self._styles.message.default.md,
            )

            if inner_callback:
                inner_callback()

    def visit_text(self, message: "TextEvent") -> None:
        content = message.content.content
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.text,
        )

    def visit_using_auto_repy(self, message: "UsingAutoReplyEvent") -> None:
        content = None
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.system,
        )

    def visit_run_completion(self, message: "RunCompletionEvent") -> None:
        # We can ignore the RunCompletionEvent as we handle RunResponse already
        pass

    def visit_text_message(self, message: TextMessage) -> None:
        content = message.body if message.body else ""
        content = content if content.strip() != "" else "*(empty message)*"
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.text,
        )

    def visit_error(self, message: "Union[Error, ErrorEvent]") -> None:
        content = (
            f"### {message.short}\n{message.long}"
            if isinstance(message, Error)
            else str(message.content.error)
        )
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.error,
            scrollable=True,
            error=True,
        )

    def visit_system_message(self, message: SystemMessage) -> None:
        content = (
            f"""#### **{message.message["heading"]}**

{message.message["body"]}
"""
            if "heading" in message.message and "body" in message.message
            else json.dumps(message.message)
        )
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.system,
            scrollable=True,
        )

    def visit_keep_alive(self, message: KeepAlive) -> None:
        def on_wakeup(e: mel.WebEvent) -> Iterator[None]:
            logger.debug("waking up, after the keep alive")
            self._conversation_message.feedback_completed = True
            yield from consume_responses(get_more_messages())

        with me.box():
            if not (self._readonly or self._conversation_message.feedback_completed):
                wakeup_component(on_wakeup=on_wakeup)

    def visit_suggested_function_call(self, message: SuggestedFunctionCall) -> None:
        content = f"""**function_name**: `{message.function_name}`<br>
**call_id**: `{message.call_id}`<br>
**arguments**: {json.dumps(message.arguments)}"""
        self.visit_default(
            message,
            content=content,
            style=self._styles.message.suggested_function_call,
            scrollable=True,
        )

    def visit_function_call_execution(self, message: FunctionCallExecution) -> None:
        content = f"""**function_name**: `{message.function_name}`<br>
**call_id**: `{message.call_id}`<br>
**retval**: {message.retval}"""
        return self.visit_default(
            message,
            content=content,
            style=self._styles.message.function_call_execution,
            scrollable=True,
        )

    def visit_text_input(self, message: TextInput) -> str:
        def on_input(feedback: str) -> Iterator[None]:
            self._conversation_message.feedback = [feedback]
            self._conversation_message.feedback_completed = True
            yield from self._provide_feedback(feedback)

        def value_if_completed() -> Optional[str]:
            message = self._conversation_message
            return message.feedback[0] if message.feedback_completed else None

        # base_color = "#dff"
        prompt = message.prompt if message.prompt else "Please enter a value"
        if message.suggestions:
            suggestions = ",".join(suggestion for suggestion in message.suggestions)
            prompt += "\n Suggestions: " + suggestions

        self.visit_default(
            message,
            content=prompt,
            style=self._styles.message.text_input,
            inner_callback=lambda: input_text(
                on_input,
                key="prompt",
                disabled=self._readonly or self._has_feedback(),
                value=value_if_completed(),
                style=self._styles.message.text_input_inner,
            ),
        )
        return ""

    def visit_multiple_choice(self, message: MultipleChoice) -> str:
        if message.single:
            return self._visit_single_choice(message)
        else:
            return self._visit_many_choices(message)

    def _visit_single_choice(self, message: MultipleChoice) -> str:
        def on_click(ev: me.ClickEvent) -> Iterator[None]:
            self._conversation_message.feedback_completed = True
            self._conversation_message.feedback = [ev.key]
            yield from self._provide_feedback(ev.key)

        prompt = message.prompt if message.prompt else "Please enter a value"

        def inner_callback() -> None:
            with me.box(
                style=self._styles.message.single_choice_inner.box,
            ):
                for choice in message.choices:
                    disabled = self._readonly or self._is_completed()
                    selected = choice in self._conversation_message.feedback
                    if selected:
                        style = self._styles.message.single_choice_inner.selected_button
                    elif disabled:
                        style = self._styles.message.single_choice_inner.disabled_button
                    else:
                        style = self._styles.message.single_choice_inner.button
                    me.button(
                        label=choice,
                        on_click=on_click,
                        color="primary",
                        type="flat",
                        key=choice,
                        disabled=disabled,
                        style=style,
                    )

        self.visit_default(
            message,
            content=prompt,
            style=self._styles.message.text_input,
            inner_callback=inner_callback,
        )
        return ""

    def _visit_many_choices(self, message: MultipleChoice) -> str:
        def on_change(ev: me.CheckboxChangeEvent) -> None:
            message_feedback = self._conversation_message.feedback
            choice = ev.key
            yes_no = ev.checked
            if yes_no:
                if choice not in message_feedback:
                    message_feedback.append(choice)
            else:
                if choice in message_feedback:
                    message_feedback.remove(choice)

        def on_click(ev: me.ClickEvent) -> Iterator[None]:
            message_feedback = self._conversation_message.feedback
            feedback = ",".join(message_feedback)
            self._conversation_message.feedback_completed = True
            yield from self._provide_feedback(feedback)

        def should_be_checked(option: str) -> bool:
            conversation_message = self._conversation_message
            return option in conversation_message.feedback

        prompt = message.prompt if message.prompt else "Please select a value:"

        def inner_callback() -> None:
            if message.choices:
                with me.box(
                    style=self._styles.message.multiple_choice_inner.box,
                ):
                    for option in message.choices:
                        me.checkbox(
                            label=option,
                            key=option,
                            checked=should_be_checked(option),
                            on_change=on_change,
                            disabled=self._readonly or self._is_completed(),
                            style=self._styles.message.multiple_choice_inner.checkbox,
                        )
                    me.button(
                        label="OK",
                        on_click=on_click,
                        color="primary",
                        type="flat",
                        style=self._styles.message.multiple_choice_inner.button,
                    )

        self.visit_default(
            message,
            content=prompt,
            style=self._styles.message.text_input,
            inner_callback=inner_callback,
        )
        return ""

    def render_error_message(
        self,
        e: Exception,
        message: IOMessage,
        *,
        content: Optional[str] = None,
        style: Optional[MesopMessageStyles] = None,
    ) -> None:
        style = self._styles.message.error or self._styles.message.default
        title = "[Error] " + message.type.replace("_", " ").capitalize()

        with me.box(style=style.box or self._styles.message.default.box):
            self._header(
                message,
                title=title,
                box_style=style.header_box or self._styles.message.default.header_box,
                md_style=style.header_md or self._styles.message.default.header_md,
            )

            content = (
                "Failed to render message:"
                + json.dumps(message.model_dump(), indent=2, cls=UUIDEncoder)
                + f"<br>Error: {e}"
            )

            logger.warning(f"render_error_message: {content=}")
            logger.warning(e, exc_info=True)
            # me.markdown(content, style=style.md or self._styles.message.default.md)
            self._render_content(content, style.md or self._styles.message.default.md)

    def process_message(self, message: IOMessage) -> Optional[str]:
        try:
            return self.visit(message)
        except Exception as e:
            logger.warning(f"Failed to render message: {e}")
            self.render_error_message(e, message)
            return None

    def _header(
        self,
        message: IOMessage,
        *,
        title: Optional[str] = None,
        box_style: Optional[me.Style] = None,
        md_style: Optional[me.Style] = None,
    ) -> None:
        if isinstance(message, IOMessage):
            sender = message.sender
            recipient = message.recipient
        else:
            sender = message.content.sender
            recipient = message.content.recipient

        with me.box(style=box_style or self._styles.message.default.header_box):
            h = title if title else message.type
            if sender and recipient:
                h += f": {sender} (to {recipient})"
            elif sender:
                h += f": to {sender}"
            elif hasattr(message, "recipient") and message.recipient:
                h += f": from {recipient}"
            if hasattr(message, "auto_reply") and message.auto_reply:
                h += " (auto-reply)"
            h = f"**{h}**"
            # style=me.Style(padding=me.Padding(top=8, right=16, left=16, bottom=8))
            me.markdown(h, style=md_style or self._styles.message.default.header_md)
