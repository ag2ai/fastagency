import getpass
import json
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

from ...base import (
    CreateWorkflowUIMixin,
    Runnable,
)
from ...logging import get_logger
from ...messages import (
    Error,
    IOMessage,
    MessageProcessorMixin,
    MultipleChoice,
    TextInput,
    TextMessage,
)

if TYPE_CHECKING:
    from autogen.events.agent_events import (
        ExecuteFunctionEvent,
        InputRequestEvent,
        RunCompletionEvent,
        TerminationEvent,
        TextEvent,
        UsingAutoReplyEvent,
    )

logger = get_logger(__name__)


class ConsoleUI(MessageProcessorMixin, CreateWorkflowUIMixin):  # implements UI
    @dataclass
    class ConsoleMessage:
        """A console message."""

        sender: Optional[str]
        recipient: Optional[str]
        heading: Optional[str]
        body: Optional[Union[str, list[dict[str, Any]]]]

    def __init__(
        self,
        super_conversation: Optional["ConsoleUI"] = None,
    ) -> None:
        """Initialize the console UI object.

        Args:
            super_conversation (Optional[UI], optional): The super conversation. Defaults to None.
        """
        self.super_conversation: Optional[ConsoleUI] = super_conversation
        self.sub_conversations: list[ConsoleUI] = []

    @contextmanager
    def create(self, app: Runnable, import_string: str) -> Iterator[None]:
        yield

    def start(
        self,
        *,
        app: Runnable,
        import_string: str,
        name: Optional[str] = None,
        params: dict[str, Any],
        single_run: bool = False,
    ) -> None:
        workflow_uuid = uuid4().hex
        ui = self.create_workflow_ui(workflow_uuid=workflow_uuid)
        name = name or app.provider.names[0]
        app.provider.run(name=name, ui=ui, **params)

    @property
    def level(self) -> int:
        return (
            0 if self.super_conversation is None else self.super_conversation.level + 1
        )

    def _format_message(self, console_msg: ConsoleMessage) -> str:
        body = self._body_to_str(console_msg.body)
        heading = f"[{console_msg.heading}]" if console_msg.heading else ""
        title = f"{console_msg.sender} (to {console_msg.recipient}) {heading}"[:74]

        s = f"""╭─ {title} {"─" * (74 - len(title))}─╮
│
{textwrap.indent(textwrap.fill(body, replace_whitespace=False, drop_whitespace=False), "│ ", predicate=lambda line: True)}
╰{"─" * 78}╯
"""
        # remove empty lines
        s = "\n".join([line for line in s.split("\n") if line.strip()])

        # add trailing withespace and │ to each line except the first and the last
        lines = s.split("\n")
        s = (
            lines[0]
            + "\n"
            + "\n".join([line + " " * (79 - len(line)) + "│" for line in lines[1:-1]])
            + "\n"
            + lines[-1]
            + "\n"
        )

        return s

    def _indent(self, text: str) -> str:
        return textwrap.indent(text, " " * 4 * self.level)

    def _format_and_print(self, console_msg: ConsoleMessage) -> None:
        msg = self._format_message(console_msg)
        msg = self._indent(msg)

        print(msg)  # noqa: T201 `print` found

    def visit_default(self, message: IOMessage) -> None:
        if hasattr(message, "content"):
            content = message.content
            console_msg = self.ConsoleMessage(
                sender=content.sender,
                recipient=content.recipient,
                heading=message.type,
                body=getattr(content, "content", None),
            )
            self._format_and_print(console_msg)
        else:
            content = message.model_dump()["content"]
            console_msg = self.ConsoleMessage(
                sender=message.sender,
                recipient=message.recipient,
                heading=message.type,
                body=json.dumps(content, indent=2),
            )
            self._format_and_print(console_msg)

    def visit_text(self, message: "TextEvent") -> None:
        content = message.content
        console_msg = self.ConsoleMessage(
            sender=content.sender,
            recipient=content.recipient,
            heading=message.type,
            body=content.content,
        )
        self._format_and_print(console_msg)

    def visit_using_auto_reply(self, message: "UsingAutoReplyEvent") -> None:
        # Do nothing if it is of type UsingAutoReplyEvent
        pass

    def visit_run_completion(self, message: "RunCompletionEvent") -> None:
        # We can ignore the RunCompletionEvent as we handle RunResponse already
        pass

    def visit_execute_function(self, message: "ExecuteFunctionEvent") -> None:
        content = message.content

        body = f"\n>>>>>>>> EXECUTING FUNCTION {content.func_name}...\nCall ID: {content.call_id}\nInput arguments: {content.arguments}"
        console_msg = self.ConsoleMessage(
            sender="Workflow",
            recipient=content.recipient,
            heading=message.type,
            body=body,
        )
        self._format_and_print(console_msg)

    def visit_error(self, message: IOMessage) -> None:
        """
        Handles error messages from both FastAgency and AG2 sources.
        AG2's ErrorEvent only has an 'error' field without sender/recipient,
        while FastAgency's Error has sender, recipient, short, and long fields.
        """
        # Case 1: Wrapped AG2 Event (has content attribute)
        if hasattr(message, "content"):
            content = message.content
            sender = "System"
            recipient = "User"

            if content is None:
                # Null content
                error_details = "An unknown error occurred."
            elif hasattr(content, "error"):
                # AG2 ErrorEvent structure
                error_details = str(getattr(content, "error", "Unknown AG2 error"))
            else:
                # Other wrapped event with unrecognized structure
                error_details = f"Unrecognized error content format: {str(content)}"

        # Case 2: Native FastAgency Error message (no content attribute)
        else:
            if hasattr(message, "sender") and hasattr(message, "recipient"):
                sender = message.sender
                recipient = message.recipient
                error_details = getattr(message, "long", getattr(message, "short", "No details provided."))
            else:
                # Fallback for unexpected message formats
                sender = "System"
                recipient = "User"
                error_details = f"An unhandled error message format was received: {str(message)}"

        console_msg = self.ConsoleMessage(
            sender=sender,
            recipient=recipient,
            heading="ERROR",
            body=str(error_details),
        )
        self._format_and_print(console_msg)

    def visit_termination(self, message: "TerminationEvent") -> None:
        pass

    def visit_text_message(self, message: TextMessage) -> None:
        console_msg = self.ConsoleMessage(
            sender=message.sender,
            recipient=message.recipient,
            heading=message.type,
            body=message.body,
        )
        self._format_and_print(console_msg)

    def visit_text_input(self, message: TextInput) -> str:
        suggestions = (
            f" (suggestions: {', '.join(message.suggestions)})"
            if message.suggestions
            else ""
        )
        console_msg = self.ConsoleMessage(
            sender=message.sender,
            recipient=message.recipient,
            heading=message.type,
            body=f"{message.prompt}{suggestions}:",
        )

        prompt = self._format_message(console_msg)
        prompt = self._indent(prompt)
        if message.password:
            return getpass.getpass(prompt)
        else:
            return input(prompt)

    def visit_input_request(self, message: "InputRequestEvent") -> str:
        prompt = message.content.prompt
        if message.content.password:
            result = getpass.getpass(prompt if prompt != "" else "Password: ")
        else:
            result = input(prompt)
        message.content.respond(result)
        return result

    def visit_multiple_choice(self, message: MultipleChoice) -> str:
        console_msg = self.ConsoleMessage(
            sender=message.sender,
            recipient=message.recipient,
            heading=message.type,
            body=f"{message.prompt} (choices: {', '.join(message.choices)}, default: {message.default})",
        )

        prompt = self._format_message(console_msg)
        prompt = self._indent(prompt)
        while True:
            # logger.info(f"visit_multiple_choice(): {prompt=}")
            retval = input(prompt)
            if retval in message.choices:
                return retval
            elif retval == "" and message.default:
                return message.default
            else:
                print(f"Invalid choice ('{retval}'). Please try again.")  # noqa: T201 `print` found

    def process_message(self, message: IOMessage) -> Optional[str]:
        # logger.info(f"process_message(): {message=}")
        return self.visit(message)

    # def process_streaming_message(self, message: IOStreamingMessage) -> str | None:
    #     raise NotImplementedError

    def create_subconversation(self) -> "ConsoleUI":
        sub_conversation = ConsoleUI(self)
        self.sub_conversations.append(sub_conversation)

        return sub_conversation
