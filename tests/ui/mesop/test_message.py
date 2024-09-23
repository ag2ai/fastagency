import json
import sys
from unittest.mock import MagicMock

import pytest

from fastagency.base import (
    FunctionCallExecution,
    MultipleChoice,
    SuggestedFunctionCall,
    SystemMessage,
    TextInput,
    TextMessage,
    WorkflowCompleted,
)

if sys.version_info >= (3, 10):
    from fastagency.ui.mesop.data_model import ConversationMessage
    from fastagency.ui.mesop.message import message_box
else:
    ConversationMessage = MagicMock()
    message_box = MagicMock()


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Mesop is not support in Python version 3.9 and below",
)
class TestMessageBox:
    def _apply_monkeypatch(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        me = MagicMock()
        me.markdown = MagicMock()

        monkeypatch.setattr("fastagency.ui.mesop.message.me", me)
        monkeypatch.setattr("fastagency.ui.mesop.components.inputs.me", me)

        return me

    def test_text_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        text_message = TextMessage(
            sender="sender",
            recipient="recipient",
            body="this is a test message",
        )
        io_message_json = json.dumps(text_message.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("this is a test message")
        me.markdown.assert_any_call("Text message: sender -> recipient")
        assert me.markdown.call_count == 2

    def test_system_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        system_message = SystemMessage(
            sender="sender",
            recipient="recipient",
            message={"type": "test", "data": "this is a test message"},
        )
        io_message_json = json.dumps(system_message.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("System Message: sender -> recipient")
        me.markdown.assert_any_call(
            '{\n  "type": "test",\n  "data": "this is a test message"\n}'
        )
        assert me.markdown.call_count == 2

    def test_suggested_function_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        suggested_function_call = SuggestedFunctionCall(
            sender="sender",
            recipient="recipient",
            function_name="function_name",
            call_id="my_call_id",
            arguments={"arg1": "value1", "arg2": "value2"},
        )
        io_message_json = json.dumps(suggested_function_call.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("Suggested Function Call: sender -> recipient")
        me.markdown.assert_any_call('{"arg1": "value1", "arg2": "value2"}')
        assert me.markdown.call_count == 2

    def test_function_call_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        function_call_execution = FunctionCallExecution(
            sender="sender",
            recipient="recipient",
            function_name="function_name",
            call_id="my_call_id",
            retval="return_value",
        )
        io_message_json = json.dumps(function_call_execution.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("Function Call Execution: sender -> recipient")
        me.markdown.assert_any_call('"return_value"')
        assert me.markdown.call_count == 2

    def test_text_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        text_input = TextInput(
            sender="sender",
            recipient="recipient",
            prompt="Who is the president of the United States?",
            suggestions=["Donald Trump", "Joe Biden"],
        )
        io_message_json = json.dumps(text_input.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("Input requested: sender -> recipient")
        me.markdown.assert_any_call(
            "Who is the president of the United States?\n Suggestions: Donald Trump,Joe Biden"
        )
        assert me.markdown.call_count == 2

    def test_multiple_choice_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        multiple_choice = MultipleChoice(
            sender="sender",
            recipient="recipient",
            prompt="Who is the president of the United States?",
            choices=["Donald Trump", "Joe Biden"],
            default="Joe Biden",
        )
        io_message_json = json.dumps(multiple_choice.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_called_once_with("Input requested: sender -> recipient")
        me.radio.assert_called_once()

    def test_multiple_choice_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        multiple_choice = MultipleChoice(
            sender="sender",
            recipient="recipient",
            prompt="Who are Snow White helpers?",
            choices=["Doc", "Grumpy", "Happy", "Sleepy", "Bashful", "Sneezy", "Dopey"],
            single=False,
        )
        io_message_json = json.dumps(multiple_choice.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_called_once_with("Input requested: sender -> recipient")
        assert me.checkbox.call_count == 7

    def test_workflow_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_completed = WorkflowCompleted(
            sender="sender", recipient="recipient", result="success"
        )
        io_message_json = json.dumps(workflow_completed.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True)

        me.markdown.assert_any_call("workflow_completed: sender -> recipient")
        me.markdown.assert_any_call("workflow_completed")
        assert me.markdown.call_count == 2
