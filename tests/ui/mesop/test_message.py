import json
import sys
from typing import Union
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from fastagency.messages import (
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
    from fastagency.ui.mesop.styles import MesopHomePageStyles
else:
    ConversationMessage = MagicMock()
    message_box = MagicMock()
    MesopHomePageStyles = MagicMock()


def assert_called_with_one_of(
    mock: MagicMock, *args: str, key: Union[int, str] = 0
) -> None:
    assert mock.call_count == len(args), (
        f"Expected {len(args)} calls, got {mock.call_count}"
    )
    for call in mock.call_args_list:
        i = 0 if isinstance(key, int) else 1
        assert call[i][key] in args, f"Parameter '{call[i][key]}' not in {args}"


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Mesop is not supported in Python version 3.9 and below",
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

        workflow_uuid = uuid4().hex

        text_message = TextMessage(
            sender="sender",
            recipient="recipient",
            body="this is a test message",
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(text_message.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "this is a test message",
            "**Text message: sender (to recipient)**",
        )

    def test_system_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        system_message = SystemMessage(
            sender="sender",
            recipient="recipient",
            message={"type": "test", "data": "this is a test message"},
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(system_message.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**System message: sender (to recipient)**",
            '\n```\n{\n    "type": "test",\n    "data": "this is a test message"\n}\n```\n',
        )

    def test_suggested_function_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        suggested_function_call = SuggestedFunctionCall(
            sender="sender",
            recipient="recipient",
            function_name="function_name",
            call_id="my_call_id",
            arguments={"arg1": "value1", "arg2": "value2"},
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(suggested_function_call.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Suggested function call: sender (to recipient)**",
            """**function_name**: `function_name`<br>
**call_id**: `my_call_id`<br>
**arguments**:
```
{
    "arg1": "value1",
    "arg2": "value2"
}
```
""",
        )

    def test_function_call_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        function_call_execution = FunctionCallExecution(
            sender="sender",
            recipient="recipient",
            function_name="function_name",
            call_id="my_call_id",
            retval="return_value",
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(function_call_execution.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Function call execution: sender (to recipient)**",
            """**function_name**: `function_name`<br>
**call_id**: `my_call_id`<br>
**retval**: return_value""",
        )

    def test_text_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        text_input = TextInput(
            sender="sender",
            recipient="recipient",
            prompt="Who is the president of the United States?",
            suggestions=["Donald Trump", "Joe Biden"],
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(text_input.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Text input: sender (to recipient)**",
            """Who is the president of the United States?
 Suggestions: Donald Trump,Joe Biden""",
        )

    def test_multiple_choice_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        multiple_choice = MultipleChoice(
            sender="sender",
            recipient="recipient",
            prompt="Who is the president of the United States?",
            choices=["Donald Trump", "Joe Biden"],
            default="Joe Biden",
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(multiple_choice.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Multiple choice: sender (to recipient)**",
            "Who is the president of the United States?",
        )
        assert_called_with_one_of(
            me.button,
            "Donald Trump",
            "Joe Biden",
            key="label",
        )

    def test_multiple_choice_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        multiple_choice = MultipleChoice(
            sender="sender",
            recipient="recipient",
            prompt="Who are Snow White helpers?",
            choices=["Doc", "Grumpy", "Happy", "Sleepy", "Bashful", "Sneezy", "Dopey"],
            single=False,
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(multiple_choice.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Multiple choice: sender (to recipient)**",
            "Who are Snow White helpers?",
        )
        assert me.checkbox.call_count == 7

    def test_workflow_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        me = self._apply_monkeypatch(monkeypatch)

        workflow_uuid = uuid4().hex

        workflow_completed = WorkflowCompleted(
            sender="sender",
            recipient="recipient",
            result="success",
            workflow_uuid=workflow_uuid,
        )
        io_message_json = json.dumps(workflow_completed.model_dump())

        message = ConversationMessage(
            io_message_json=io_message_json,
            conversation_id="conversation_id",
        )

        message_box(message=message, read_only=True, styles=MesopHomePageStyles())

        assert_called_with_one_of(
            me.markdown,
            "**Workflow completed: sender (to recipient)**",
            """
```
{
    "result": "success"
}
```
""",
        )
