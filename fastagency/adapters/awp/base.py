import asyncio
import json
import threading
from asyncio import Queue
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional
from uuid import uuid4

import autogen
import autogen.messages
import autogen.messages.agent_messages
from agentwire.core import (
    BaseMessage,
    CustomEvent,
    EventType,
    RunAgentInput,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserMessage,
)
from agentwire.encoder import EventEncoder
from asyncer import asyncify, syncify
from autogen.tools import Tool as AG2Tool
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from fastagency.logging import get_logger

from ...base import (
    UI,
    CreateWorkflowUIMixin,
    ProviderProtocol,
    Runnable,
    UIBase,
)
from ...exceptions import (
    FastAgencyConnectionError,
    FastAgencyKeyError,
)
from ...messages import (
    IOMessage,
    InitiateWorkflowModel,
    MessageProcessorMixin,
    TextInput,
    TextMessage,
)

if TYPE_CHECKING:
    from autogen.io.run_response import RunResponse


class WorkflowInfo(BaseModel):
    name: str
    description: str


# thread is used here in context of agent wire protocol thread, not python threading


class AWPThreadInfo:
    def __init__(self, run_agent_input: RunAgentInput, workflow_id: str) -> None:
        """Represent AWP thread.

        Args:
            run_agent_input (RunAgentInput): run agent input from the request
            workflow_id (str): The workflow id.
        """
        self.run_agent_input = run_agent_input
        self.awp_id = run_agent_input.thread_id
        self.run_id = run_agent_input.run_id
        self.workflow_id = workflow_id
        self.out_queue: Queue[BaseMessage] = Queue()
        self.input_queue: Queue[str] = Queue()
        self.active = True
        self.encoder = EventEncoder()
        self.run_response = None
        self.tools: list[AG2Tool] = []
        self.tool_names: list[str] = []
        self.ui_tool_announced: bool = False
        # all messages that have been attempted to send in one run
        self.sent_messages: list[BaseMessage] = []

    def has_text_input_widget(self) -> bool:
        return False

    def next_message_id(self) -> str:
        return str(uuid4().hex)


workflow_ids = threading.local()
workflow_ids.workflow_uuid = None


class AWPAdapter(MessageProcessorMixin, CreateWorkflowUIMixin):
    def __init__(
        self,
        provider: ProviderProtocol,
        *,
        discovery_path: str = "/fastagency/discovery",
        awp_path: str = "/fastagency/awp",
        wf_name: Optional[str] = None,
        get_user_id: Optional[Callable[..., Optional[str]]] = None,
        filter: Optional[Callable[[BaseMessage], bool]] = None,
    ) -> None:
        """Provider for AWP.

        Args:
            provider (ProviderProtocol): The provider.
            discovery_path (str, optional): The discovery path. Defaults to "/fastagency/discovery".
            awp_path (str, optional): The agent wire protocol path. Defaults to "/fastagency/awp".
            wf_name (str, optional): The name of the workflow to run Defaults to first workflow in adapter.
            get_user_id (Optional[Callable[[], Optional[UUID]]], optional): The get user id. Defaults to None.
            filter (Optional[Callable[[BaseMessage], bool]], optional): The filter   function. Defaults to None.
        """
        self.provider = provider
        self.discovery_path = discovery_path
        self.awp_path = awp_path
        self.get_user_id = get_user_id or (lambda: None)
        self._awp_threads: dict[str, AWPThreadInfo] = {}
        if wf_name is None:
            wf_name = self.provider.names[0]
        self.wf_name = wf_name
        self.router = self.setup_routes()
        self.filter = filter

    def create_awp_workflow_ui(
        self: UIBase, workflow_uuid: str, thread_info: AWPThreadInfo
    ) -> "UI":
        return AWPUI(uibase=self, workflow_uuid=workflow_uuid, thread_info=thread_info)

    def visit(self, message: IOMessage) -> Optional[str]:
        if self.filter and not self.filter(message):
            logger.info(f"Message filtered out: {message}")
            return None
        # call the super class visit method
        return super().visit(message)

    def get_thread_info_of_workflow(
        self, workflow_uuid: str
    ) -> Optional[AWPThreadInfo]:
        thread_info = next(
            (x for x in self._awp_threads.values() if x.workflow_id == workflow_uuid),
            None,
        )
        if thread_info is None:
            logger.error(
                f"Workflow {workflow_uuid} not found in threads: {self._awp_threads}"
            )
            raise RuntimeError(
                f"Workflow {workflow_uuid} not found in threads: {self._awp_threads}"
            )
        return thread_info

    def get_thread_info_of_awp(self, awp_id: str) -> Optional[AWPThreadInfo]:
        return self._awp_threads.get(awp_id)

    def send_to_thread(self, thread_id: str, message: str) -> None:
        thread_info = self._awp_threads.get(thread_id)
        if thread_info:
            if not thread_info.active:
                logger.error(f"Thread {thread_id} is not active")
                return
            thread_info.out_queue.put_nowait(message)
        else:
            logger.error(f"Thread {thread_id} not found")

    def end_of_thread(self, thread_id: str) -> None:
        thread_info = self._awp_threads.pop(thread_id, None)
        if thread_info:
            thread_info.active = False
            logger.info(f"Ended awp thread: {thread_info}")

    def create_tool_callback(
        self, tool_name: str, thread_info: AWPThreadInfo
    ) -> Callable[..., None]:
        def callback(args: Any) -> None:
            logger.info(f"----------------------Tool {tool_name} called with args: {args}")
            out_queue = thread_info.out_queue
            call_id = str(uuid4().hex)
            tool_call_start = ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_name=tool_name,
                tool_call_id=call_id,
            )
            out_queue.put_nowait(tool_call_start)

            delta = json.dumps(args)
            logger.info(f"Tool call args: |{delta}|")
            tool_call_args = ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=call_id,
                delta=delta,
            )
            out_queue.put_nowait(tool_call_args)

            tool_call_end = ToolCallEndEvent(
                type=EventType.TOOL_CALL_END, tool_call_id=call_id
            )
            out_queue.put_nowait(tool_call_end)
            
            run_finished = RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_info.awp_id,
                run_id=thread_info.run_id,
            )
            out_queue.put_nowait(run_finished)

        return callback
    

    def set_run_tools(self, tools: list[AG2Tool], thread_info: AWPThreadInfo) -> None:
        ui_tools: list[AG2Tool] = []
        tool_names :list[str] = []
        for tool in tools:
            logger.info(f"Setting up tool: {tool.name}")
            tool_names.append(tool.name)
            callback = self.create_tool_callback(tool.name, thread_info)

            try:
                ag2_tool = AG2Tool(
                    name=tool.name,
                    description=tool.description,
                    func_or_tool=callback,
                    parameters_json_schema=tool.parameters,
                )
                ui_tools.append(ag2_tool)
            except Exception as e:
                logger.error(f"Error creating tool: {e}")

        thread_info.tools = ui_tools
        thread_info.tool_names = tool_names

    async def run_thread(
        self, input: RunAgentInput, request: Request
    ) -> AsyncIterator[str]:
        thread_info = self._awp_threads.get(input.thread_id)
        if thread_info is None:
            logger.error(f"Thread {input.thread_id} not found")
            raise RuntimeError(f"Thread {input.thread_id} not found")

        self.set_run_tools(input.tools, thread_info)

        run_started = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_info.awp_id,
            run_id=thread_info.run_id,
        )
        yield self._sse_send(run_started, thread_info)

        while not await request.is_disconnected():
            try:
                message = await asyncio.wait_for(
                    thread_info.out_queue.get(), timeout=0.5
                )
                logger.info(
                    f"++++++++++Sending message in thread {input.thread_id}: {message}"
                )
                yield self._sse_send(message, thread_info)
                if isinstance(message, RunFinishedEvent):
                    break
                if isinstance(message, CustomEvent) and message.name == "thread_over":
                    run_finished = RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=thread_info.awp_id,
                        run_id=thread_info.run_id,
                    )
                    yield self._sse_send(run_finished, thread_info)
                    logger.info(f"Thread {input.thread_id} is over")
                    self.end_of_thread(input.thread_id)
                    break
            except asyncio.TimeoutError:
                await asyncio.sleep(
                    0
                )  # Yield control briefly, might not be strictly needed
                continue  # Go back to the top and check if request is still open

        logger.info(f"run thread {input.thread_id} completed")

    def _sse_send(self, message: BaseMessage, thread_info: AWPThreadInfo) -> str:
        thread_info.sent_messages.append(message)
        return str(thread_info.encoder.encode(message))

    def ducttape_request(self, data: dict[str, Any]) -> None:
        """Remove unwanted data from the request.

        This method is used to remove unwanted data from the request.
        Once agenwire packahge is fixed this can be removed
        """
        messages = data.get("messages", [])
        for message in messages:
            role = message.get("role", None)
            if role == "tool":
                # If role is 'tool', remove it from the message
                print("Removing 'name' from tool message:", message)
                del message["name"]

    def setup_routes(self) -> APIRouter:
        router = APIRouter()

        @router.post(self.awp_path)
        async def run_agent(
            # input: RunAgentInput,
            request: Request,
            user_id: Optional[str] = Depends(self.get_user_id),
        ) -> StreamingResponse:
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx: prevent buffering
            }

            # todo hacka round
            body_bytes = await request.body()
            body_str = body_bytes.decode()
            logger.debug(f"RAW REQUEST BODY: {body_str}")  # Or use logger

            try:
                # Manually try to parse and validate
                data = await request.json()
                self.ducttape_request(data)   #remove unwanted data from request
                input = RunAgentInput(**data)
                # If successful, do something with item_model
            except ValidationError as e:
                print("PYDANTIC VALIDATION FAILED:")
                print(e.errors())  # Detailed Pydantic errors
                print(e.json())  # JSON representation of errors
                # FastAPI would normally do this automatically:
                raise HTTPException(status_code=422, detail=e.errors())
            except Exception as e:
                print(f"OTHER PARSING ERROR: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid request body: {e}"
                )

            input = RunAgentInput(**await request.json())
            if input.thread_id in self._awp_threads:
                ## existing thread, resume
                logger.info(f"Resuming thread: {input.thread_id}")
                logger.info(f"Messages: {input.messages}")
                thread_info = self._awp_threads[input.thread_id]
                last_message = input.messages[-1]
                if isinstance(last_message, UserMessage):
                    thread_info.input_queue.put_nowait(last_message.content)
                return StreamingResponse(
                    self.run_thread(input, request), headers=headers
                )

            ## new thread, create
            workflow_uuid: str = uuid4().hex

            thread_info = AWPThreadInfo(input, workflow_id=workflow_uuid)
            self._awp_threads[input.thread_id] = thread_info
            logger.info(f"Created new thread: {input.thread_id}")

            init_msg = InitiateWorkflowModel(
                user_id=user_id,
                workflow_uuid=workflow_uuid,
                params={},
                name=self.wf_name,
            )

            async def process_messages_in_background(
                thread_info: AWPThreadInfo,
            ) -> None:
                def a_process_messages_in_background(
                    thread_info: AWPThreadInfo,
                ) -> None:
                    # store thread_info in thread local storage
                    logger.info(
                        f"Processing messages in background {threading.get_ident()}"
                    )
                    workflow_uuid = thread_info.workflow_id
                    workflow_ids.workflow_uuid = workflow_uuid
                    self.provider.run(
                        name=init_msg.name,
                        ui=self.create_awp_workflow_ui(workflow_uuid, thread_info),
                        user_id=user_id if user_id else "None",
                        **init_msg.params,
                    )

                await asyncify(a_process_messages_in_background)(thread_info)
                workflow_ids.workflow_uuid = None

            try:
                task = asyncio.create_task(
                    # process_messages_in_background(workflow_uuid)
                    process_messages_in_background(thread_info)
                )
                logger.info(f"Started task: {task}")
            except Exception as e:
                logger.error(f"Error in awp endpoint: {e}", stack_info=True)
            finally:
                ...
                # self.end_of_thread(request.thread_id)
            return StreamingResponse(self.run_thread(input, request), headers=headers)

        @router.get(
            self.discovery_path,
            responses={
                404: {"detail": "Key Not Found"},
                504: {"detail": "Unable to connect to provider"},
            },
        )
        def discovery(
            user_id: Optional[str] = Depends(self.get_user_id),
        ) -> list[WorkflowInfo]:
            try:
                names = self.provider.names
            except FastAgencyConnectionError as e:
                raise HTTPException(status_code=504, detail=str(e)) from e

            try:
                descriptions = [self.provider.get_description(name) for name in names]
            except FastAgencyKeyError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e

            return [
                WorkflowInfo(name=name, description=description)
                for name, description in zip(names, descriptions)
            ]

        return router

    def visit_default(self, message: IOMessage) -> Optional[str]:
        async def a_visit_default(
            self: AWPAdapter, message: IOMessage, workflow_uuid: str
        ) -> Optional[str]:
            logger.info(f"Default Visiting message: {message}")

            return None

        if isinstance(message, IOMessage):
            workflow_uuid = message.workflow_uuid
        else:
            logger.warning(
                f"Message is not an IOMessage, message type: {type(message)}"
            )
            workflow_uuid = workflow_ids.workflow_uuid

        return syncify(a_visit_default)(self, message, workflow_uuid)

    def visit_text_message(self, message: TextMessage) -> None:
        async def a_visit_text_message(self: AWPAdapter, message: TextMessage) -> None:
            workflow_uuid = message.workflow_uuid
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                return
            out_queue = thread_info.out_queue

            message_started = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message.uuid,
                role="assistant",
            )
            out_queue.put_nowait(message_started)

            message_content = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message.uuid,
                delta=message.body,
            )
            out_queue.put_nowait(message_content)

            message_end = TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=message.uuid
            )
            out_queue.put_nowait(message_end)

        syncify(a_visit_text_message)(self, message)

    def visit_text_input(self, message: TextInput) -> str:
        async def a_visit_text_input(self: AWPAdapter, message: TextInput) -> str:
            workflow_uuid = message.workflow_uuid
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                raise KeyError(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )

            if thread_info.ui_tool_announced:
                logger.info(
                    f"UI Tool auto allowed {thread_info.awp_id}"
                )
                thread_info.ui_tool_announced = False
                return ""
            
            out_queue = thread_info.out_queue

            message_started = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message.uuid,
                role="assistant",
            )
            out_queue.put_nowait(message_started)

            if message.prompt:
                prompt = message.prompt.replace(
                    "Press enter to skip and use auto-reply",
                    "Answer continue to skip and use auto-reply",
                )
            message_content = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message.uuid,
                delta=prompt,
            )
            out_queue.put_nowait(message_content)

            message_end = TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=message.uuid
            )
            out_queue.put_nowait(message_end)

            if thread_info.has_text_input_widget():
                # todo : invoke function to get an answer
                ...

            ## send end of run message, so that the UI can acquire answer and call us back
            run_finished = RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_info.awp_id,
                run_id=thread_info.run_id,
            )
            out_queue.put_nowait(run_finished)

            # wait for the answer to be sent back
            response = await thread_info.input_queue.get()
            if response == "continue":
                response = ""
            return response

        return syncify(a_visit_text_input)(self, message)

    # Non fastagency messages``

    def visit_text(self, message: autogen.messages.agent_messages.TextMessage) -> None:
        async def a_visit_text(
            self: AWPAdapter,
            message: autogen.messages.agent_messages.TextMessage,
            workflow_uuid: str,
        ) -> None:
            logger.info(f"Visiting text event: {message}")
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                return

            out_queue = thread_info.out_queue
            content = message.content
            uuid = str(content.uuid)
            if content.content:
                message_started = TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START, message_id=uuid, role="assistant"
                )
                out_queue.put_nowait(message_started)

                message_content = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=uuid,
                    delta=content.content,
                )
                out_queue.put_nowait(message_content)

                message_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END, message_id=uuid
                )
                out_queue.put_nowait(message_end)

        workflow_uuid = workflow_ids.workflow_uuid
        syncify(a_visit_text)(self, message, workflow_uuid)

    def visit_tool_call(
        self, message: autogen.messages.agent_messages.ToolCallMessage
    ) -> None:
        async def a_visit_tool_call(
            self: AWPAdapter,
            message: autogen.messages.agent_messages.ToolCallMessage,
            workflow_uuid: str,
        ) -> None:
            logger.info(f"Visiting tool call event: {message}")
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                return

            tool_name = message.content.tool_calls[0].function.name
            if tool_name in thread_info.tool_names:
                thread_info.ui_tool_announced = True
                logger.info(
                    f"Tool {tool_name} is announced in thread {thread_info.awp_id}"
                )
                return

            out_queue = thread_info.out_queue
            content = message.content
            uuid = str(content.uuid)
            message_started = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id=uuid, role="assistant"
            )
            out_queue.put_nowait(message_started)

            message_content = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=uuid,
                delta=f"AG2 wants to invoke tool: {tool_name}",
            )
            out_queue.put_nowait(message_content)

            message_end = TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=uuid
            )
            out_queue.put_nowait(message_end)

        workflow_uuid = workflow_ids.workflow_uuid
        syncify(a_visit_tool_call)(self, message, workflow_uuid)

    def visit_input_request(
        self, message: autogen.events.agent_events.InputRequestEvent
    ) -> None:
        async def a_visit_input_request(
            self: AWPAdapter,
            message: autogen.events.agent_events.InputRequestEvent,
            workflow_uuid: str,
        ) -> None:
            logger.info(f"Visiting input request: {message}")
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                raise KeyError(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )

            if thread_info.ui_tool_announced:
                logger.info(
                    f"UI Tool auto allowed {thread_info.awp_id}"
                )
                thread_info.ui_tool_announced = False
                message.content.respond("")
                return

            out_queue = thread_info.out_queue
            uuid = str(uuid4().hex)
            message_started = TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id=uuid, role="assistant"
            )
            out_queue.put_nowait(message_started)

            prompt = message.content.prompt.replace(
                "Press enter to skip and use auto-reply",
                "Answer continue to skip and use auto-reply",
            )

            message_content = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=uuid,
                delta=prompt,
            )
            out_queue.put_nowait(message_content)

            message_end = TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=uuid
            )
            out_queue.put_nowait(message_end)

            ## send end of run message, so that the UI can acquire answer and call us back

            run_finished = RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_info.awp_id,
                run_id=thread_info.run_id,
            )
            out_queue.put_nowait(run_finished)
            input_queue = thread_info.input_queue
            response = await input_queue.get()
            if response == "continue":
                response = ""
            message.content.respond(response)

        workflow_uuid = workflow_ids.workflow_uuid
        syncify(a_visit_input_request)(self, message, workflow_uuid)

    def visit_run_completion(
        self, message: autogen.events.agent_events.RunCompletionEvent
    ) -> None:
        async def a_visit_run_completion(
            self: AWPAdapter,
            message: autogen.events.agent_events.RunCompletionEvent,
            workflow_uuid: str,
        ) -> None:
            logger.info(f"Visiting run completion: {message}")
            thread_info = self.get_thread_info_of_workflow(workflow_uuid)
            if thread_info is None:
                logger.error(
                    f"Thread info not found for workflow {workflow_uuid}: {self._awp_threads}"
                )
                return
            out_queue = thread_info.out_queue

            thread_over = CustomEvent(
                type=EventType.CUSTOM, name="thread_over", value={}
            )
            out_queue.put_nowait(thread_over)

        workflow_uuid = workflow_ids.workflow_uuid
        return syncify(a_visit_run_completion)(self, message, workflow_uuid)

    def create_subconversation(self) -> UIBase:
        return self

    @contextmanager
    def create(self, app: Runnable, import_string: str) -> Iterator[None]:
        raise NotImplementedError("create")

    def start(
        self,
        *,
        app: "Runnable",
        import_string: str,
        name: Optional[str] = None,
        params: dict[str, Any],
        single_run: bool = False,
    ) -> None:
        raise NotImplementedError("start")

    @classmethod
    def create_provider(
        cls,
        fastapi_url: str,
    ) -> ProviderProtocol:
        raise NotImplementedError("create")


class AWPUI(UI):
    def __init__(
        self, uibase: UIBase, workflow_uuid: str, thread_info: AWPThreadInfo
    ) -> None:
        """Initialize the AWPUI.

        Args:
            uibase (UIBase): The UI base.
            workflow_uuid (str): The workflow UUID.
            thread_info (AWPThreadInfo): The thread info.
        """
        super().__init__(uibase=uibase, workflow_uuid=workflow_uuid)
        logger.info(f"Creating AWPUI {thread_info}")
        self.thread_info = thread_info

    def process(self, response: "RunResponse") -> str:
        """Process the response from the workflow.

        This method processes the events in the response and waits for the
        summary to be ready.
        """
        logger.info(
            f"Calling process method in thread {threading.current_thread().name} , {threading.get_ident()}"
        )
        thread_info = self.thread_info
        thread_info.run_response = response
        try:
            logger.info("Setting up the tools")
            response.set_ui_tools(thread_info.tools)
        except:
            logger.error(f"Error setting up the tools: {thread_info.tools}")
            raise
        for event in response.events:
            self.process_message(event)
        # remove response from the thread variables
        thread_info.run_response = None

        return str(response.summary)


logger = get_logger(__name__)
