import asyncio
import threading
from asyncio import Queue
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, Optional
from uuid import uuid4

import autogen
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
    UserMessage,
)
from agentwire.encoder import EventEncoder
from asyncer import asyncify, syncify
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastagency.logging import get_logger

from ...base import (
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
        get_user_id: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        """Provider for AWP.

        Args:
            provider (ProviderProtocol): The provider.
            discovery_path (str, optional): The discovery path. Defaults to "/fastagency/discovery".
            awp_path (str, optional): The agent wire protocol path. Defaults to "/fastagency/awp".
            get_user_id (Optional[Callable[[], Optional[UUID]]], optional): The get user id. Defaults to None.
        """
        self.provider = provider
        self.discovery_path = discovery_path
        self.awp_path = awp_path
        self.get_user_id = get_user_id or (lambda: None)
        self._awp_threads: dict[str, AWPThreadInfo] = {}
        self.router = self.setup_routes()

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

    async def run_thread(
        self, input: RunAgentInput, request: Request
    ) -> AsyncIterator[str]:
        thread_info = self._awp_threads.get(input.thread_id)
        if thread_info is None:
            logger.error(f"Thread {input.thread_id} not found")
            raise RuntimeError(f"Thread {input.thread_id} not found")

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

    def setup_routes(self) -> APIRouter:
        router = APIRouter()

        @router.post(self.awp_path)
        async def run_agent(
            input: RunAgentInput,
            request: Request,
            user_id: Optional[str] = Depends(self.get_user_id),
        ) -> StreamingResponse:
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx: prevent buffering
            }

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
                name="simple_learning",
            )

            async def process_messages_in_background(workflow_uuid: str) -> None:
                def a_process_messages_in_background(
                    workflow_uuid: str,
                ) -> None:
                    workflow_ids.workflow_uuid = workflow_uuid
                    self.provider.run(
                        name=init_msg.name,
                        ui=self.create_workflow_ui(workflow_uuid),
                        user_id=user_id if user_id else "None",
                        **init_msg.params,
                    )

                await asyncify(a_process_messages_in_background)(workflow_uuid)
                workflow_ids.workflow_uuid = None

            try:
                task = asyncio.create_task(
                    process_messages_in_background(workflow_uuid)
                )
                logger.info(f"Started task: {task}")
                # asyncio.create_task(
                #    asyncify(process_messages_in_background)(workflow_uuid)
                # )
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
            logger.error(f"Dang! Message is not an IOMessage: {message}")
            logger.error(f"Message type: {type(message)}")
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
                delta=message.prompt,
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
            return await thread_info.input_queue.get()

        return syncify(a_visit_text_input)(self, message)

    # Non fastagency messages``

    def visit_text(self, message: autogen.events.agent_events.TextEvent) -> None:
        async def a_visit_text(
            self: AWPAdapter,
            message: autogen.events.agent_events.InputRequestEvent,
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

            out_queue = thread_info.out_queue
            ## send end of run message, so that the UI can acquire answer and call us back
            run_finished = RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_info.awp_id,
                run_id=thread_info.run_id,
            )
            out_queue.put_nowait(run_finished)
            input_queue = thread_info.input_queue
            message.content.respond(await input_queue.get())

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


logger = get_logger(__name__)
