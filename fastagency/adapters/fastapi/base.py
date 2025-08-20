## IONats part

import asyncio
import json
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

import requests
import websockets
from asyncer import asyncify, syncify
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from fastapi.dependencies.utils import get_dependant, solve_dependencies
from pydantic import BaseModel

from fastagency.logging import get_logger

from ...base import (
    UI,
    CreateWorkflowUIMixin,
    ProviderProtocol,
    Runnable,
    UIBase,
    WorkflowsProtocol,
)
from ...exceptions import (
    FastAgencyConnectionError,
    FastAgencyFastAPIConnectionError,
    FastAgencyKeyError,
)
from ...messages import (
    AskingMessage,
    IOMessage,
    InitiateWorkflowModel,
    InputResponseModel,
    MessageProcessorMixin,
)

logger = get_logger(__name__)


class InititateChatModel(BaseModel):
    workflow_name: str
    workflow_uuid: str
    user_id: Optional[str]
    params: dict[str, Any]


class WorkflowInfo(BaseModel):
    name: str
    description: str


class FastAPIAdapter(MessageProcessorMixin, CreateWorkflowUIMixin):
    def __init__(
        self,
        provider: ProviderProtocol,
        *,
        initiate_workflow_path: str = "/fastagency/initiate_workflow",
        discovery_path: str = "/fastagency/discovery",
        ws_path: str = "/fastagency/ws",
        get_user_id: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        """Provider for FastAPI.

        Args:
            provider (ProviderProtocol): The provider.
            initiate_workflow_path (str, optional): The initiate workflow path. Defaults to "/fastagency/initiate_workflow".
            discovery_path (str, optional): The discovery path. Defaults to "/fastagency/discovery".
            ws_path (str, optional): The websocket path. Defaults to "/fastagency/ws".
            get_user_id (Optional[Callable[[], Optional[UUID]]], optional): The get user id. Defaults to None.
        """
        self.provider = provider

        self.initiate_workflow_path = initiate_workflow_path
        self.discovery_path = discovery_path
        self.ws_path = ws_path

        self.get_user_id = get_user_id or (lambda: None)

        self.websockets: dict[str, WebSocket] = {}

        self.router = self.setup_routes()

    async def get_user_id_websocket(self, websocket: WebSocket) -> Optional[str]:
        def get_user_id_depends_stub(
            user_id: Optional[str] = Depends(self.get_user_id),
        ) -> Optional[str]:
            raise RuntimeError(
                "Stub get_user_id_depends_stub called"
            )  # pragma: no cover

        dependant = get_dependant(path="", call=get_user_id_depends_stub)

        try:
            async with AsyncExitStack() as cm:
                scope = websocket.scope
                scope["type"] = "http"

                solved_dependency = await solve_dependencies(
                    dependant=dependant,
                    request=Request(scope=scope),  # Inject the request here
                    body=None,
                    dependency_overrides_provider=None,
                    async_exit_stack=cm,
                    embed_body_fields=False,
                )
        except HTTPException as e:
            raise e
        finally:
            scope["type"] = "websocket"

        return solved_dependency.values["user_id"]  # type: ignore[no-any-return]

    def setup_routes(self) -> APIRouter:
        router = APIRouter()

        @router.post(self.initiate_workflow_path)
        async def initiate_chat(
            initiate_chat: InititateChatModel,
            user_id: Optional[str] = Depends(self.get_user_id),
        ) -> InitiateWorkflowModel:
            workflow_uuid: UUID = uuid4()

            init_msg = InitiateWorkflowModel(
                user_id=user_id,
                workflow_uuid=workflow_uuid,
                params=initiate_chat.params,
                name=initiate_chat.workflow_name,
            )

            return init_msg

        @router.websocket(self.ws_path)
        async def websocket_endpoint(
            websocket: WebSocket,
        ) -> None:
            try:
                user_id = await self.get_user_id_websocket(websocket)
            except HTTPException as e:
                headers = getattr(e, "headers", None)
                await websocket.send_denial_response(
                    Response(status_code=e.status_code, headers=headers)
                )
                return

            logger.info("Websocket connected")
            await websocket.accept()
            logger.info("Websocket accepted")

            init_msg_json = await websocket.receive_text()
            logger.info(f"Received message: {init_msg_json}")

            init_msg = InitiateWorkflowModel.model_validate_json(init_msg_json)

            workflow_uuid = init_msg.workflow_uuid.hex
            self.websockets[workflow_uuid] = websocket

            try:
                await asyncify(self.provider.run)(
                    name=init_msg.name,
                    ui=self.create_workflow_ui(workflow_uuid),
                    user_id=user_id if user_id else "None",
                    **init_msg.params,
                )
            except Exception as e:
                logger.error(f"Error in websocket_endpoint: {e}", stack_info=True)
            finally:
                self.websockets.pop(workflow_uuid)

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
                for name, description in zip(names, descriptions, strict=False)
            ]

        return router

    def visit_default(self, message: IOMessage) -> Optional[str]:
        async def a_visit_default(
            self: FastAPIAdapter, message: IOMessage
        ) -> Optional[str]:
            workflow_uuid = message.workflow_uuid
            if workflow_uuid not in self.websockets:
                logger.error(
                    f"Workflow {workflow_uuid} not found in websockets: {self.websockets}"
                )
                raise RuntimeError(
                    f"Workflow {workflow_uuid} not found in websockets: {self.websockets}"
                )
            websocket = self.websockets[workflow_uuid]  # type: ignore[index]
            await websocket.send_text(json.dumps(message.model_dump()))

            if isinstance(message, AskingMessage):
                response = await websocket.receive_text()
                return response
            return None

        return syncify(a_visit_default)(self, message)

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
        return FastAPIProvider(
            fastapi_url=fastapi_url,
        )


class FastAPIProvider(ProviderProtocol):
    def __init__(
        self,
        fastapi_url: str,
        initiate_workflow_path: str = "/fastagency/initiate_workflow",
        discovery_path: str = "/fastagency/discovery",
        ws_path: str = "/fastagency/ws",
    ) -> None:
        """Initialize the fastapi workflows."""
        self._workflows: dict[
            str, tuple[Callable[[WorkflowsProtocol, UIBase, str, str], str], str]
        ] = {}

        self.fastapi_url = (
            fastapi_url[:-1] if fastapi_url.endswith("/") else fastapi_url
        )
        self.ws_url = "ws" + self.fastapi_url[4:]

        self.is_broker_running: bool = False

        self.initiate_workflow_path = initiate_workflow_path
        self.discovery_path = discovery_path
        self.ws_path = ws_path

    def _send_initiate_chat_msg(
        self,
        workflow_name: str,
        workflow_uuid: str,
        user_id: Optional[str],
        params: dict[str, Any],
    ) -> InitiateWorkflowModel:
        msg = InititateChatModel(
            workflow_name=workflow_name,
            workflow_uuid=workflow_uuid,
            user_id=user_id,
            params=params,
        )

        payload = msg.model_dump()

        resp = requests.post(
            f"{self.fastapi_url}{self.initiate_workflow_path}", json=payload, timeout=5
        )
        logger.info(f"Initiate chat response: {resp.json()}")
        retval = InitiateWorkflowModel(**resp.json())
        return retval

    async def _publish_websocket_message(
        self,
        websocket: websockets.WebSocketClientProtocol,
        message: InputResponseModel,
    ) -> None:
        payload = message.model_dump_json()
        await websocket.send(payload)
        logger.info(f"Message sent to websocket ({websocket}): {message}")

    async def _run_websocket_subscriber(
        self,
        ui: UI,
        workflow_name: str,
        user_id: Optional[str],
        from_server_subject: str,
        to_server_subject: str,
        params: dict[str, Any],
    ) -> None:
        connect_url = f"{self.ws_url}{self.ws_path}"
        async with websockets.connect(connect_url) as websocket:
            init_workflow_msg = InitiateWorkflowModel(
                name=workflow_name,
                workflow_uuid=ui._workflow_uuid,
                user_id=user_id,
                params=params,
            )
            await websocket.send(init_workflow_msg.model_dump_json())

            while True:
                response = await websocket.recv()
                response = (
                    response.decode() if isinstance(response, bytes) else response
                )

                logger.info(f"Received message: {response}")

                msg = IOMessage.create(**json.loads(response))

                retval = await asyncify(ui.process_message)(msg)
                logger.info(f"Message {msg}: processed with response {retval}")

                if isinstance(msg, AskingMessage):
                    if retval is None:
                        logger.warning(
                            f"Message {msg}: response is None. Skipping response to websocket"
                        )
                    else:
                        await websocket.send(retval)
                        logger.info(
                            f"Message {msg}: response {retval} sent to websocket"
                        )

    def run(
        self,
        name: str,
        ui: UI,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        workflow_uuid = ui._workflow_uuid

        initiate_workflow = self._send_initiate_chat_msg(
            name, workflow_uuid=workflow_uuid, user_id=user_id, params=kwargs
        )
        user_id = initiate_workflow.user_id if initiate_workflow.user_id else "None"
        workflow_uuid = initiate_workflow.workflow_uuid.hex

        _from_server_subject = f"chat.client.messages.{user_id}.{workflow_uuid}"
        _to_server_subject = f"chat.server.messages.{user_id}.{workflow_uuid}"

        async def _setup_and_run() -> None:
            await self._run_websocket_subscriber(
                ui,
                name,
                user_id,
                _from_server_subject,
                _to_server_subject,
                kwargs,
            )

        async def run_lifespan() -> None:
            if not self.is_broker_running:
                self.is_broker_running = True
                await _setup_and_run()
            else:
                await _setup_and_run()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(run_lifespan())

        return "FastAPIWorkflows.run() completed"

    def _get_workflow_info(self) -> list[dict[str, str]]:
        try:
            resp = requests.get(f"{self.fastapi_url}/{self.discovery_path}", timeout=15)
        except requests.exceptions.ConnectionError as e:
            raise FastAgencyFastAPIConnectionError(
                f"Unable to connect to FastAPI server at {self.fastapi_url}"
            ) from e
        if resp.status_code == 504:
            raise FastAgencyConnectionError(resp.json()["detail"])
        elif resp.status_code == 404:
            raise FastAgencyKeyError(resp.json()["detail"])
        return resp.json()  # type: ignore [no-any-return]

    def _get_names(self) -> list[str]:
        return [workflow["name"] for workflow in self._get_workflow_info()]

    def _get_description(self, name: str) -> str:
        return next(
            workflow["description"]
            for workflow in self._get_workflow_info()
            if workflow["name"] == name
        )

    @property
    def names(self) -> list[str]:
        return self._get_names()

    def get_description(self, name: str) -> str:
        return self._get_description(name)
