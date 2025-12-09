from unittest.mock import MagicMock, patch

from autogen import UserProxyAgent

from fastagency.api.openapi import OpenAPI
from fastagency.api.openapi.security import APIKeyHeader


@patch("fastagency.api.openapi.openapi.requests.post")
def test_real_whatsapp_end2end(
    mock_post: MagicMock,
) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    api = OpenAPI.create(
        openapi_url="https://dev.infobip.com/openapi/products/whatsapp.json",
        servers=[{"url": "https://api.infobip.com"}],
    )

    assert isinstance(api, OpenAPI)

    header_authorization = "App something"  # pragma: allowlist secret
    api.set_security_params(APIKeyHeader.Parameters(value=header_authorization))

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    functions = ["send_whatsapp_text_message"]
    api._register_for_execution(user_proxy, functions=functions)

    assert tuple(user_proxy._function_map.keys()) == ("send_whatsapp_text_message",)

    send_whatsapp_text_message = user_proxy._function_map["send_whatsapp_text_message"]

    send_whatsapp_text_message(
        **{
            "body": {
                "from": "447860099299",
                "to": "38591152131",
                "messageId": "test-message-123",
                "content": {"text": "Hello, World!"},
                "callbackData": "Callback data",
            }
        }
    )

    mock_post.assert_called_once_with(
        "https://api.infobip.com/whatsapp/1/message/text",
        params={},
        json={
            "from_": "447860099299",
            "to": "38591152131",
            "messageId": "test-message-123",
            "content": {"text": "Hello, World!", "previewUrl": False},
            "callbackData": "Callback data",
            "notifyUrl": None,
            "urlOptions": None,
            "entityId": None,
            "applicationId": None,
            "context": None,
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": header_authorization,
        },
    )
