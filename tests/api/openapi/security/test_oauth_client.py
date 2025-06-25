import json
from typing import Annotated, Any

import pytest
import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer as FastAPIOAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm

from docs.docs_src.user_guide.external_rest_apis.security_examples import (
    configure_oauth_client,
    configure_oauth_client_token,
)


def create_oauth2_fastapi_app(host: str, port: int) -> FastAPI:
    app = FastAPI(
        title="OAuth2",
        servers=[
            {"url": f"http://{host}:{port}", "description": "Local development server"}
        ],
    )

    oauth2_scheme = FastAPIOAuth2PasswordBearer(tokenUrl="token")

    @app.post("/token")
    async def login(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> dict[str, str]:
        if (
            form_data.username != "user"
            or form_data.password != "password"  # pragma: allowlist secret
        ):
            raise HTTPException(
                status_code=400, detail="Incorrect username or password"
            )
        return {"access_token": "token123", "token_type": "bearer"}

    @app.post("/low", summary="Low Level")
    async def post_oauth(
        message: str, token: Annotated[str, Depends(oauth2_scheme)]
    ) -> dict[str, str]:
        if token != "token123":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return {"message": message}

    return app


@pytest.fixture
def openapi_oauth2_schema() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "OAuth2", "version": "0.1.0"},
        "servers": [
            {"url": "http://127.0.0.1:60473", "description": "Local development server"}
        ],
        "paths": {
            "/token": {
                "post": {
                    "summary": "Login",
                    "operationId": "login_token_post",
                    "requestBody": {
                        "content": {
                            "application/x-www-form-urlencoded": {
                                "schema": {
                                    "$ref": "#/components/schemas/Body_login_token_post"
                                }
                            }
                        },
                        "required": True,
                    },
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "additionalProperties": {"type": "string"},
                                        "type": "object",
                                        "title": "Response Login Token Post",
                                    }
                                }
                            },
                        },
                        "422": {
                            "description": "Validation Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HTTPValidationError"
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/low": {
                "post": {
                    "summary": "Low Level",
                    "operationId": "post_oauth_low_post",
                    "security": [{"OAuth2PasswordBearer": []}],
                    "parameters": [
                        {
                            "name": "message",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "title": "Message"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "additionalProperties": {"type": "string"},
                                        "title": "Response Post Oauth Low Post",
                                    }
                                }
                            },
                        },
                        "422": {
                            "description": "Validation Error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HTTPValidationError"
                                    }
                                }
                            },
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Body_login_token_post": {
                    "properties": {
                        "grant_type": {
                            "anyOf": [
                                {"type": "string", "pattern": "^password$"},
                                {"type": "null"},
                            ],
                            "title": "Grant Type",
                        },
                        "username": {"type": "string", "title": "Username"},
                        "password": {
                            "type": "string",
                            "title": "Password",
                            "format": "password",
                        },
                        "scope": {"type": "string", "title": "Scope", "default": ""},
                        "client_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Client Id",
                        },
                        "client_secret": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Client Secret",
                            "format": "password",
                        },
                    },
                    "type": "object",
                    "required": ["username", "password"],
                    "title": "Body_login_token_post",
                },
                "HTTPValidationError": {
                    "properties": {
                        "detail": {
                            "items": {"$ref": "#/components/schemas/ValidationError"},
                            "type": "array",
                            "title": "Detail",
                        }
                    },
                    "type": "object",
                    "title": "HTTPValidationError",
                },
                "ValidationError": {
                    "properties": {
                        "loc": {
                            "items": {
                                "anyOf": [{"type": "string"}, {"type": "integer"}]
                            },
                            "type": "array",
                            "title": "Location",
                        },
                        "msg": {"type": "string", "title": "Message"},
                        "type": {"type": "string", "title": "Error Type"},
                    },
                    "type": "object",
                    "required": ["loc", "msg", "type"],
                    "title": "ValidationError",
                },
            },
            "securitySchemes": {
                "OAuth2PasswordBearer": {
                    "type": "oauth2",
                    "flows": {"password": {"scopes": {}, "tokenUrl": "token"}},
                }
            },
        },
    }


@pytest.mark.parametrize(
    "fastapi_openapi_url",
    [(create_oauth2_fastapi_app)],
    indirect=["fastapi_openapi_url"],
)
def test_oauth2_fastapi_app(
    fastapi_openapi_url: str, openapi_oauth2_schema: dict[str, Any]
) -> None:
    with requests.get(fastapi_openapi_url, timeout=10) as response:
        response.raise_for_status()
        openapi_json = json.loads(response.text)

    openapi_oauth2_schema.pop("servers")
    servers = openapi_json.pop("servers")

    assert servers[0]["url"] == fastapi_openapi_url.split("/openapi.json")[0]
    assert openapi_oauth2_schema == openapi_json


@pytest.mark.parametrize(
    "fastapi_openapi_url",
    [(create_oauth2_fastapi_app)],
    indirect=["fastapi_openapi_url"],
)
def test_generate_oauth2_client_token(fastapi_openapi_url: str) -> None:
    api_client = configure_oauth_client_token(fastapi_openapi_url, "token123")

    expected = ["post_oauth_low_post", "login_token_post"]

    functions = list(api_client._get_functions_to_register())
    assert [f.__name__ for f in functions] == expected

    post_oauth_f = functions[0]

    response = post_oauth_f(message="message")

    assert response == {"message": "message"}


@pytest.mark.parametrize(
    "fastapi_openapi_url",
    [(create_oauth2_fastapi_app)],
    indirect=["fastapi_openapi_url"],
)
def test_generate_oauth2_client_password(fastapi_openapi_url: str) -> None:
    api_client = configure_oauth_client(fastapi_openapi_url, "user", "password")

    expected = ["post_oauth_low_post", "login_token_post"]

    functions = list(api_client._get_functions_to_register())
    assert [f.__name__ for f in functions] == expected

    post_oauth_f = functions[0]

    response = post_oauth_f(message="message")

    assert response == {"message": "message"}


@pytest.mark.parametrize(
    "fastapi_openapi_url",
    [(create_oauth2_fastapi_app)],
    indirect=["fastapi_openapi_url"],
)
def test_generate_oauth2_client_wrong_password(fastapi_openapi_url: str) -> None:
    api_client = configure_oauth_client(fastapi_openapi_url, "user", "wrong_password")
    expected = ["post_oauth_low_post", "login_token_post"]

    functions = list(api_client._get_functions_to_register())
    assert [f.__name__ for f in functions] == expected

    post_oauth_f = functions[0]

    with pytest.raises(requests.exceptions.HTTPError) as e:
        post_oauth_f(message="message")

    assert (
        str(e.value)
        == f"400 Client Error: Bad Request for url: {fastapi_openapi_url.split('/openapi.json')[0]}/token"
    )
