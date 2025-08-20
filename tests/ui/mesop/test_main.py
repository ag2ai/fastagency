import sys
from typing import Any, Callable

import pytest

from fastagency.logging import get_logger

logger = get_logger(__name__)


if sys.version_info >= (3, 10):
    import mesop as me

    from fastagency.ui.mesop import MesopUI
    from fastagency.ui.mesop.auth.firebase import FirebaseAuth, FirebaseConfig
    from fastagency.ui.mesop.main import DEFAULT_SECURITY_POLICY, MesopHomePage

    class TestMesopHomePage:
        @pytest.fixture
        def firebase_auth(self, monkeypatch: pytest.MonkeyPatch) -> FirebaseAuth:
            # Ensure required environment variables are set mocked
            monkeypatch.setenv(
                "GOOGLE_APPLICATION_CREDENTIALS", "/path/to/credentials.json"
            )

            config = FirebaseConfig(
                api_key="test-key",
                auth_domain="test.firebaseapp.com",
                project_id="test-project",
                storage_bucket="test-bucket",
                messaging_sender_id="test-sender",
                app_id="test-app",
            )
            return FirebaseAuth(
                sign_in_methods=["google"], config=config, allowed_users="all"
            )

        @pytest.fixture
        def mesop_ui(self) -> MesopUI:
            return MesopUI()

        @pytest.fixture
        def base_policy(self) -> me.SecurityPolicy:
            return me.SecurityPolicy(
                dangerously_disable_trusted_types=False,
                allowed_connect_srcs=["custom.domain.com"],
                allowed_script_srcs=["custom.script.com"],
            )

        class TestWithoutAuth:
            """Testing without auth provider, and with and without base security policy."""

            def test_with_security_policy(
                self, mesop_ui: MesopUI, base_policy: me.SecurityPolicy
            ) -> None:
                """Test scenario: No auth provider, base security policy provided.

                Expected: Should use the base security policy as-is.
                """
                homepage = MesopHomePage(
                    ui=mesop_ui,
                    security_policy=base_policy,
                )

                assert homepage._security_policy == base_policy
                assert homepage._security_policy.allowed_connect_srcs == [
                    "custom.domain.com"
                ]
                assert homepage._security_policy.allowed_script_srcs == [
                    "custom.script.com"
                ]

            def test_without_security_policy(self, mesop_ui: MesopUI) -> None:
                """Test scenario: No auth provider, no security policy provided.

                Expected: Should use the default security policy.
                """
                homepage = MesopHomePage(ui=mesop_ui)

                assert homepage._security_policy == DEFAULT_SECURITY_POLICY

        class TestWithAuth:
            """Testing with auth provider, and with and without base security policy."""

            def test_firebase_auth_without_credentials(
                self, monkeypatch: pytest.MonkeyPatch
            ) -> None:
                """Test scenario: FirebaseAuth initialization without GOOGLE_APPLICATION_CREDENTIALS env variable.

                Expected: Should raise EnvironmentError when GOOGLE_APPLICATION_CREDENTIALS is not set.
                """
                # Remove GOOGLE_APPLICATION_CREDENTIALS from environment if it exists
                monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

                config = FirebaseConfig(
                    api_key="test-key",
                    auth_domain="test.firebaseapp.com",
                    project_id="test-project",
                    storage_bucket="test-bucket",
                    messaging_sender_id="test-sender",
                    app_id="test-app",
                )
                with pytest.raises(
                    ValueError, match="A service account key is required."
                ):
                    FirebaseAuth(
                        sign_in_methods=["google"], config=config, allowed_users="all"
                    )

            def test_with_security_policy(
                self,
                mesop_ui: MesopUI,
                firebase_auth: FirebaseAuth,
                base_policy: me.SecurityPolicy,
            ) -> None:
                """Test scenario: Auth provider present, custom security policy provided.

                Expected: Should merge custom policy with Firebase requirements.
                """
                homepage = MesopHomePage(
                    ui=mesop_ui, security_policy=base_policy, auth=firebase_auth
                )

                assert set(homepage._security_policy.allowed_connect_srcs) == {
                    "*.googleapis.com",
                    "custom.domain.com",
                }
                assert set(homepage._security_policy.allowed_script_srcs) == {
                    "*.google.com",
                    "https://www.gstatic.com",
                    "https://cdn.jsdelivr.net",
                    "custom.script.com",
                }

            def test_without_security_policy(
                self, mesop_ui: MesopUI, firebase_auth: FirebaseAuth
            ) -> None:
                """Test scenario: Auth provider present, no security policy provided.

                Expected: Should use default policy merged with Firebase requirements.
                """
                homepage = MesopHomePage(ui=mesop_ui, auth=firebase_auth)

                assert set(homepage._security_policy.allowed_connect_srcs) == {
                    "*.googleapis.com"
                }
                assert set(homepage._security_policy.allowed_script_srcs) == {
                    "*.google.com",
                    "https://www.gstatic.com",
                    "https://cdn.jsdelivr.net",
                }

    class TestFirebaseAuth:
        """Test cases for Firebase authentication authorization checks."""

        INVALID_FIREBASE_TOKEN_ERROR_MSG = (
            "Invalid response from Firebase: `email` field is missing in the token"
        )

        @pytest.fixture
        def valid_token(self) -> dict[str, Any]:
            """Fixture for a valid token with email."""
            return {"email": "user@example.com", "other_field": "value"}

        @pytest.fixture
        def firebase_config(self) -> FirebaseConfig:
            """Fixture for Firebase configuration."""
            return FirebaseConfig(
                api_key="test-key",
                auth_domain="test.firebaseapp.com",
                project_id="test-project",
                storage_bucket="test-bucket",
                messaging_sender_id="test-sender",
                app_id="test-app",
            )

        @pytest.fixture
        def auth_factory(
            self, firebase_config: FirebaseConfig, monkeypatch: pytest.MonkeyPatch
        ) -> Callable[[Any], FirebaseAuth]:
            """Fixture for creating FirebaseAuth instances."""

            def _create_auth(allowed_users: Any) -> FirebaseAuth:
                # Ensure required environment variables are set mocked
                monkeypatch.setenv(
                    "GOOGLE_APPLICATION_CREDENTIALS", "/path/to/credentials.json"
                )
                return FirebaseAuth(
                    sign_in_methods=["google"],
                    config=firebase_config,
                    allowed_users=allowed_users,
                )

            return _create_auth

        def test_token_validation(
            self, auth_factory: Callable[[Any], FirebaseAuth]
        ) -> None:
            """Test token validation for missing email cases."""
            auth = auth_factory("all")

            # Test empty token
            with pytest.raises(
                ValueError, match=TestFirebaseAuth.INVALID_FIREBASE_TOKEN_ERROR_MSG
            ):
                auth.is_authorized({})

            # Test missing email field
            with pytest.raises(
                ValueError, match=TestFirebaseAuth.INVALID_FIREBASE_TOKEN_ERROR_MSG
            ):
                auth.is_authorized({"other_field": "value"})

        def test_all_access(
            self,
            auth_factory: Callable[[Any], FirebaseAuth],
            valid_token: dict[str, Any],
        ) -> None:
            """Test 'all' access configuration."""
            auth = auth_factory("all")
            assert auth.is_authorized(valid_token) is True

        def test_single_email_access(
            self,
            auth_factory: Callable[[Any], FirebaseAuth],
            valid_token: dict[str, Any],
        ) -> None:
            """Test single email access configuration."""
            # Test exact match
            auth = auth_factory("user@example.com")
            assert auth.is_authorized(valid_token) is True

            # Test email mismatch
            auth = auth_factory("other@example.com")
            assert auth.is_authorized(valid_token) is False

            # Test empty allowed email
            auth = auth_factory("")
            assert auth.is_authorized(valid_token) is False

        def test_email_list_access(
            self,
            auth_factory: Callable[[Any], FirebaseAuth],
            valid_token: dict[str, Any],
        ) -> None:
            """Test email list access configuration."""
            # Test email in list
            auth = auth_factory(
                [
                    "other@example.com",
                    "user@example.com",
                    "another@example.com",
                ]
            )
            assert auth.is_authorized(valid_token) is True

            # Test email not in list
            auth = auth_factory(["other@example.com", "another@example.com"])
            assert auth.is_authorized(valid_token) is False

            # Test empty list
            auth = auth_factory([])
            assert auth.is_authorized(valid_token) is False

            # Test list with empty values
            auth = auth_factory(["", None, "user@example.com", "  "])
            assert auth.is_authorized(valid_token) is True

        def test_callable_access(
            self,
            auth_factory: Callable[[Any], FirebaseAuth],
            valid_token: dict[str, Any],
        ) -> None:
            """Test callable access configuration."""
            # Test callable returns True
            auth = auth_factory(lambda token: True)
            assert auth.is_authorized(valid_token) is True

            # Test callable returns False
            auth = auth_factory(lambda token: False)
            assert auth.is_authorized(valid_token) is False

            # Test callable raises exception
            def raise_error(token: dict[str, Any]) -> bool:
                raise ValueError("Custom validation error")

            auth = auth_factory(raise_error)
            with pytest.raises(ValueError, match="Custom validation error"):
                auth.is_authorized(valid_token)

        @pytest.mark.parametrize(
            ("test_input", "expected_error", "error_match"),
            [
                (None, TypeError, "allowed_users must be one of"),
                (123, TypeError, "allowed_users must be one of"),
                ({}, TypeError, "allowed_users must be one of"),
                (set(), TypeError, "allowed_users must be one of"),
            ],
        )
        def test_invalid_allowed_users(
            self,
            auth_factory: Callable[[Any], FirebaseAuth],
            valid_token: dict[str, Any],
            test_input: Any,
            expected_error: type[Exception],
            error_match: str,
        ) -> None:
            """Test invalid allowed_users configurations."""
            auth = auth_factory(test_input)
            with pytest.raises(expected_error, match=error_match):
                auth.is_authorized(valid_token)
