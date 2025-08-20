import sys

import bcrypt
import pytest

from fastagency.logging import get_logger

logger = get_logger(__name__)

if sys.version_info >= (3, 10):
    from fastagency.ui.mesop.auth.basic_auth import BasicAuth

    class TestBasicAuthMesopHomePage:
        @pytest.fixture
        def basic_auth(self) -> BasicAuth:
            """Fixture to create a BasicAuth instance with test users."""
            # Create test users with hashed passwords
            test_password = "test_password"  # pragma: allowlist secret
            hashed_password = bcrypt.hashpw(
                test_password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            allowed_users = {
                "test_user": hashed_password,
                "another_user": hashed_password,
            }

            return BasicAuth(allowed_users)

        def test_is_authorized_valid_credentials(self, basic_auth: BasicAuth) -> None:
            """Test authorization with valid username and password."""
            assert basic_auth.is_authorized("test_user", "test_password") is True

        def test_is_authorized_invalid_username(self, basic_auth: BasicAuth) -> None:
            """Test authorization with invalid username."""
            assert basic_auth.is_authorized("invalid_user", "test_password") is False
