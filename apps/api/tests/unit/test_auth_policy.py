from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
import pytest
from evidencedesk_api.auth import Action, Role, UserPrincipal, role_allows
from evidencedesk_api.security import create_access_token, decode_access_token


def test_role_matrix_is_enforced_by_the_domain_policy() -> None:
    assert role_allows(Role.ADMIN, Action.DELETE_DOCUMENT)
    assert role_allows(Role.ADMIN, Action.VIEW_AUDIT)
    assert role_allows(Role.ANALYST, Action.UPLOAD_DOCUMENT)
    assert role_allows(Role.ANALYST, Action.ASK_QUESTION)
    assert not role_allows(Role.ANALYST, Action.DELETE_DOCUMENT)
    assert not role_allows(Role.READER, Action.UPLOAD_DOCUMENT)
    assert role_allows(Role.READER, Action.ASK_QUESTION)


def test_access_token_round_trip_and_expiration() -> None:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    user = UserPrincipal(
        id=UUID("d8a1e09b-2cb0-4e3f-8cc6-7e175cb64a11"),
        username="demo.reader",
        role=Role.READER,
    )
    secret = "test-secret-that-is-long-enough-123456"

    token = create_access_token(user, secret=secret, now=now, ttl=timedelta(minutes=30))

    assert decode_access_token(token, secret=secret, now=now + timedelta(minutes=29)) == user
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, secret=secret, now=now + timedelta(minutes=31))
