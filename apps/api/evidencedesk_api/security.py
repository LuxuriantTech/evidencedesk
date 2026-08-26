from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from evidencedesk_api.auth import Role, UserPrincipal

_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    return _password_hash.verify(password, encoded)


def create_access_token(
    user: UserPrincipal,
    *,
    secret: str,
    now: datetime | None = None,
    ttl: timedelta = timedelta(minutes=30),
) -> str:
    issued_at = now or datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + ttl).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, *, secret: str, now: datetime | None = None) -> UserPrincipal:
    payload = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False})
    checked_at = now or datetime.now(UTC)
    if int(payload["exp"]) <= int(checked_at.timestamp()):
        raise jwt.ExpiredSignatureError("Token has expired")
    return UserPrincipal(
        id=UUID(payload["sub"]),
        username=str(payload["username"]),
        role=Role(payload["role"]),
    )
