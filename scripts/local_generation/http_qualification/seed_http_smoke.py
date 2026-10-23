#!/usr/bin/env python3
"""Create only run-specific synthetic smoke identities in the disposable DB."""

from __future__ import annotations

import asyncio
import os

from evidencedesk_api.auth import Role
from evidencedesk_api.config import get_settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.models import User
from evidencedesk_api.security import hash_password


async def main() -> None:
    users = (
        (
            os.environ["SMOKE_ANALYST_USERNAME"],
            os.environ["SMOKE_ANALYST_PASSWORD"],
            Role.ANALYST,
        ),
        (
            os.environ["SMOKE_READER_USERNAME"],
            os.environ["SMOKE_READER_PASSWORD"],
            Role.READER,
        ),
    )
    engine = build_engine(get_settings())
    try:
        factory = build_session_factory(engine)
        async with factory() as session:
            session.add_all(
                User(username=name, password_hash=hash_password(password), role=role)
                for name, password, role in users
            )
            await session.commit()
    finally:
        await engine.dispose()
    print("PASS step=synthetic_identity_seed created=2")


if __name__ == "__main__":
    asyncio.run(main())
