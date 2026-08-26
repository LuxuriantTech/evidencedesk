from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from evidencedesk_api.auth import Role
from evidencedesk_api.config import Settings
from evidencedesk_api.db import build_engine, build_session_factory
from evidencedesk_api.models import Dossier, User
from evidencedesk_api.security import hash_password

DEMO_DOSSIER_ID = UUID("08dcdf59-6086-43da-bb59-29fb37bd67f6")


async def _ensure_user(session: AsyncSession, *, username: str, password: str, role: Role) -> None:
    existing = await session.scalar(select(User).where(User.username == username))
    if existing is None:
        session.add(User(username=username, password_hash=hash_password(password), role=role))


async def seed_demo_data(settings: Settings, *, engine: AsyncEngine | None = None) -> None:
    owned_engine = engine is None
    active_engine = engine or build_engine(settings)
    factory: async_sessionmaker[AsyncSession] = build_session_factory(active_engine)
    async with factory() as session:
        await _ensure_user(
            session,
            username="demo.admin",
            password=settings.demo_admin_password,
            role=Role.ADMIN,
        )
        await _ensure_user(
            session,
            username="demo.analyst",
            password=settings.demo_analyst_password,
            role=Role.ANALYST,
        )
        await _ensure_user(
            session,
            username="demo.reader",
            password=settings.demo_reader_password,
            role=Role.READER,
        )
        dossier = await session.get(Dossier, DEMO_DOSSIER_ID)
        if dossier is None:
            session.add(
                Dossier(
                    id=DEMO_DOSSIER_ID,
                    name="Northwind supplier review",
                    description="Synthetic supplier agreement and incident evidence.",
                    is_synthetic=True,
                )
            )
        await session.commit()
    if owned_engine:
        await active_engine.dispose()
