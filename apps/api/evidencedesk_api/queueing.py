from typing import Protocol
from uuid import UUID

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings


class TaskQueue(Protocol):
    async def enqueue_document(
        self, document_id: UUID, task_id: UUID, correlation_id: UUID
    ) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> bool: ...


class ArqTaskQueue:
    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_document(
        self, document_id: UUID, task_id: UUID, correlation_id: UUID
    ) -> None:
        await self._pool.enqueue_job(
            "process_document",
            str(document_id),
            str(task_id),
            str(correlation_id),
            _job_id=f"document:{document_id}",
        )

    async def close(self) -> None:
        await self._pool.aclose()

    async def ping(self) -> bool:
        return bool(await self._pool.ping())


async def build_task_queue(redis_url: str) -> TaskQueue:
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    return ArqTaskQueue(pool)
