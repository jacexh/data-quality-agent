import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
from agent.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_process_and_log():
    """Prevent actual S3 downloads in all main tests."""
    with patch("agent.main._process_and_log"):
        yield


@pytest.fixture(autouse=True)
async def reset_queue_state():
    """Drain queue and clear _processing between tests."""
    from agent.main import _queue, _processing
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except Exception:
            break
    _processing.clear()
    yield
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except Exception:
            break
    _processing.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _mcap_payload(key: str = "session.mcap", bucket: str = "robot-uploads") -> dict:
    return {
        "Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]
    }


# ── Existing behavior (must still pass) ──────────────────────────────────────

async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_notify_returns_accepted_for_mcap(client):
    response = await client.post("/notify", json=_mcap_payload("session.mcap"))
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


async def test_notify_ignores_non_mcap(client):
    response = await client.post("/notify", json=_mcap_payload("session.bag"))
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


async def test_notify_returns_401_on_bad_token():
    from agent.config import settings
    original = settings.webhook_auth_token
    settings.webhook_auth_token = "secret"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/notify", json={}, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
    finally:
        settings.webhook_auth_token = original


async def test_notify_returns_401_when_no_auth_header():
    """Missing Authorization header (not just wrong token) must also return 401."""
    from agent.config import settings
    original = settings.webhook_auth_token
    settings.webhook_auth_token = "secret"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/notify", json=_mcap_payload())  # no Authorization header
        assert response.status_code == 401
    finally:
        settings.webhook_auth_token = original


# ── New behavior ──────────────────────────────────────────────────────────────

async def test_notify_returns_429_when_queue_full(client):
    """When the queue is at capacity, /notify returns 429."""
    from agent.main import _queue
    from agent.config import settings

    # Fill the queue to capacity
    for i in range(settings.max_queue_size):
        await _queue.put((f"bucket-{i}", f"file-{i}.mcap"))

    response = await client.post("/notify", json=_mcap_payload("overflow.mcap"))
    assert response.status_code == 429
    assert response.json()["status"] == "queue_full"


async def test_notify_returns_duplicate_for_in_progress_key(client):
    """If a key is already being processed, /notify returns duplicate."""
    from agent.main import _processing

    _processing.add("session.mcap")
    response = await client.post("/notify", json=_mcap_payload("session.mcap"))
    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"


async def test_processing_key_removed_after_worker_completes():
    """_processing is cleared after _worker finishes an item (success or failure)."""
    from agent.main import _worker, _queue, _processing
    import asyncio

    mock_client = MagicMock()
    _processing.add("test.mcap")
    await _queue.put(("bucket", "test.mcap"))

    with patch("agent.main._process_and_log", side_effect=RuntimeError("boom")):
        task = asyncio.create_task(_worker(mock_client))
        await _queue.join()  # wait until task_done() is called (reliable sync point)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert "test.mcap" not in _processing


async def test_notify_has_no_background_tasks_param():
    """notify() must not accept BackgroundTasks — it would allow unbounded queueing."""
    import inspect
    from agent.main import notify
    sig = inspect.signature(notify)
    param_types = [p.annotation for p in sig.parameters.values()]
    from fastapi import BackgroundTasks
    assert BackgroundTasks not in param_types
