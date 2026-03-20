import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from agent.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_notify_returns_200_immediately(client):
    payload = {
        "EventName": "s3:ObjectCreated:Put",
        "Key": "robot-uploads/session.mcap",
        "Records": [{"s3": {"bucket": {"name": "robot-uploads"}, "object": {"key": "session.mcap"}}}],
    }
    with patch("agent.main._process_mcap") as mock_process:
        response = await client.post("/notify", json=payload)
    assert response.status_code == 200


async def test_notify_ignores_non_mcap(client):
    payload = {
        "EventName": "s3:ObjectCreated:Put",
        "Key": "robot-uploads/session.bag",
        "Records": [{"s3": {"bucket": {"name": "robot-uploads"}, "object": {"key": "session.bag"}}}],
    }
    with patch("agent.main._process_mcap") as mock_process:
        response = await client.post("/notify", json=payload)
    assert response.status_code == 200
    mock_process.assert_not_called()


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
