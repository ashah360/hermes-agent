from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


def _adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-secret")
    return PhotonAdapter(
        PlatformConfig(
            enabled=True,
            token="",
            extra={"conversation_actions_enabled": True},
        )
    )


@pytest.mark.asyncio
async def test_send_reply_uses_exact_native_reply_endpoint(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def call(path, body):
        calls.append((path, body))
        return {"ok": True, "messageIds": ["reply-guid"]}

    adapter._sidecar_call = call
    result = await adapter.send(
        "current-space", "anchored answer", reply_to="p:2/parent-guid"
    )

    assert result.success is True
    assert result.message_id == "reply-guid"
    assert calls == [
        (
            "/reply",
            {
                "spaceId": "current-space",
                "messageId": "p:2/parent-guid",
                "items": [{"type": "text", "text": "anchored answer"}],
            },
        )
    ]


@pytest.mark.asyncio
async def test_failed_reply_never_falls_back_to_top_level(monkeypatch):
    adapter = _adapter(monkeypatch)
    top_level_calls = []

    async def failed_send(*args, **kwargs):
        return SendResult(success=False, error="target not found")

    async def top_level(*args, **kwargs):
        top_level_calls.append((args, kwargs))
        return SendResult(success=True, message_id="wrong")

    adapter.send = failed_send
    adapter._sidecar_send = top_level

    result = await adapter._send_with_retry(
        "current-space", "must stay threaded", reply_to="missing-guid"
    )

    assert result.success is False
    assert top_level_calls == []


@pytest.mark.asyncio
async def test_attachment_reply_keeps_every_part_threaded(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG")
    monkeypatch.setattr(
        PhotonAdapter,
        "validate_media_delivery_path",
        staticmethod(lambda path: path),
    )
    calls = []

    async def call(path, body):
        calls.append((path, body))
        return {"ok": True, "messageIds": ["image-reply", "caption-reply"]}

    adapter._sidecar_call = call
    result = await adapter.send_image_file(
        "current-space",
        str(image),
        caption="details",
        reply_to="target-guid",
    )

    assert result.success is True
    assert calls[0][0] == "/reply"
    assert calls[0][1]["messageId"] == "target-guid"
    assert [item["type"] for item in calls[0][1]["items"]] == [
        "attachment",
        "text",
    ]
