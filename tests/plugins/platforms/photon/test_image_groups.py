import os

import pytest

from gateway.config import PlatformConfig
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
async def test_image_group_validates_all_files_then_preserves_order_and_ids(
    monkeypatch, tmp_path
):
    paths = []
    for name in ("first.png", "second.jpg", "third.webp"):
        path = tmp_path / name
        path.write_bytes(b"image")
        paths.append(str(path))
    monkeypatch.setattr(
        PhotonAdapter,
        "validate_media_delivery_path",
        staticmethod(lambda path: path if os.path.isfile(path) else None),
    )
    adapter = _adapter(monkeypatch)
    calls = []

    async def call(path, body):
        calls.append((path, body))
        return {
            "ok": True,
            "parentMessageId": "parent-guid",
            "childMessageIds": ["child-0", "child-1", "child-2", "caption-3"],
            "partCount": 4,
        }

    adapter._sidecar_call = call
    result = await adapter.send_image_group(
        "current-space", paths, caption="compare"
    )

    assert result.success is True
    assert calls[0][0] == "/send-group"
    assert [item["path"] for item in calls[0][1]["images"]] == paths
    assert result.raw_response == {
        "parent_message_id": "parent-guid",
        "child_message_ids": ["child-0", "child-1", "child-2", "caption-3"],
        "part_count": 4,
        "presentation": "group",
    }


@pytest.mark.asyncio
async def test_missing_image_prevents_any_sidecar_send(monkeypatch, tmp_path):
    valid = tmp_path / "valid.png"
    valid.write_bytes(b"image")
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(
        PhotonAdapter,
        "validate_media_delivery_path",
        staticmethod(lambda path: path if os.path.isfile(path) else None),
    )
    adapter = _adapter(monkeypatch)
    calls = []

    async def call(path, body):
        calls.append((path, body))
        raise AssertionError("sidecar must not be called")

    adapter._sidecar_call = call
    result = await adapter.send_image_group(
        "current-space", [str(valid), str(missing)]
    )

    assert result.success is False
    assert calls == []
