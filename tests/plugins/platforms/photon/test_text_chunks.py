import re

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter
from plugins.platforms.photon.text_chunks import chunk_photon_text


def _adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-secret")
    return PhotonAdapter(
        PlatformConfig(
            enabled=True,
            token="",
            extra={
                "semantic_chunking_enabled": True,
                "semantic_chunk_soft_chars": 240,
                "semantic_chunk_hard_chars": 320,
            },
        )
    )


def test_short_text_is_one_unchanged_bubble():
    assert chunk_photon_text("short answer", soft_chars=240, hard_chars=320) == [
        "short answer"
    ]


def test_long_text_is_lossless_unlabelled_and_has_no_chunk_count_cap():
    paragraphs = [
        f"Section {index}\n" + (f"detail-{index} " * 20).strip()
        for index in range(12)
    ]
    text = "\n\n".join(paragraphs)

    chunks = chunk_photon_text(text, soft_chars=200, hard_chars=280)

    assert len(chunks) > 5
    assert all(len(chunk) <= 280 for chunk in chunks)
    assert "".join(chunks) == text
    assert not any(re.search(r"\(\d+/\d+\)", chunk) for chunk in chunks)


def test_oversized_fence_is_hard_bounded_and_balanced_per_bubble():
    text = "```python\n" + "\n".join(
        f"print('line {index:03d}')" for index in range(80)
    ) + "\n```"

    chunks = chunk_photon_text(text, soft_chars=220, hard_chars=300)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


@pytest.mark.asyncio
async def test_adapter_submits_all_chunks_as_one_ordered_batch(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = []

    async def call(path, body):
        calls.append((path, body))
        return {
            "ok": True,
            "complete": True,
            "messageIds": [
                f"chunk-{index}" for index, _ in enumerate(body["chunks"])
            ],
        }

    adapter._sidecar_call = call
    text = "\n\n".join(f"Paragraph {i}: " + "word " * 50 for i in range(5))
    result = await adapter.send("current-space", text)

    assert result.success is True
    assert len(calls) == 1
    assert calls[0][0] == "/send-batch"
    assert "".join(calls[0][1]["chunks"]) == text
    assert result.raw_response["part_count"] == len(calls[0][1]["chunks"])


@pytest.mark.asyncio
async def test_partial_batch_receipt_prevents_whole_response_retry(monkeypatch):
    adapter = _adapter(monkeypatch)
    top_level = []

    async def call(path, body):
        assert path == "/send-batch"
        return {
            "ok": True,
            "complete": False,
            "messageIds": ["chunk-0"],
            "failedIndex": 1,
            "error": "provider unavailable",
        }

    async def top_level_send(*args, **kwargs):
        top_level.append((args, kwargs))
        raise AssertionError("must not resend an acknowledged prefix")

    adapter._sidecar_call = call
    adapter._sidecar_send = top_level_send
    text = "\n\n".join(f"Paragraph {i}: " + "word " * 50 for i in range(5))

    result = await adapter._send_with_retry("current-space", text)

    assert result.success is False
    assert result.raw_response["acknowledged_message_ids"] == ["chunk-0"]
    assert top_level == []
