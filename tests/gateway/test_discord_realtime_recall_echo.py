"""Live behavior fixes on b8681170: dispatch-first tool contracts (B),
recall_result double-speak suppression (C), transcript persistence (D),
and the execute-fully worker preamble (amendment).

Live logs 06:01:54-06:02:48: after a completed event was spoken, the model
immediately called recall_result for the same dispatch and spoke the result
AGAIN. And with no transcripts persisted, nobody could review what was said.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config
from plugins.platforms.discord.realtime.events import WorkerEvent


class _Transport:
    def __init__(self):
        self.items = []
        self.creates = []
        self.fn_outputs = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self, *a, **k):
        return None

    async def inject_item(self, role, text):
        self.items.append((role, text))

    async def send_function_output(self, call_id, output):
        self.fn_outputs.append((call_id, output))

    async def create_response(self, *, instructions=None):
        self.creates.append(instructions)

    async def close(self):
        return None


async def _make_lane(**cfg):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

    transport = _Transport()
    lane = RealtimeVoiceLane(
        guild_id=111, text_channel_id=222,
        config=load_realtime_voice_config(dict({"enabled": True}, **cfg)),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )
    assert await lane.start() is True
    return lane, transport


def _completed_event(lane, payload="July deposits were 38 million."):
    record = lane.registry.register(
        goal="check deposits", guild_id=111, text_channel_id=222, user_id=42
    )
    lane.store_result_payload(record.dispatch_id, {
        "final_response": payload, "sources": [], "fetched_at": "2026-09-02",
    })
    lane.registry.mark_terminal(record.dispatch_id, "completed")
    return WorkerEvent(
        dispatch_id=record.dispatch_id, epoch=record.epoch, type="completed",
        spoken_hint="The result is ready.", detail_ref=None, sources=(),
        unsourced=True, ts=0.0, spoken_synthesis=None,
    )


class TestToolContracts:
    def test_dispatch_description_is_mandatory_immediate(self):
        from plugins.platforms.discord.realtime.tools import lane_tool_schemas

        schemas = {s["name"]: s for s in lane_tool_schemas()}
        dispatch = schemas["hermes_dispatch"]["description"].lower()
        assert "any request" in dispatch
        assert "immediately" in dispatch
        assert "do not ask" in dispatch

        recall = schemas["recall_result"]["description"].lower()
        assert "never call it right after" in recall
        assert "already spoken" in recall


class TestRecallEchoSuppression:
    @pytest.mark.asyncio
    async def test_recall_same_dispatch_after_completed_speaks_nothing_new(self):
        """(C) completed event → one response → immediate recall_result for
        the SAME dispatch, no intervening user speech → tool output is
        answered but ZERO additional response.create."""
        lane, transport = await _make_lane()
        event = _completed_event(lane)
        assert lane.deliver_worker_event(event) is True
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1        # the spoken result

        lane.on_response_done()
        # Model echoes with a recall for the same dispatch (live bug shape).
        lane.on_function_call(
            "recall_result", {"dispatch_id": event.dispatch_id}, "call_echo"
        )
        await asyncio.sleep(0.05)
        # The tool call IS answered (provider needs the output)…
        assert transport.fn_outputs and transport.fn_outputs[-1][0] == "call_echo"
        payload = json.loads(transport.fn_outputs[-1][1])
        assert payload["status"] == "completed"
        # …but no second spoken response is created.
        assert len(transport.creates) == 1
        assert lane.telemetry.counter("recall_echo_suppressed") == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_bare_recall_right_after_completed_is_also_suppressed(self):
        # The live model often calls recall_result with no args.
        lane, transport = await _make_lane()
        event = _completed_event(lane)
        lane.deliver_worker_event(event)
        await asyncio.sleep(0.05)
        lane.on_response_done()
        lane.on_function_call("recall_result", {}, "call_echo2")
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_user_speech_between_resets_suppression(self):
        """After the user actually speaks, recall is a genuine question and
        answers normally (with its continuation)."""
        lane, transport = await _make_lane()
        event = _completed_event(lane)
        lane.deliver_worker_event(event)
        await asyncio.sleep(0.05)
        lane.on_response_done()
        lane.on_user_speech_started()
        lane.on_user_speech_stopped()
        lane.on_function_call(
            "recall_result", {"dispatch_id": event.dispatch_id}, "call_real"
        )
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 2        # genuine follow-up answered
        await lane.stop()


class TestTranscriptLogging:
    @pytest.mark.asyncio
    async def test_transcripts_logged_and_persisted(self, caplog):
        """(D) user + assistant transcripts hit INFO logs and the JSONL file."""
        lane, transport = await _make_lane()
        with caplog.at_level(logging.INFO):
            lane.on_input_transcript("what were July deposits?")
            lane.on_response_created("r1")
            lane.on_output_transcript_delta("July deposits were ", "r1")
            lane.on_output_transcript_delta("38 million.", "r1")
            lane.on_response_done()
        await lane.stop()

        lines = [r.message for r in caplog.records
                 if "discord_realtime transcript" in r.message]
        assert any("role=user" in l and "July deposits" in l for l in lines)
        assert any("role=assistant" in l and "38 million" in l for l in lines)

        jsonl = Path(os.environ["HERMES_HOME"]) / "logs" / "discord_realtime_transcript.jsonl"
        assert jsonl.is_file()
        rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        roles = [r["role"] for r in rows]
        assert "user" in roles and "assistant" in roles
        assert all(r["guild"] == 111 for r in rows)

    @pytest.mark.asyncio
    async def test_transcript_logging_can_be_disabled(self):
        lane, transport = await _make_lane(transcript_logging=False)
        lane.on_input_transcript("secret words")
        lane.on_response_created("r1")
        lane.on_output_transcript_delta("reply", "r1")
        lane.on_response_done()
        await lane.stop()
        jsonl = Path(os.environ["HERMES_HOME"]) / "logs" / "discord_realtime_transcript.jsonl"
        assert not jsonl.exists()

    def test_input_transcription_default_model(self):
        cfg = load_realtime_voice_config({})
        assert cfg.input_transcription_model == "gpt-4o-mini-transcribe"
        assert load_realtime_voice_config({}).transcript_logging is True


class TestWorkerExecuteFullyPreamble:
    @pytest.mark.asyncio
    async def test_context_snapshot_orders_full_execution_no_hedging(self):
        from plugins.platforms.discord.realtime.worker_bridge import WorkerBridge

        seen = {}

        def runner(spec, hooks):
            seen["snapshot"] = spec.context_snapshot
            return {"status": "completed", "final_response": "ok"}

        lane, transport = await _make_lane()
        bridge = WorkerBridge(lane=lane, adapter=None, child_runner=runner, outbox=None)
        lane.worker_bridge = bridge
        bridge.dispatch(task="draft the memo", user_id=42)
        bridge.drain(timeout=5)
        await lane.stop()

        snap = seen["snapshot"].lower()
        assert "execute the task fully" in snap
        assert "finished result" in snap