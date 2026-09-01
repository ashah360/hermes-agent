"""Turn sequencing vs. dispatch ownership (ADR D8), and the in-lane tools
(ADR D6): follow-up/barge-in never revokes workers; explicit cancel,
supersession, reset, and leave do; delivery is exactly-once and exactly
routed; recall_result cannot fabricate.
"""

import asyncio
import json
import threading

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


class _FakeTransport:
    def __init__(self):
        self.items = []
        self.responses = []
        self.cancels = 0

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self):
        self.cancels += 1

    async def inject_item(self, role, text):
        self.items.append((role, text))

    async def create_response(self, *, instructions=None):
        self.responses.append(instructions)

    async def close(self):
        return None


class _Outbox:
    def __init__(self):
        self.posts = []

    def post_result(self, record, result):
        self.posts.append(record.dispatch_id)
        return f"msg-{len(self.posts)}"


def _make(child_runner=None, *, gate=None):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane
    from plugins.platforms.discord.realtime.worker_bridge import WorkerBridge

    if child_runner is None:
        release = threading.Event()

        def child_runner(spec, hooks):  # noqa: F811 — default slow worker
            release.wait(timeout=5)
            return {"status": "completed", "final_response": "done"}

        child_runner.release = release

    transport = _FakeTransport()
    lane = RealtimeVoiceLane(
        guild_id=111,
        text_channel_id=222,
        config=load_realtime_voice_config({"enabled": True}),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )
    outbox = _Outbox()
    bridge = WorkerBridge(lane=lane, adapter=None, child_runner=child_runner, outbox=outbox)
    lane.worker_bridge = bridge
    return lane, bridge, transport, outbox, child_runner


class TestFollowUpVsCorrection:
    @pytest.mark.asyncio
    async def test_follow_up_and_barge_in_do_not_revoke_dispatch(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        record = bridge.dispatch(task="long research", user_id=42)

        # Ordinary follow-up speech + barge-in on the audio plane.
        lane.on_user_speech_started()
        lane.on_user_speech_stopped()
        lane.on_user_speech_started()
        lane.on_user_speech_stopped()
        assert lane.turn_seq == 2

        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()

        # Completion still delivered — barge-in/follow-up never staled it.
        assert "completed" in [e.type for e in lane.delivered_events]
        assert lane.telemetry.counter("stale_events_dropped") == 0

    @pytest.mark.asyncio
    async def test_explicit_cancel_revokes_and_interrupts(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        record = bridge.dispatch(task="obsolete work", user_id=42)

        result = bridge.cancel(record.dispatch_id, reason="user corrected")
        assert result is True
        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()

        types = [e.type for e in lane.delivered_events]
        assert "completed" not in types
        assert lane.telemetry.counter("stale_events_dropped") >= 1

    @pytest.mark.asyncio
    async def test_supersession_revokes_prior_dispatch(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        old = bridge.dispatch(task="Q2 numbers", user_id=42)
        new = bridge.dispatch(task="Q3 numbers actually", user_id=42,
                              supersedes_dispatch_id=old.dispatch_id)
        assert new.dispatch_id != old.dispatch_id
        assert lane.registry.get(old.dispatch_id).status == "cancelled"

        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()

        completed = [e for e in lane.delivered_events if e.type == "completed"]
        assert [e.dispatch_id for e in completed] == [new.dispatch_id]

    @pytest.mark.asyncio
    async def test_leave_drops_voice_events_but_outbox_still_posts(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        bridge.dispatch(task="still running at leave", user_id=42)
        await lane.stop(reason="leave")   # leave: voice delivery revoked

        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.05)

        assert "completed" not in [e.type for e in lane.delivered_events]
        # Text delivery survives the lane: detailed result still posts once.
        assert len(outbox.posts) == 1

    @pytest.mark.asyncio
    async def test_duplicate_completion_delivers_exactly_once(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        record = bridge.dispatch(task="t", user_id=42)
        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)

        # Replay the terminal event (duplicated delivery attempt).
        completed = [e for e in lane.delivered_events if e.type == "completed"]
        assert len(completed) == 1
        redelivered = lane.deliver_worker_event(completed[0])
        assert redelivered is False
        await lane.stop()
        assert len([e for e in lane.delivered_events if e.type == "completed"]) == 1

    @pytest.mark.asyncio
    async def test_unknown_dispatch_event_is_dropped(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        from plugins.platforms.discord.realtime.events import WorkerEvent

        ghost = WorkerEvent(
            dispatch_id="disp-unknown", epoch=0, type="completed",
            spoken_hint="ghost", detail_ref=None, sources=(), unsourced=True,
            ts=0.0, spoken_synthesis=None,
        )
        assert lane.deliver_worker_event(ghost) is False
        runner.release.set()
        bridge.drain(timeout=5)
        await lane.stop()
        assert lane.telemetry.counter("stale_events_dropped") >= 1


class TestInLaneTools:
    @pytest.mark.asyncio
    async def test_tool_schemas_are_the_four_lane_tools(self):
        from plugins.platforms.discord.realtime.tools import lane_tool_schemas

        schemas = lane_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"voice_context", "recall_result", "hermes_dispatch", "hermes_cancel"}
        assert all(s["type"] == "function" for s in schemas)

    @pytest.mark.asyncio
    async def test_recall_result_registry_only_never_fabricates(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        from plugins.platforms.discord.realtime.tools import handle_lane_tool

        empty = json.loads(handle_lane_tool(lane, "recall_result", {}))
        assert empty["status"] == "no_result"

        record = bridge.dispatch(task="t", user_id=42)
        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)

        found = json.loads(handle_lane_tool(
            lane, "recall_result", {"dispatch_id": record.dispatch_id}
        ))
        assert found["status"] == "completed"
        assert found["result"] == "done"

        ghost = json.loads(handle_lane_tool(
            lane, "recall_result", {"dispatch_id": "disp-nope"}
        ))
        assert ghost["status"] == "no_result"
        await lane.stop()

    @pytest.mark.asyncio
    async def test_dispatch_and_cancel_tools_route_to_bridge(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        from plugins.platforms.discord.realtime.tools import handle_lane_tool

        out = json.loads(handle_lane_tool(
            lane, "hermes_dispatch", {"task": "check deposits"}
        ))
        assert out["status"] == "dispatched"
        dispatch_id = out["dispatch_id"]

        cancelled = json.loads(handle_lane_tool(
            lane, "hermes_cancel", {"dispatch_id": dispatch_id, "reason": "changed mind"}
        ))
        assert cancelled["status"] == "cancelled"
        runner.release.set()
        bridge.drain(timeout=5)
        await lane.stop()
