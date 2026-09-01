"""Worker bridge contracts (ADR D7): isolated per-dispatch child workers,
typed lifecycle events, immutable context snapshots, no direct user
messaging, and event hygiene (no tool names, no heartbeats).

The child runner is injectable; these tests fake the run and assert the
bridge's behavior. One test exercises the REAL ``_build_child_agent`` path
to prove the worker principal contract (identity flag, blocked tools,
lineage).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


class _FakeTransport:
    def __init__(self):
        self.items = []
        self.responses = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self):
        return None

    async def inject_item(self, role, text):
        self.items.append((role, text))

    async def create_response(self, *, instructions=None):
        self.responses.append(instructions)

    async def close(self):
        return None


class _FakeOutbox:
    def __init__(self):
        self.posts = []

    def post_result(self, record, result):
        self.posts.append((record.dispatch_id, result))
        return f"msg-{len(self.posts)}"


def _make_lane_with_bridge(child_runner, *, max_inflight=3):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane
    from plugins.platforms.discord.realtime.worker_bridge import WorkerBridge

    transport = _FakeTransport()
    lane = RealtimeVoiceLane(
        guild_id=111,
        text_channel_id=222,
        config=load_realtime_voice_config(
            {"enabled": True, "max_inflight_dispatches": max_inflight}
        ),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )
    outbox = _FakeOutbox()
    bridge = WorkerBridge(lane=lane, adapter=None, child_runner=child_runner, outbox=outbox)
    lane.worker_bridge = bridge
    return lane, bridge, transport, outbox


class TestDispatchLifecycle:
    @pytest.mark.asyncio
    async def test_typed_events_started_finding_completed(self):
        def runner(spec, hooks):
            hooks.finding("Deposits data located; June totals reconcile.")
            return {
                "status": "completed",
                "final_response": (
                    "July deposits were $38M, up 4% ([Mercury](https://mercury.com/x)).\n"
                    "SPOKEN SUMMARY: July deposits were $38M, up 4 percent from June."
                ),
            }

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True

        record = bridge.dispatch(task="check July deposits", user_id=42)
        assert record.dispatch_id
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()

        types = [e.type for e in lane.delivered_events]
        assert types == ["started", "finding", "completed"]
        completed = lane.delivered_events[-1]
        assert completed.dispatch_id == record.dispatch_id
        # Sources extracted from worker citations, with freshness.
        assert completed.sources and completed.sources[0]["url"] == "https://mercury.com/x"
        assert completed.unsourced is False
        # Exact sourced synthesis, normalized for speech (38M -> 38 million).
        assert "38 million" in completed.spoken_synthesis
        assert "$38M" not in completed.spoken_synthesis
        # Detail posted through the outbox exactly once; event carries the ref.
        assert len(outbox.posts) == 1
        assert completed.detail_ref == "msg-1"

    @pytest.mark.asyncio
    async def test_no_interim_content_means_no_finding_no_heartbeat(self):
        def runner(spec, hooks):
            time.sleep(0.05)  # long enough that a fake-cadence bug would fire
            return {"status": "completed", "final_response": "Done. Nothing notable."}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True
        bridge.dispatch(task="quiet task", user_id=42)
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()
        types = [e.type for e in lane.delivered_events]
        assert types == ["started", "completed"]

    @pytest.mark.asyncio
    async def test_spoken_hints_never_contain_tool_names(self):
        def runner(spec, hooks):
            hooks.finding("Checked `web_search` and search_files() — deposits found.")
            return {"status": "completed", "final_response": "ok"}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True
        bridge.dispatch(task="t", user_id=42)
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()
        for event in lane.delivered_events:
            assert "web_search" not in event.spoken_hint
            assert "search_files" not in event.spoken_hint

    @pytest.mark.asyncio
    async def test_blocker_event_from_failure(self):
        def runner(spec, hooks):
            hooks.blocker("Needs approval to run a deletion command.")
            return {"status": "failed", "error": "approval denied"}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True
        bridge.dispatch(task="t", user_id=42)
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()
        types = [e.type for e in lane.delivered_events]
        assert types == ["started", "blocker", "failed"]

    @pytest.mark.asyncio
    async def test_capacity_rejection_is_structured(self):
        import threading

        release = threading.Event()

        def runner(spec, hooks):
            release.wait(timeout=5)
            return {"status": "completed", "final_response": "ok"}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner, max_inflight=1)
        assert await lane.start() is True
        first = bridge.dispatch(task="a", user_id=42)
        assert first.dispatch_id
        rejected = bridge.dispatch(task="b", user_id=42)
        assert rejected is None or getattr(rejected, "rejected", False)
        release.set()
        bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_context_snapshot_is_immutable_and_high_signal(self):
        seen = {}

        def runner(spec, hooks):
            seen["snapshot"] = spec.context_snapshot
            return {"status": "completed", "final_response": "ok"}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True
        lane.note_transcript("user", "what were July deposits?")
        bridge.dispatch(task="check July deposits", user_id=42)
        # Later conversation must not mutate the running worker's context.
        lane.note_transcript("user", "unrelated later chatter")
        bridge.drain(timeout=5)
        await lane.stop()

        snap = seen["snapshot"]
        assert "check July deposits" in snap
        assert "what were July deposits?" in snap
        assert "unrelated later chatter" not in snap
        assert "guild 111" in snap or "111" in snap

    @pytest.mark.asyncio
    async def test_reinjection_defers_while_user_is_speaking(self):
        def runner(spec, hooks):
            return {"status": "completed", "final_response": "ok"}

        lane, bridge, transport, outbox = _make_lane_with_bridge(runner)
        assert await lane.start() is True
        lane.on_user_speech_started()  # user talking
        bridge.dispatch(task="t", user_id=42)
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        # Nothing injected into the provider conversation yet.
        assert transport.items == []
        lane.on_user_speech_stopped()  # boundary → flush
        await asyncio.sleep(0.1)
        assert transport.items != []
        await lane.stop()


class TestRealChildPrincipal:
    def test_default_child_carries_jeeves_identity_and_blocked_tools(self):
        """E2E through the real _build_child_agent: same Jeeves identity flag
        (SOUL.md via load_soul_identity), no direct user messaging (leaf
        blocklist), parent lineage recorded."""
        from plugins.platforms.discord.realtime.worker_bridge import build_worker_child

        parent = SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key-1234567890",
            model="test/model",
            provider=None,
            api_mode=None,
            enabled_toolsets=["terminal", "file", "messaging"],
            disabled_toolsets=None,
            session_id="parent-session-1",
            prefill_messages=None,
            request_overrides={},
        )
        with patch("run_agent.check_toolset_requirements", return_value={}), \
             patch("run_agent.OpenAI"):
            child = build_worker_child(
                goal="verify July deposits",
                context_snapshot="ctx",
                principal=parent,
                max_iterations=10,
            )
        try:
            assert child.load_soul_identity is True
            assert child._parent_session_id == "parent-session-1"
            enabled = set(child.enabled_toolsets or [])
            disabled = set(child.disabled_toolsets or [])
            # Workers never address the user directly: send_message is
            # gateway-session-injected and never part of AIAgent toolsets, so
            # a platform="subagent" child cannot have it; clarify/delegation
            # ride the child deny list.
            assert "send_message" not in set(child.valid_tool_names or set())
            assert "clarify" in disabled
            assert "delegation" not in enabled
        finally:
            close = getattr(child, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
