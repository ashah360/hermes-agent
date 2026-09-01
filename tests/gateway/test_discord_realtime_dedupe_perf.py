"""Live follow-up contracts: goal dedupe, running-aware recall, worker
performance config.

Production evidence driving these: two identical 180-char weather goals
dispatched 35s apart both ran to completion, posted two results, and spoke
serially — because dispatch had no live-goal dedupe and ``recall_result``
returned ``no_result`` for a RUNNING worker (inviting the redispatch).
Workers also ran with the session's default reasoning/service tier and a
hardcoded 120-iteration cap.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


class _Transport:
    def __init__(self):
        self.items = []
        self.creates = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self):
        return None

    async def inject_item(self, role, text):
        self.items.append((role, text))

    async def create_response(self, *, instructions=None):
        self.creates.append(instructions)

    async def close(self):
        return None


class _Outbox:
    def __init__(self):
        self.posts = []

    def post_result(self, record, result):
        self.posts.append(record.dispatch_id)
        return f"msg-{len(self.posts)}"


def _make(child_runner=None, raw_config=None):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane
    from plugins.platforms.discord.realtime.worker_bridge import WorkerBridge

    if child_runner is None:
        release = threading.Event()
        calls = []

        def child_runner(spec, hooks):  # noqa: F811
            calls.append(spec)
            release.wait(timeout=5)
            return {"status": "completed", "final_response": "done"}

        child_runner.release = release
        child_runner.calls = calls

    transport = _Transport()
    lane = RealtimeVoiceLane(
        guild_id=111, text_channel_id=222,
        config=load_realtime_voice_config(dict(raw_config or {}, enabled=True)),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )
    outbox = _Outbox()
    bridge = WorkerBridge(lane=lane, adapter=None, child_runner=child_runner, outbox=outbox)
    lane.worker_bridge = bridge
    return lane, bridge, transport, outbox, child_runner


WEATHER_GOAL = (
    "Check the current National Weather Service forecast for Austin, Texas "
    "for today and tomorrow, including temperature highs and lows, rain "
    "chances, and any active weather alerts for the area."
)


class TestLiveGoalDedupe:
    @pytest.mark.asyncio
    async def test_identical_goal_reuses_running_dispatch(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        first = bridge.dispatch(task=WEATHER_GOAL, user_id=42)
        second = bridge.dispatch(task=WEATHER_GOAL, user_id=42)

        assert second.dispatch_id == first.dispatch_id
        assert second.reused is True
        # One worker, one started event — never a second spawn.
        assert len(runner.calls) == 1
        await asyncio.sleep(0.05)  # let call_soon_threadsafe deliveries land
        assert [e.type for e in lane.delivered_events] == ["started"]

        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()
        # The exact live scenario: repeated user ask → ONE completion, ONE post.
        assert len([e for e in lane.delivered_events if e.type == "completed"]) == 1
        assert len(outbox.posts) == 1

    @pytest.mark.asyncio
    async def test_punctuation_and_whitespace_variants_reuse(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        first = bridge.dispatch(task="Check the   weather in Austin!", user_id=42)
        second = bridge.dispatch(task="check the weather in austin", user_id=42)
        assert second.dispatch_id == first.dispatch_id
        assert second.reused is True
        assert len(runner.calls) == 1
        runner.release.set()
        bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_supersedes_still_replaces_running_duplicate(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        first = bridge.dispatch(task=WEATHER_GOAL, user_id=42)
        replacement = bridge.dispatch(
            task=WEATHER_GOAL, user_id=42, supersedes_dispatch_id=first.dispatch_id
        )
        assert replacement.dispatch_id != first.dispatch_id
        assert replacement.reused is False
        assert lane.registry.get(first.dispatch_id).status == "cancelled"
        runner.release.set()
        bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_completed_goal_can_be_dispatched_again(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        first = bridge.dispatch(task=WEATHER_GOAL, user_id=42)
        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        # Finished → a fresh ask is a fresh dispatch (freshness matters).
        second = bridge.dispatch(task=WEATHER_GOAL, user_id=42)
        assert second is not None and second.dispatch_id != first.dispatch_id
        bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_simultaneous_duplicates_race_to_one_record(self):
        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(bridge.dispatch, task=WEATHER_GOAL, user_id=42)
                for _ in range(2)
            ]
            records = [f.result(timeout=5) for f in futures]
        assert records[0].dispatch_id == records[1].dispatch_id
        assert len(runner.calls) == 1
        runner.release.set()
        bridge.drain(timeout=5)
        await lane.stop()


class TestRecallRunning:
    @pytest.mark.asyncio
    async def test_recall_reports_running_and_forbids_redispatch(self):
        from plugins.platforms.discord.realtime.tools import handle_lane_tool

        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        record = bridge.dispatch(task=WEATHER_GOAL, user_id=42)

        by_id = json.loads(handle_lane_tool(
            lane, "recall_result", {"dispatch_id": record.dispatch_id}
        ))
        assert by_id["status"] == "running"
        assert by_id["dispatch_id"] == record.dispatch_id
        assert by_id["task"] == WEATHER_GOAL
        assert "not" in by_id["instruction"].lower()
        assert "dispatch" in by_id["instruction"].lower()

        # No id given, nothing completed yet → the live one, not no_result.
        latest = json.loads(handle_lane_tool(lane, "recall_result", {}))
        assert latest["status"] == "running"

        runner.release.set()
        bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        done = json.loads(handle_lane_tool(
            lane, "recall_result", {"dispatch_id": record.dispatch_id}
        ))
        assert done["status"] == "completed"
        assert done["result"] == "done"
        await lane.stop()

    @pytest.mark.asyncio
    async def test_no_result_only_when_registry_is_empty(self):
        from plugins.platforms.discord.realtime.tools import handle_lane_tool

        lane, bridge, transport, outbox, runner = _make()
        assert await lane.start() is True
        empty = json.loads(handle_lane_tool(lane, "recall_result", {}))
        assert empty["status"] == "no_result"
        runner.release.set()
        await lane.stop()


class TestWorkerPerformanceConfig:
    def test_defaults_and_bounds(self):
        cfg = load_realtime_voice_config({})
        assert cfg.worker_reasoning_effort == "low"
        assert cfg.worker_service_tier == "priority"
        assert cfg.worker_max_iterations == 30

        bounded = load_realtime_voice_config({"worker_max_iterations": 100000})
        assert bounded.worker_max_iterations <= 500
        floor = load_realtime_voice_config({"worker_max_iterations": 0})
        assert floor.worker_max_iterations >= 1
        bad_effort = load_realtime_voice_config({"worker_reasoning_effort": "ultra"})
        assert bad_effort.worker_reasoning_effort == "low"

    def test_principal_carries_same_model_low_reasoning_priority_tier(self):
        from plugins.platforms.discord.realtime.lane import _principal_provider_for

        base_spec = SimpleNamespace(
            model="gpt-5.6-sol", provider="openrouter", api_key="sk",
            base_url="https://x", api_mode=None,
            request_overrides={"extra_body": {"keep": 1}},
            enabled_toolsets=["terminal"], disabled_toolsets=None,
            session_id=None, prefill_messages=None,
        )
        runner = SimpleNamespace(
            build_realtime_worker_principal=lambda **kw: base_spec
        )
        adapter = SimpleNamespace(gateway_runner=runner)
        lane = SimpleNamespace(
            text_channel_id=222, last_speaker_user_id=42,
            config=load_realtime_voice_config({"enabled": True}),
        )
        principal = _principal_provider_for(adapter, lane)()
        # SAME model/provider — never a quality downgrade.
        assert principal.model == "gpt-5.6-sol"
        assert principal.provider == "openrouter"
        # Speed knobs: low reasoning + priority service tier, merged not
        # clobbered.
        assert principal.reasoning_config == {"enabled": True, "effort": "low"}
        assert principal.request_overrides["service_tier"] == "priority"
        assert principal.request_overrides["extra_body"] == {"keep": 1}

    @pytest.mark.asyncio
    async def test_worker_spec_carries_bounded_iterations(self):
        seen = {}

        def runner(spec, hooks):
            seen["max_iterations"] = spec.max_iterations
            return {"status": "completed", "final_response": "ok"}

        lane, bridge, transport, outbox, _ = _make(
            child_runner=runner, raw_config={"worker_max_iterations": 25}
        )
        assert await lane.start() is True
        bridge.dispatch(task="t", user_id=42)
        bridge.drain(timeout=5)
        await lane.stop()
        assert seen["max_iterations"] == 25

    def test_default_child_runner_uses_spec_iterations(self):
        from plugins.platforms.discord.realtime import worker_bridge as wb

        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                chat=lambda goal: "SPOKEN SUMMARY: ok",
                is_interrupted=False,
                interrupt=lambda reason=None: None,
                close=lambda: None,
            )

        spec = wb.WorkerSpec(
            dispatch_id="disp-x", goal="g", context_snapshot="c",
            user_id=42, guild_id=111, text_channel_id=222, max_iterations=25,
        )
        hooks = SimpleNamespace(interrupt_fn=lambda fn: None, finding=lambda t: None)
        with patch.object(wb, "build_worker_child", side_effect=fake_build):
            result = wb.default_child_runner(
                spec, hooks, principal=SimpleNamespace(model="m")
            )
        assert result["status"] == "completed"
        assert captured["max_iterations"] == 25
