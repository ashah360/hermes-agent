"""Production wiring + response coordination (live correctness emergency).

Live evidence fixed here, as contracts:
A. ``create_lane`` never instantiated a WorkerBridge → the live model got
   "worker dispatch unavailable". Production lane creation must build a
   working bridge with a REAL principal provider from
   ``adapter.gateway_runner`` (never a running cached session agent).
B. Response ownership was racy: function-call continuation + ``started``
   event + queued findings each issued ``response.create`` concurrently
   (~4 overlapping replies live). One coordinator per lane: at most ONE
   provider response in flight, ownership marked synchronously, worker
   events serialized one per response cycle, ``started`` never triggers a
   second response.
C. ``response.cancel`` was sent on every speech start (7×
   ``response_cancel_not_active`` live). Cancel only when a response is
   actually active, exactly once.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


class CoordTransport:
    """Fake transport recording the full ordered outbound stream."""

    def __init__(self):
        self.creates = []
        self.cancels = 0
        self.items = []
        self.fn_outputs = []
        self.order = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self):
        self.cancels += 1
        self.order.append("cancel")

    async def inject_item(self, role, text):
        self.items.append((role, text))
        self.order.append(f"item:{text.splitlines()[0][:40]}")

    async def send_function_output(self, call_id, output):
        self.fn_outputs.append((call_id, output))
        self.order.append(f"fn_output:{call_id}")

    async def create_response(self, *, instructions=None):
        self.creates.append(instructions)
        self.order.append("create")

    async def close(self):
        return None


def _make_adapter(runner=None):
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform, PlatformConfig

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_mixers = {}
    adapter._voice_receivers = {}
    adapter._voice_text_channels = {111: 222}
    adapter.gateway_runner = runner
    return adapter


def _production_lane(runner=None, transport=None):
    """Lane through the REAL production create_lane (no manual bridge)."""
    import plugins.platforms.discord.realtime.lane as lane_mod

    transport = transport or CoordTransport()
    adapter = _make_adapter(runner=runner)
    lane = lane_mod.create_lane(
        adapter=adapter,
        guild_id=111,
        voice_client=MagicMock(),
        receiver=None,
        raw_config={"enabled": True},
        transport_factory=lambda **kw: transport,
    )
    lane._api_key = "sk-test"  # hermetic env has no real secret
    return lane, transport, adapter


class TestProductionBridgeWiring:
    @pytest.mark.asyncio
    async def test_create_lane_builds_working_bridge(self):
        lane, transport, adapter = _production_lane()
        assert lane.worker_bridge is not None
        assert await lane.start() is True

        from plugins.platforms.discord.realtime.tools import handle_lane_tool

        def fake_runner(spec, hooks, *, principal=None):
            return {"status": "completed", "final_response": "done"}

        with patch(
            "plugins.platforms.discord.realtime.worker_bridge.default_child_runner",
            fake_runner,
        ):
            out = json.loads(handle_lane_tool(lane, "hermes_dispatch", {"task": "t"}))
        # The live defect: this returned status=error "worker dispatch
        # unavailable". Production lanes must dispatch.
        assert out["status"] == "dispatched"
        lane.worker_bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_principal_provider_comes_from_gateway_runner(self):
        sentinel = SimpleNamespace(model="m", api_key="k")
        runner = SimpleNamespace(
            build_realtime_worker_principal=lambda **kw: sentinel
        )
        lane, transport, adapter = _production_lane(runner=runner)
        assert lane.worker_bridge._principal_provider() is sentinel

    @pytest.mark.asyncio
    async def test_missing_principal_fails_closed_with_clear_blocker(self):
        lane, transport, adapter = _production_lane(runner=None)
        assert await lane.start() is True
        record = lane.worker_bridge.dispatch(task="t", user_id=42)
        assert record is not None
        lane.worker_bridge.drain(timeout=5)
        await asyncio.sleep(0.1)
        await lane.stop()
        terminal = [e for e in lane.delivered_events if e.type == "failed"]
        assert len(terminal) == 1
        assert "principal" in terminal[0].spoken_hint.lower()

    def test_gateway_runner_principal_factory_resolves_session_runtime(self):
        # The narrow factory must reuse the gateway's own resolution chain,
        # not invent a parallel one.
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        calls = {}

        def _resolve(**kw):
            calls["resolved"] = True
            return "test/model", {
                "api_key": "sk-live", "base_url": "https://x", "provider": "openrouter",
                "api_mode": None, "request_overrides": {"a": 1},
            }

        runner._resolve_session_agent_runtime = _resolve
        runner._resolve_enabled_toolsets_for_source = (
            lambda cfg, source, platform_key: ["terminal", "file"]
        )
        with patch("gateway.run._load_gateway_config", return_value={"agent": {}}):
            principal = runner.build_realtime_worker_principal(
                chat_id="222", user_id="42", user_name="Shaan"
            )
        assert calls.get("resolved") is True
        assert principal.model == "test/model"
        assert principal.api_key == "sk-live"
        assert principal.enabled_toolsets == ["terminal", "file"]
        # Never a live session agent: a spec object with no run loop.
        assert not hasattr(principal, "run_conversation")

        # Fail closed without credentials.
        runner._resolve_session_agent_runtime = lambda **kw: ("m", {"api_key": ""})
        with patch("gateway.run._load_gateway_config", return_value={"agent": {}}):
            assert runner.build_realtime_worker_principal(chat_id="222") is None


class TestResponseCoordinator:
    @pytest.mark.asyncio
    async def test_function_call_plus_started_yield_exactly_one_response(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True

        gate = threading.Event()

        def slow_runner(spec, hooks, *, principal=None):
            gate.wait(timeout=5)
            return {"status": "completed", "final_response": "done"}

        with patch(
            "plugins.platforms.discord.realtime.worker_bridge.default_child_runner",
            slow_runner,
        ):
            # Realistic provider order: model response begins, calls the tool.
            lane.on_response_created("r1")
            lane.on_function_call("hermes_dispatch", {"task": "t"}, "call_1")
            await asyncio.sleep(0.1)
            # While the calling response is still open: output item sent,
            # started item may be queued/injected, but NO response.create yet.
            assert transport.fn_outputs and transport.fn_outputs[0][0] == "call_1"
            assert len(transport.creates) == 0
            # Calling response finishes → exactly ONE continuation response.
            lane.on_response_done()
            await asyncio.sleep(0.1)
            assert len(transport.creates) == 1

            gate.set()
            lane.worker_bridge.drain(timeout=5)
        await lane.stop()

    @pytest.mark.asyncio
    async def test_concurrent_finding_and_completion_never_overlap(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True
        record = lane.registry.register(
            goal="g", guild_id=111, text_channel_id=222, user_id=42
        )
        from plugins.platforms.discord.realtime.events import WorkerEvent

        def _evt(etype, hint):
            return WorkerEvent(
                dispatch_id=record.dispatch_id, epoch=record.epoch, type=etype,
                spoken_hint=hint, detail_ref=None, sources=(), unsourced=True,
                ts=0.0, spoken_synthesis=None,
            )

        lane.deliver_worker_event(_evt("finding", "found a thing"))
        lane.deliver_worker_event(_evt("completed", "all done"))
        await asyncio.sleep(0.1)
        # Only ONE response scheduled; the completion waits its turn.
        assert len(transport.creates) == 1
        lane.on_response_done()
        await asyncio.sleep(0.1)
        assert len(transport.creates) == 2
        await lane.stop()

    @pytest.mark.asyncio
    async def test_four_queued_events_serialize_one_per_response_cycle(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True
        from plugins.platforms.discord.realtime.events import WorkerEvent

        records = [
            lane.registry.register(goal=f"g{i}", guild_id=111,
                                   text_channel_id=222, user_id=42)
            for i in range(4)
        ]
        for i, record in enumerate(records):
            lane.deliver_worker_event(WorkerEvent(
                dispatch_id=record.dispatch_id, epoch=record.epoch,
                type="completed", spoken_hint=f"done {i}", detail_ref=None,
                sources=(), unsourced=True, ts=0.0, spoken_synthesis=None,
            ))
        await asyncio.sleep(0.1)
        assert len(transport.creates) == 1
        for expected in (2, 3, 4):
            lane.on_response_done()
            await asyncio.sleep(0.05)
            assert len(transport.creates) == expected
        # All four delivered, none dropped, none overlapped.
        assert len([e for e in lane.delivered_events if e.type == "completed"]) == 4
        await lane.stop()

    @pytest.mark.asyncio
    async def test_started_event_injects_context_without_response(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True
        record = lane.registry.register(
            goal="g", guild_id=111, text_channel_id=222, user_id=42
        )
        from plugins.platforms.discord.realtime.events import WorkerEvent

        lane.deliver_worker_event(WorkerEvent(
            dispatch_id=record.dispatch_id, epoch=record.epoch, type="started",
            spoken_hint="Working on it", detail_ref=None, sources=(),
            unsourced=True, ts=0.0, spoken_synthesis=None,
        ))
        await asyncio.sleep(0.1)
        assert transport.items          # context injected
        assert transport.creates == []  # but no second voice response
        await lane.stop()


class TestCancelDiscipline:
    @pytest.mark.asyncio
    async def test_idle_speech_start_sends_zero_cancels_but_clears_playback(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True
        cleared = []

        class _Child:
            finished = False

            def clear(self):
                cleared.append(True)

            def end(self):
                pass

        lane._stream_child = _Child()
        lane.on_user_speech_started()   # no response active
        await lane.stop()
        assert cleared == [True]
        assert transport.cancels == 0

    @pytest.mark.asyncio
    async def test_active_response_gets_exactly_one_cancel(self):
        lane, transport, adapter = _production_lane()
        assert await lane.start() is True
        lane.on_response_created("r1")
        lane.on_user_speech_started()
        lane.on_user_speech_started()   # same still-active response: no double
        assert transport.cancels == 1
        lane.on_response_done()
        lane.on_response_created("r2")
        lane.on_user_speech_started()
        assert transport.cancels == 2
        await lane.stop()


class TestMinimalOutbox:
    @pytest.mark.asyncio
    async def test_result_posts_exactly_once_to_bound_channel(self):
        from plugins.platforms.discord.realtime.outbox import DiscordTextOutbox

        sends = []

        class _Adapter:
            async def send(self, chat_id, content, **kw):
                sends.append((chat_id, content))
                return SimpleNamespace(success=True, message_id="m1")

        lane = SimpleNamespace(_loop=asyncio.get_running_loop())
        outbox = DiscordTextOutbox(adapter=_Adapter(), lane=lane)
        record = SimpleNamespace(dispatch_id="disp-1", text_channel_id=222,
                                 goal="check deposits")
        result = {"status": "completed",
                  "final_response": "July deposits were $38M ([src](https://x))."}

        def _post():
            return outbox.post_result(record, result)

        ref1 = await asyncio.to_thread(_post)
        ref2 = await asyncio.to_thread(_post)
        assert ref1 == "m1"
        assert ref2 == "m1"          # idempotent — never a duplicate post
        assert len(sends) == 1
        assert sends[0][0] == "222"  # bound channel only
        assert "deposits" in sends[0][1]

    @pytest.mark.asyncio
    async def test_no_bound_channel_means_no_post(self):
        from plugins.platforms.discord.realtime.outbox import DiscordTextOutbox

        outbox = DiscordTextOutbox(adapter=None, lane=SimpleNamespace(_loop=None))
        record = SimpleNamespace(dispatch_id="d", text_channel_id=None, goal="g")
        assert outbox.post_result(record, {"final_response": "x"}) is None
