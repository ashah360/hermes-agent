"""Gateway voice-join ordering vs. realtime lane construction (live bug).

`_handle_voice_channel_join` used to set `adapter._voice_text_channels` /
`_voice_sources` only AFTER `adapter.join_voice_channel(...)` returned — but
the realtime lane (and its WorkerBridge/outbox/projection) is constructed
INSIDE the join. Production lanes therefore saw `text_channel_id=None`:
principal chat_id empty, outbox unbound.

Contracts:
- the binding is established BEFORE the adapter join starts, so
  `create_lane` sees the exact source;
- join failure or exception rolls the prebound entries back to their PRIOR
  values (an existing valid binding is never blindly deleted);
- success retains the binding and legacy callback/voice-mode behavior;
- the principal provider carries the CURRENT requesting voice user at
  build time (reads `lane.last_speaker_user_id`), never a 0 captured at
  join forever; the text channel stays fixed to the bound channel.

These tests run the REAL GatewayRunner handler, not a reimplementation.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class RecordingRunnerStub:
    """Fake gateway_runner for the principal factory: records build kwargs."""

    def __init__(self):
        self.calls = []

    def build_realtime_worker_principal(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(model="m", api_key="k", **kwargs)


class FakeJoinAdapter:
    """Adapter double whose join constructs the REAL production lane."""

    def __init__(self, *, join_result=True, join_raises=False):
        self._voice_text_channels = {}
        self._voice_sources = {}
        self._voice_input_callback = None
        self._on_voice_disconnect = None
        self._voice_mode_getter = None
        self.gateway_runner = RecordingRunnerStub()
        self._client = MagicMock()
        self.lane = None
        self.binding_at_join = None
        self._join_result = join_result
        self._join_raises = join_raises

    async def get_user_voice_channel(self, guild_id, user_id):
        channel = MagicMock()
        channel.name = "war-room"
        channel.guild.id = guild_id
        return channel

    async def join_voice_channel(self, channel, **kwargs):
        guild_id = channel.guild.id
        # What production create_lane() sees DURING the join:
        self.binding_at_join = self._voice_text_channels.get(guild_id)
        if self._join_raises:
            raise RuntimeError("connect blew up")
        if not self._join_result:
            return False
        from plugins.platforms.discord.realtime.lane import create_lane

        self.lane = create_lane(
            adapter=self,
            guild_id=guild_id,
            voice_client=MagicMock(),
            receiver=None,
            raw_config={"enabled": True},
        )
        return True


def _make_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {}
    runner._save_voice_modes = lambda: None
    runner._set_adapter_auto_tts_enabled = lambda *a, **k: None
    runner._set_adapter_auto_tts_disabled = lambda *a, **k: None
    runner._adapter_for_source = lambda source: adapter
    runner._get_guild_id = lambda event: 111
    return runner


def _make_event():
    return MessageEvent(
        source=SessionSource(
            platform=Platform.DISCORD, chat_id="222", user_id="42",
            user_name="Shaan", chat_type="channel",
        ),
        text="/voice join",
        raw_message=SimpleNamespace(guild_id=111, guild=None),
    )


async def _run_join(runner, event):
    from gateway.run import GatewayRunner

    return await GatewayRunner._handle_voice_channel_join(runner, event)


class TestJoinOrdering:
    @pytest.mark.asyncio
    async def test_lane_constructed_during_join_sees_bound_channel(self):
        adapter = FakeJoinAdapter()
        runner = _make_runner(adapter)
        reply = await _run_join(runner, _make_event())
        assert "Joined" in reply
        # The binding existed BEFORE the adapter join ran…
        assert adapter.binding_at_join == 222
        # …so the production lane was constructed with the real channel.
        assert adapter.lane is not None
        assert adapter.lane.text_channel_id == 222
        assert adapter.lane.worker_bridge is not None
        # And it is retained after success.
        assert adapter._voice_text_channels[111] == 222
        assert adapter._voice_sources[111]["chat_id"] == "222"

    @pytest.mark.asyncio
    async def test_principal_factory_gets_bound_chat_and_current_speaker(self):
        adapter = FakeJoinAdapter()
        runner = _make_runner(adapter)
        await _run_join(runner, _make_event())
        lane = adapter.lane

        # No speaker yet: chat id bound, user id absent (not a stale 0).
        principal = lane.worker_bridge._principal_provider()
        call = adapter.gateway_runner.calls[-1]
        assert call["chat_id"] == "222"
        assert not call.get("user_id")

        # Speaker attribution is read at BUILD time, not captured at join.
        lane.last_speaker_user_id = 42
        lane.worker_bridge._principal_provider()
        assert adapter.gateway_runner.calls[-1]["user_id"] == "42"
        assert adapter.gateway_runner.calls[-1]["chat_id"] == "222"
        lane.last_speaker_user_id = 43
        lane.worker_bridge._principal_provider()
        assert adapter.gateway_runner.calls[-1]["user_id"] == "43"

    @pytest.mark.asyncio
    async def test_join_returning_false_rolls_back_to_prior_binding(self):
        adapter = FakeJoinAdapter(join_result=False)
        # An existing valid binding must never be blindly deleted.
        adapter._voice_text_channels[111] = 999
        adapter._voice_sources[111] = {"chat_id": "999"}
        runner = _make_runner(adapter)
        reply = await _run_join(runner, _make_event())
        assert "Failed" in reply
        assert adapter.binding_at_join == 222   # prebound during the attempt
        assert adapter._voice_text_channels[111] == 999  # rolled back
        assert adapter._voice_sources[111] == {"chat_id": "999"}

    @pytest.mark.asyncio
    async def test_join_raising_rolls_back_and_removes_fresh_binding(self):
        adapter = FakeJoinAdapter(join_raises=True)
        runner = _make_runner(adapter)
        reply = await _run_join(runner, _make_event())
        assert "Failed" in reply or "voice" in reply.lower()
        # No prior binding existed → the prebound entry is removed entirely.
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
