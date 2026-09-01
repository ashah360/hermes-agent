"""Lane selection & flag gating for the Discord realtime voice lane.

Contract (ADR docs/discord-realtime-voice-architecture.md, D1/D2):
- Flag off (default): the realtime package is never imported; the cascaded
  path owns audio (receiver silence-detection active, no frame sink).
- Flag on + lane start failure (e.g. missing secret): join proceeds on the
  cascaded lane exactly as today — never a hard failure, never two consumers.
- Flag on + lane start success: the lane takes single ownership — the
  receiver runs in frame-tap mode and the silence path is inert.
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_adapter():
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
    return adapter


def _make_receiver():
    from plugins.platforms.discord.adapter import VoiceReceiver

    mock_vc = MagicMock()
    mock_vc._connection.secret_key = [0] * 32
    mock_vc._connection.dave_session = None
    mock_vc._connection.ssrc = 9999
    mock_vc._connection.hook = None
    return VoiceReceiver(mock_vc)


class TestFlagGate:
    def test_disabled_by_default_and_no_import(self):
        adapter = _make_adapter()
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert adapter._realtime_voice_enabled() is False
            lane = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                adapter._start_realtime_lane_if_enabled(111, MagicMock(), _make_receiver())
            )
        assert lane is None
        # The whole point of the lazy gate: disabled deployments never even
        # import the realtime package.
        assert not any(
            m.startswith("plugins.platforms.discord.realtime") or m == "realtime"
            for m in sys.modules
        )

    def test_enabled_flag_reads_gateway_platforms_path(self):
        adapter = _make_adapter()
        cfg = {
            "gateway": {
                "platforms": {
                    "discord": {"voice": {"realtime": {"enabled": True}}}
                }
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert adapter._realtime_voice_enabled() is True

    def test_enabled_flag_reads_top_level_platforms_path(self):
        # gateway/config.py also resolves a top-level ``platforms`` map.
        adapter = _make_adapter()
        cfg = {"platforms": {"discord": {"voice": {"realtime": {"enabled": True}}}}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert adapter._realtime_voice_enabled() is True


class TestLaneStartFallback:
    @pytest.mark.asyncio
    async def test_start_failure_falls_back_to_cascaded(self, monkeypatch):
        # Enabled, but no API secret resolvable in the hermetic env: the lane
        # must fail BEFORE any turn and the join must proceed cascaded.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = _make_adapter()
        receiver = _make_receiver()
        cfg = {
            "gateway": {
                "platforms": {
                    "discord": {"voice": {"realtime": {"enabled": True}}}
                }
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            lane = await adapter._start_realtime_lane_if_enabled(
                111, MagicMock(), receiver
            )
        assert lane is None
        assert getattr(adapter, "_realtime_lanes", {}).get(111) is None
        # Cascaded lane still owns audio: no frame sink installed.
        assert receiver._frame_sink is None

    @pytest.mark.asyncio
    async def test_start_success_takes_single_ownership(self):
        adapter = _make_adapter()
        receiver = _make_receiver()
        cfg = {
            "gateway": {
                "platforms": {
                    "discord": {"voice": {"realtime": {"enabled": True}}}
                }
            }
        }

        class _FakeLane:
            def __init__(self):
                self.frames = []

            async def start(self):
                return True

            def on_input_frame(self, user_id, ssrc, pcm):
                self.frames.append((user_id, ssrc, pcm))

        fake_lane = _FakeLane()
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            import plugins.platforms.discord.realtime as rt

            with patch.object(rt, "create_lane", return_value=fake_lane):
                lane = await adapter._start_realtime_lane_if_enabled(
                    111, MagicMock(), receiver
                )
        assert lane is fake_lane
        assert adapter._realtime_lanes[111] is fake_lane
        # Frame-tap mode: sink installed → silence path inert.
        assert receiver._frame_sink is not None


class TestFrameTapSingleConsumer:
    def test_frame_sink_bypasses_silence_buffers(self):
        receiver = _make_receiver()
        sink_frames = []
        receiver.set_frame_sink(lambda user_id, ssrc, pcm: sink_frames.append(pcm))
        receiver.map_ssrc(100, 42)

        receiver._deliver_pcm(100, b"\x01\x02" * 960)

        assert sink_frames == [b"\x01\x02" * 960]
        # Never duplicated into the cascaded buffers.
        assert len(receiver._buffers) == 0
        assert receiver.check_silence() == []

    def test_without_sink_pcm_buffers_as_today(self):
        receiver = _make_receiver()
        receiver.map_ssrc(100, 42)
        receiver._deliver_pcm(100, b"\x01\x02" * 960)
        assert len(receiver._buffers[100]) == 1920 * 2 // 2  # bytes retained


class TestLaneStateMachine:
    def test_states_and_config_defaults(self):
        from plugins.platforms.discord.realtime.config import (
            load_realtime_voice_config,
        )
        from plugins.platforms.discord.realtime.lane import LaneState

        cfg = load_realtime_voice_config({})
        assert cfg.enabled is False
        assert cfg.model == "gpt-realtime-2.1"
        assert cfg.voice == "cedar"
        assert cfg.reasoning_effort == "low"
        assert cfg.api_key_secret == "OPENAI_API_KEY"
        assert cfg.rollover_margin_seconds == 300
        assert cfg.max_inflight_dispatches == 3
        assert cfg.synthesis_tts_model == "gpt-4o-mini-tts"
        assert cfg.synthesis_voice == "cedar"
        assert cfg.synthesis_framing is True
        assert {s.name for s in LaneState} == {"CASCADED", "REALTIME", "DEMOTING"}

    def test_config_overrides_parse(self):
        from plugins.platforms.discord.realtime.config import (
            load_realtime_voice_config,
        )

        cfg = load_realtime_voice_config(
            {
                "enabled": True,
                "voice": "cedar",
                "connect_timeout_seconds": 3,
                "synthesis": {"voice": "cedar", "framing": False},
            }
        )
        assert cfg.enabled is True
        assert cfg.connect_timeout_seconds == 3
        assert cfg.synthesis_framing is False
