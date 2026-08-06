"""Photon sidecar S8 — ``/locations/watch`` stream lifecycle.

Covers: epoch-first framing, connection-scoped sequences, SDK-liveness
heartbeats, abort cleanup (cancelled subscription, released slot), and
EOF/error termination without synthetic frames.

See ``locations_harness.py`` for the SDK-boundary mock and process harness,
and ``sidecar/LOCATIONS_PROTOCOL.md`` for the contract.
"""
from __future__ import annotations

import shutil
import time
from typing import Iterator

import pytest

from tests.plugins.platforms.photon.locations_harness import (
    UUID_RE,
    SidecarHarness,
    assert_no_sdk_addresses_in_logs,
    make_sidecar_harness,
    read_frame,
    read_non_heartbeat,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required for sidecar tests"
)


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SidecarHarness]:
    harness = make_sidecar_harness(tmp_path_factory.mktemp("photon-loc-watch"))
    try:
        yield harness
    finally:
        harness.stop()


def test_watch_epoch_first_then_scoped_updates(sidecar: SidecarHarness) -> None:
    address = "+15552220001"
    with sidecar.watch(address) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = resp.iter_lines()

        epoch_frame = read_frame(lines)
        assert epoch_frame["type"] == "epoch"
        assert UUID_RE.match(epoch_frame["connectionEpoch"])
        assert isinstance(epoch_frame["startedAtMs"], int)
        assert abs(epoch_frame["startedAtMs"] - time.time() * 1000) < 60_000

        for i in range(3):
            sidecar.feed_update(
                address,
                name=f"fix {i}",
                latitude=40.0 + i,
                longitude=-70.0 - i,
            )
        updates = [read_non_heartbeat(lines) for _ in range(3)]

    for i, frame in enumerate(updates):
        assert frame["type"] == "update"
        assert frame["connectionEpoch"] == epoch_frame["connectionEpoch"]
        # Connection-scoped sequence: starts at 1, increments by 1 — NOT the
        # SDK's process-scoped counter (the mock starts that at 1000).
        assert frame["sourceSequence"] == i + 1
        assert frame["location"]["name"] == f"fix {i}"
        assert frame["location"]["latitude"] == pytest.approx(40.0 + i)


def test_watch_heartbeats_are_liveness_only(sidecar: SidecarHarness) -> None:
    address = "+15552220002"
    with sidecar.watch(address) as resp:
        lines = resp.iter_lines()
        epoch_frame = read_frame(lines)
        assert epoch_frame["type"] == "epoch"

        sidecar.feed(address, [{"kind": "heartbeat"}])
        frame = read_frame(lines)
        assert frame["type"] == "heartbeat"
        assert frame["connectionEpoch"] == epoch_frame["connectionEpoch"]
        assert isinstance(frame["atMs"], int)
        # A heartbeat never fabricates a location payload.
        assert "location" not in frame

        # Heartbeats must not consume sequence numbers.
        sidecar.feed(address, [{"kind": "heartbeat"}])
        sidecar.feed_update(address, locationType="legacy")
        update = read_non_heartbeat(lines)
        assert update["type"] == "update"
        assert update["sourceSequence"] == 1


def test_watch_abort_cancels_subscription_and_releases_slot(
    sidecar: SidecarHarness,
) -> None:
    address = "+15552220004"
    before = sidecar.diag()
    with sidecar.watch(address) as resp:
        lines = resp.iter_lines()
        assert read_frame(lines)["type"] == "epoch"
        sidecar.feed_update(address)
        assert read_non_heartbeat(lines)["sourceSequence"] == 1
        sidecar.wait_active_watchers(1)
    # Consumer aborted: the sidecar must close the SDK stream and release the
    # watcher slot — no leaked tasks or listeners.
    sidecar.wait_diag(lambda d: d["watchClosed"] >= before["watchClosed"] + 1)
    sidecar.wait_active_watchers(0)


def test_watch_sdk_eof_ends_stream(sidecar: SidecarHarness) -> None:
    address = "+15552220005"
    before = sidecar.diag()
    with sidecar.watch(address) as resp:
        lines = resp.iter_lines()
        assert read_frame(lines)["type"] == "epoch"
        sidecar.feed(address, [{"kind": "eof"}])
        # The response must end cleanly (EOF) with no synthetic frames —
        # Presence synthesizes the subscription end from EOF.
        remaining = list(lines)
    assert remaining == []
    sidecar.wait_diag(lambda d: d["watchClosed"] >= before["watchClosed"] + 1)
    sidecar.wait_active_watchers(0)


def test_watch_sdk_error_ends_stream_without_leaking(
    sidecar: SidecarHarness,
) -> None:
    address = "+15552220006"
    with sidecar.watch(address) as resp:
        lines = resp.iter_lines()
        assert read_frame(lines)["type"] == "epoch"
        sidecar.feed(
            address,
            [
                {
                    "kind": "error",
                    "error": {
                        "name": "ConnectionError",
                        "code": "networkError",
                        "message": f"stream broke for {address}",
                    },
                }
            ],
        )
        remaining = list(lines)
    # No error/location payload is fabricated — the stream just ends.
    assert remaining == []
    sidecar.wait_active_watchers(0)
    time.sleep(0.3)
    log_text = sidecar.output()
    assert address not in log_text
    assert "stream broke for" not in log_text


def test_no_request_targets_in_sidecar_logs(sidecar: SidecarHarness) -> None:
    assert_no_sdk_addresses_in_logs(sidecar)
