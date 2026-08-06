"""Photon sidecar S8 — ``/locations/watch`` concurrency and backpressure.

Covers: independent per-connection epochs/sequences for concurrent
consumers, the watcher cap (and slot release), and slow-consumer
backpressure propagating to the SDK stream instead of unbounded buffering.

See ``locations_harness.py`` for the SDK-boundary mock and process harness,
and ``sidecar/LOCATIONS_PROTOCOL.md`` for the contract.
"""
from __future__ import annotations

import contextlib
import shutil
import time
from typing import Iterator

import pytest

from tests.plugins.platforms.photon.locations_harness import (
    MAX_WATCHERS,
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
    harness = make_sidecar_harness(tmp_path_factory.mktemp("photon-loc-conc"))
    try:
        yield harness
    finally:
        harness.stop()


def test_watch_two_consumers_independent_epochs_and_sequences(
    sidecar: SidecarHarness,
) -> None:
    address = "+15552220003"
    with sidecar.watch(address) as resp_a, sidecar.watch(address) as resp_b:
        lines_a = resp_a.iter_lines()
        lines_b = resp_b.iter_lines()
        epoch_a = read_frame(lines_a)
        epoch_b = read_frame(lines_b)
        assert epoch_a["type"] == "epoch" and epoch_b["type"] == "epoch"
        assert epoch_a["connectionEpoch"] != epoch_b["connectionEpoch"]

        for i in range(2):
            sidecar.feed_update(address, name=f"shared fix {i}")
        updates_a = [read_non_heartbeat(lines_a) for _ in range(2)]
        updates_b = [read_non_heartbeat(lines_b) for _ in range(2)]

    for updates, epoch in ((updates_a, epoch_a), (updates_b, epoch_b)):
        assert [u["sourceSequence"] for u in updates] == [1, 2]
        assert all(
            u["connectionEpoch"] == epoch["connectionEpoch"] for u in updates
        )
        assert [u["location"]["name"] for u in updates] == [
            "shared fix 0",
            "shared fix 1",
        ]
    sidecar.wait_active_watchers(0)


def test_watch_watcher_cap(sidecar: SidecarHarness) -> None:
    with contextlib.ExitStack() as stack:
        # NB: every iter_lines() generator must be retained for the stack's
        # lifetime. httpx ties response finalization to the iterator — a
        # dropped generator is GC-finalized, which CLOSES its response and
        # the sidecar (correctly) reaps that watcher as disconnected.
        line_iters = []
        for i in range(MAX_WATCHERS):
            resp = stack.enter_context(sidecar.watch(f"+1555333000{i}"))
            line_iters.append(resp.iter_lines())
            assert read_frame(line_iters[-1])["type"] == "epoch"
        sidecar.wait_active_watchers(MAX_WATCHERS)
        overflow = sidecar.post(
            "/locations/watch", json_body={"address": "+15553330999"}
        )
        assert overflow.status_code == 429
        assert overflow.json()["ok"] is False
    # Slots must be released once the consumers disconnect.
    sidecar.wait_active_watchers(0)
    with sidecar.watch("+15553331000") as resp:
        lines = resp.iter_lines()
        assert read_frame(lines)["type"] == "epoch"
    sidecar.wait_active_watchers(0)


def test_watch_slow_consumer_backpressure(sidecar: SidecarHarness) -> None:
    """A stalled consumer must pause the SDK pull, not buffer unboundedly."""
    address = "+15554440001"
    total = 100
    padding = "x" * 262_144  # 256 KiB per update ⇒ ~25 MiB total
    base_pulled = sidecar.diag()["updatesPulled"]
    sidecar.feed(
        address,
        [
            {
                "kind": "update",
                "location": {
                    "address": address,
                    "name": f"big {i}",
                    "longAddress": padding,
                    "isLocatingInProgress": False,
                    "locationType": "live",
                },
            }
            for i in range(total)
        ],
    )
    with sidecar.watch(address, read_timeout=60.0) as resp:
        # Do NOT read the body yet — let socket buffers fill.
        deadline = time.time() + 15.0
        last = -1
        stable_since = time.time()
        while time.time() < deadline:
            pulled = sidecar.diag()["updatesPulled"] - base_pulled
            if pulled != last:
                last = pulled
                stable_since = time.time()
            elif time.time() - stable_since > 1.0 and pulled > 0:
                break
            time.sleep(0.05)
        stalled_at = sidecar.diag()["updatesPulled"] - base_pulled
        assert 0 < stalled_at < total, (
            f"sidecar pulled {stalled_at}/{total} updates while the consumer "
            "was stalled — backpressure is not propagating to the SDK stream"
        )
        # Now drain: every update must arrive, in order, exactly once.
        lines = resp.iter_lines()
        assert read_frame(lines)["type"] == "epoch"
        sequences = []
        for _ in range(total):
            frame = read_non_heartbeat(lines)
            assert frame["type"] == "update"
            sequences.append(frame["sourceSequence"])
        assert sequences == list(range(1, total + 1))
    sidecar.wait_active_watchers(0)


def test_single_spectrum_app_across_watchers(sidecar: SidecarHarness) -> None:
    diag = sidecar.diag()
    assert diag["spectrumInstances"] == 1
    assert diag["watchCalls"] >= 3


def test_no_request_targets_in_sidecar_logs(sidecar: SidecarHarness) -> None:
    assert_no_sdk_addresses_in_logs(sidecar)
