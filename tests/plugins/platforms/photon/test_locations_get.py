"""Photon sidecar S8 — auth, request-shape validation, and ``/locations/get``.

See ``locations_harness.py`` for the SDK-boundary mock and process harness,
and ``sidecar/LOCATIONS_PROTOCOL.md`` for the contract.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from typing import Iterator

import httpx
import pytest

from tests.plugins.platforms.photon.locations_harness import (
    MAX_WATCHERS,
    SidecarHarness,
    assert_no_sdk_addresses_in_logs,
    iso_ms,
    make_sidecar_harness,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required for sidecar tests"
)


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SidecarHarness]:
    harness = make_sidecar_harness(tmp_path_factory.mktemp("photon-loc-get"))
    try:
        yield harness
    finally:
        harness.stop()


# ---------------------------------------------------------------------------
# Auth + request shape


def test_locations_get_requires_token(sidecar: SidecarHarness) -> None:
    resp = sidecar.post(
        "/locations/get", json_body={"address": "+15550000001"}, token=None
    )
    assert resp.status_code == 401
    resp = sidecar.post(
        "/locations/get", json_body={"address": "+15550000001"}, token="wrong"
    )
    assert resp.status_code == 401


def test_locations_watch_requires_token(sidecar: SidecarHarness) -> None:
    resp = sidecar.post(
        "/locations/watch", json_body={"address": "+15550000001"}, token=None
    )
    assert resp.status_code == 401


def test_locations_rejects_non_post(sidecar: SidecarHarness) -> None:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(
            f"{sidecar.base}/locations/get", headers=sidecar.headers()
        )
        assert resp.status_code == 405
        resp = client.get(
            f"{sidecar.base}/locations/watch", headers=sidecar.headers()
        )
        assert resp.status_code == 405


def test_locations_address_in_query_is_not_accepted(
    sidecar: SidecarHarness,
) -> None:
    """The address travels in the body — a query string is never a target."""
    resp = sidecar.post("/locations/get?address=%2B15550000001", json_body={})
    assert resp.status_code in (400, 404)
    assert resp.json().get("ok") is not True
    resp = sidecar.post("/locations/watch?address=%2B15550000001", json_body={})
    assert resp.status_code in (400, 404)
    assert resp.json().get("ok") is not True


@pytest.mark.parametrize(
    "raw_body",
    [
        "",  # empty body
        "not json",  # invalid JSON
        "[]",  # not an object
        '"+15550000001"',  # bare string
        "{}",  # missing address
        '{"address": 5}',  # wrong type
        '{"address": ""}',  # empty
        '{"address": "not a handle"}',  # invalid format
        '{"address": "+123"}',  # too short for E.164
        '{"address": "+' + "1" * 40 + '"}',  # too long for E.164
        '{"address": "+15550000001", "extra": true}',  # unknown key
        '{"address": "alice@"}',  # malformed email
    ],
)
def test_locations_get_rejects_malformed_bodies(
    sidecar: SidecarHarness, raw_body: str
) -> None:
    resp = sidecar.post("/locations/get", raw=raw_body)
    assert resp.status_code == 400, raw_body
    body = resp.json()
    assert body["ok"] is False
    # The rejected input must never be echoed back.
    if raw_body not in ("", "{}", "[]"):
        assert "15550000001" not in body.get("error", "")


def test_locations_watch_rejects_malformed_body(sidecar: SidecarHarness) -> None:
    resp = sidecar.post("/locations/watch", raw='{"address": 42}')
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_locations_get_rejects_oversized_body(sidecar: SidecarHarness) -> None:
    raw = '{"address": "' + "a" * 8192 + '"}'
    resp = sidecar.post("/locations/get", raw=raw)
    assert resp.status_code == 413
    assert resp.json()["ok"] is False


# ---------------------------------------------------------------------------
# /locations/get snapshots


def test_locations_get_full_fidelity(sidecar: SidecarHarness) -> None:
    address = "+15551110001"
    ts_ms = 1754400000123
    expires_ms = 1754500000456
    sidecar.set_get_response(
        address,
        {
            "location": {
                "address": address,
                "name": "Alice Example",
                "latitude": 37.774929,
                "longitude": -122.419416,
                "accuracy": 12.5,
                "longAddress": "1 Test Way, San Francisco, CA",
                "shortAddress": "SoMa",
                "isLocatingInProgress": False,
                "locationType": "live",
                "locationTimestamp": ts_ms,
                "expiresAt": expires_ms,
            }
        },
    )
    resp = sidecar.post("/locations/get", json_body={"address": address})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    loc = body["location"]
    assert loc["address"] == address
    assert loc["name"] == "Alice Example"
    assert loc["latitude"] == pytest.approx(37.774929)
    assert loc["longitude"] == pytest.approx(-122.419416)
    assert loc["accuracy"] == pytest.approx(12.5)
    assert loc["longAddress"] == "1 Test Way, San Francisco, CA"
    assert loc["shortAddress"] == "SoMa"
    assert loc["isLocatingInProgress"] is False
    assert loc["locationType"] == "live"
    # SDK Date objects serialize as ISO-8601 strings.
    assert loc["locationTimestamp"] == iso_ms(ts_ms)
    assert loc["expiresAt"] == iso_ms(expires_ms)


def test_locations_get_email_handle(sidecar: SidecarHarness) -> None:
    address = "friend.one@example.com"
    sidecar.set_get_response(
        address,
        {
            "location": {
                "address": address,
                "isLocatingInProgress": True,
                "locationType": "shallow",
            }
        },
    )
    resp = sidecar.post("/locations/get", json_body={"address": address})
    assert resp.status_code == 200
    loc = resp.json()["location"]
    assert loc["address"] == address
    assert loc["isLocatingInProgress"] is True
    # Coordinates absent while locating — no fabricated fields.
    assert loc.get("latitude") is None


def test_locations_get_not_sharing_returns_null(sidecar: SidecarHarness) -> None:
    # No canned response → the mock throws NotFoundError with
    # code == sharedFriendLocationNotFound, mirroring the real SDK.
    resp = sidecar.post("/locations/get", json_body={"address": "+15551110002"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["location"] is None


def test_locations_get_generic_not_found_is_upstream_error(
    sidecar: SidecarHarness,
) -> None:
    """Only the exact ``sharedFriendLocationNotFound`` code means "not
    sharing". Any other NotFoundError (or one with no code) is an upstream
    failure — mapping it to a null location would fabricate an answer."""
    cases = [
        ("+15551110004", {"name": "NotFoundError", "code": "chatNotFound"}),
        ("+15551110005", {"name": "NotFoundError"}),  # code missing
    ]
    for address, error in cases:
        sidecar.set_get_response(
            address,
            {"error": {**error, "message": f"not found: {address}"}},
        )
        resp = sidecar.post("/locations/get", json_body={"address": address})
        assert resp.status_code == 502, (address, error)
        body = resp.json()
        assert body["ok"] is False
        assert body.get("location") is None
        assert address not in json.dumps(body)
    time.sleep(0.3)
    log_text = sidecar.output()
    for address, _ in cases:
        assert address not in log_text


def test_locations_get_upstream_error_is_generic_and_leak_free(
    sidecar: SidecarHarness,
) -> None:
    address = "+15551110003"
    sidecar.set_get_response(
        address,
        {
            "error": {
                "name": "ConnectionError",
                "code": "networkError",
                "message": f"boom contacting {address} at lat 11.22",
            }
        },
    )
    resp = sidecar.post("/locations/get", json_body={"address": address})
    assert resp.status_code == 502
    body = resp.json()
    assert body["ok"] is False
    assert address not in json.dumps(body)
    time.sleep(0.3)
    log_text = sidecar.output()
    assert address not in log_text
    assert "boom contacting" not in log_text


# ---------------------------------------------------------------------------
# Shared-owner + regression invariants


def test_single_spectrum_app_and_process(sidecar: SidecarHarness) -> None:
    """All location traffic must reuse the one Spectrum app in the one sidecar."""
    diag = sidecar.diag()
    assert diag["spectrumInstances"] == 1
    assert diag["getCalls"] >= 1
    if sys.platform != "win32":
        out = subprocess.run(  # noqa: S603, S607
            ["ps", "-eo", "args"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
        ).stdout
        marker = str(sidecar.sidecar_dir / "index.mjs")
        assert sum(1 for line in out.splitlines() if marker in line) == 1


def test_healthz_reports_location_watchers(sidecar: SidecarHarness) -> None:
    data = sidecar.healthz()
    assert data["ok"] is True
    assert "stream" in data  # existing health payload unchanged
    assert data["locations"]["activeWatchers"] == 0
    assert data["locations"]["maxWatchers"] == MAX_WATCHERS


def test_existing_message_endpoints_unchanged(sidecar: SidecarHarness) -> None:
    resp = sidecar.post(
        "/send", json_body={"spaceId": "+15559990000", "text": "hello"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["messageId"] == "mock-msg-1"


def test_no_request_targets_in_sidecar_logs(sidecar: SidecarHarness) -> None:
    assert_no_sdk_addresses_in_logs(sidecar)
    # Nor coordinates from any canned payload.
    assert "37.774929" not in sidecar.output()
