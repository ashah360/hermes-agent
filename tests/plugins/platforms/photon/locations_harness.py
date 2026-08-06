"""Shared harness for the Photon sidecar friend-location protocol tests (S8).

Runs the **actual sidecar scripts** (``sidecar/index.mjs`` +
``sidecar/locations.mjs``, byte-identical copies staged into a tempdir)
against a **true SDK boundary mock**: a fake ``spectrum-ts`` package whose
iMessage runtime exposes the same ``advanced-imessage`` client surface the
real SDK does (``app.__internal.platforms → RemoteClient[] →
client.locations`` with ``get``/``watch``/``_onHeartbeat``, Date-typed
timestamps, ``NotFoundError`` with ``code == "sharedFriendLocationNotFound"``,
and a ``TypedEventStream``-shaped stream with ``close()`` interrupt
semantics).

The mock is driven through files in ``MOCK_SPECTRUM_DIR``:

* ``get-<hex(address)>.json``    — canned ``locations.get`` response
* ``feed-<hex(address)>.ndjson`` — script for ``locations.watch`` streams
  (``update`` / ``heartbeat`` / ``error`` / ``eof`` entries)
* ``diag.json``                  — mock-side observability (Spectrum app
  instantiations, watch open/close counts, updates pulled, addresses seen)

Protocol contract under test: ``plugins/platforms/photon/sidecar/LOCATIONS_PROTOCOL.md``.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx

REPO_SIDECAR_DIR = Path("plugins/platforms/photon/sidecar")
TOKEN = "test-sidecar-token-0123456789abcdef"
MAX_WATCHERS = 3

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# The Hermes patch marker — a dist chunk carrying it makes patchSpectrumTs()
# report "already patched" so the real patch script runs cleanly against the
# mock install.
_PATCH_MARKER = (
    "// Hermes patch: Preserve mixed text + attachment iMessage payloads\n"
)

# ---------------------------------------------------------------------------
# Mock spectrum-ts package (true SDK boundary mock)

_MOCK_PACKAGE_JSON = """\
{
  "name": "spectrum-ts",
  "version": "8.0.0",
  "type": "module",
  "exports": {
    ".": "./index.mjs",
    "./providers/imessage": "./providers/imessage.mjs"
  }
}
"""

_MOCK_INDEX_MJS = """\
// Mock spectrum-ts entry point for Hermes Photon sidecar tests.
import { locationsMock, diag, writeDiag } from "./state.mjs";

export async function Spectrum(options) {
  diag.spectrumInstances += 1;
  writeDiag();
  const client = [
    { phone: "+15550009999", client: { locations: locationsMock } },
  ];
  return {
    // Long-lived inbound stream: never yields, mirroring an idle line.
    messages: (async function* () {
      await new Promise(() => {});
    })(),
    async stop() {},
    __internal: { platforms: new Map([["iMessage", { client }]]) },
  };
}

const builder = (type) => (...args) => ({ type, args });
export const text = builder("text");
export const markdown = builder("markdown");
export const typing = builder("typing");
export const attachment = builder("attachment");
export const voice = builder("voice");
"""

_MOCK_IMESSAGE_MJS = """\
// Mock spectrum-ts iMessage provider for Hermes Photon sidecar tests.
function makeSpace(id) {
  return {
    id,
    type: "dm",
    phone: "+15550009999",
    async send(content) {
      return { id: "mock-msg-1" };
    },
    async getMessage(messageId) {
      return null;
    },
  };
}

export function imessage(app) {
  return {
    space: {
      async create(target) {
        return makeSpace(typeof target === "string" ? `any;-;${target}` : "sp");
      },
      async get(id) {
        return makeSpace(id);
      },
    },
  };
}
imessage.config = () => ({ __mock: true });
"""

_MOCK_STATE_MJS = """\
// Shared state for the mock spectrum-ts package: the advanced-imessage
// locations resource boundary, plus file-driven scripting + diagnostics.
import fs from "node:fs";
import path from "node:path";

const dir = process.env.MOCK_SPECTRUM_DIR;
if (!dir) {
  throw new Error("MOCK_SPECTRUM_DIR must be set for the spectrum-ts mock");
}

export const diag = {
  spectrumInstances: 0,
  getCalls: 0,
  watchCalls: 0,
  watchClosed: 0,
  updatesPulled: 0,
  heartbeatsSent: 0,
  addresses: [],
};

export function writeDiag() {
  const tmp = path.join(dir, "diag.json.tmp");
  fs.writeFileSync(tmp, JSON.stringify(diag));
  fs.renameSync(tmp, path.join(dir, "diag.json"));
}

function hexAddr(address) {
  return Buffer.from(String(address ?? "all"), "utf8").toString("hex");
}

// Mirror the real SDK: locationTimestamp / expiresAt are Date objects on the
// SDK-facing type (scripted here as epoch milliseconds).
function reviveLocation(raw) {
  const loc = { ...raw };
  if (typeof loc.locationTimestamp === "number") {
    loc.locationTimestamp = new Date(loc.locationTimestamp);
  }
  if (typeof loc.expiresAt === "number") {
    loc.expiresAt = new Date(loc.expiresAt);
  }
  return loc;
}

function mockError(spec) {
  const err = new Error(spec.message || "mock error");
  if (spec.name) err.name = spec.name;
  if (spec.code) err.code = spec.code;
  return err;
}

// Process-scoped source sequence, like the real SDK's Watch* streams. Starts
// far from 1 so a passed-through SDK sequence can never masquerade as a
// connection-scoped one in assertions.
let processSequence = 1000;

export const locationsMock = {
  // The real advanced-imessage resources hold the client-level heartbeat
  // callback in this private field, read at watch() call time.
  _onHeartbeat: undefined,

  async get(address) {
    diag.getCalls += 1;
    diag.addresses.push(String(address));
    writeDiag();
    const file = path.join(dir, `get-${hexAddr(address)}.json`);
    if (!fs.existsSync(file)) {
      const err = new Error(`no shared friend location for ${address}`);
      err.name = "NotFoundError";
      err.code = "sharedFriendLocationNotFound";
      throw err;
    }
    const spec = JSON.parse(fs.readFileSync(file, "utf8"));
    if (spec.error) throw mockError(spec.error);
    return reviveLocation(spec.location);
  },

  watch(address) {
    diag.watchCalls += 1;
    diag.addresses.push(String(address));
    writeDiag();
    return makeStream(path.join(dir, `feed-${hexAddr(address)}.ndjson`));
  },
};

// TypedEventStream-shaped mock: async iterable + close() that interrupts a
// pending next(), exactly like the real SDK's cancel promise.
function makeStream(feedPath) {
  let closed = false;
  let cancelResolve = () => {};
  const cancelPromise = new Promise((resolve) => {
    cancelResolve = resolve;
  });
  let consumed = 0;

  function readNewEntries() {
    let raw = "";
    try {
      raw = fs.readFileSync(feedPath, "utf8");
    } catch {
      return [];
    }
    let lines = raw.split("\\n");
    if (!raw.endsWith("\\n")) lines = lines.slice(0, -1); // partial write guard
    lines = lines.filter((line) => line.trim().length > 0);
    const fresh = lines.slice(consumed);
    consumed = lines.length;
    return fresh.map((line) => JSON.parse(line));
  }

  const gen = (async function* () {
    while (!closed) {
      for (const entry of readNewEntries()) {
        if (closed) return;
        if (entry.kind === "heartbeat") {
          diag.heartbeatsSent += 1;
          writeDiag();
          try {
            locationsMock._onHeartbeat?.();
          } catch {}
          continue;
        }
        if (entry.kind === "error") throw mockError(entry.error || {});
        if (entry.kind === "eof") return;
        if (entry.kind === "update") {
          processSequence += 1;
          diag.updatesPulled += 1;
          writeDiag();
          yield {
            sourceSequence: processSequence,
            location: reviveLocation(entry.location || {}),
          };
        }
      }
      await Promise.race([
        new Promise((resolve) => setTimeout(resolve, 20)),
        cancelPromise,
      ]);
    }
  })();

  return {
    [Symbol.asyncIterator]() {
      return gen;
    },
    async close() {
      if (closed) return;
      closed = true;
      cancelResolve();
      diag.watchClosed += 1;
      writeDiag();
      try {
        await gen.return(undefined);
      } catch {}
    },
  };
}
"""


# ---------------------------------------------------------------------------
# Harness


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SidecarHarness:
    """One live sidecar process (real scripts) over one mock SDK install."""

    def __init__(self, sidecar_dir: Path, mock_dir: Path, port: int) -> None:
        self.sidecar_dir = sidecar_dir
        self.mock_dir = mock_dir
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: Optional[subprocess.Popen] = None
        self._output: List[str] = []
        self._output_lock = threading.Lock()

    # -- process ---------------------------------------------------------

    def start(self) -> None:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PHOTON_PROJECT_ID": "mock-project",
            "PHOTON_PROJECT_SECRET": "mock-secret",
            "PHOTON_SIDECAR_PORT": str(self.port),
            "PHOTON_SIDECAR_TOKEN": TOKEN,
            "PHOTON_MAX_LOCATION_WATCHERS": str(MAX_WATCHERS),
            "MOCK_SPECTRUM_DIR": str(self.mock_dir),
        }
        self.proc = subprocess.Popen(  # noqa: S603
            ["node", str(self.sidecar_dir / "index.mjs")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._pump_output, daemon=True).start()
        deadline = time.time() + 20.0
        last_error: Optional[Exception] = None
        with httpx.Client(timeout=2.0) as client:
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise AssertionError(
                        f"sidecar exited early (code {self.proc.returncode}):\n"
                        + self.output()
                    )
                try:
                    resp = client.post(
                        f"{self.base}/healthz", headers=self.headers()
                    )
                    if resp.status_code == 200:
                        return
                except httpx.RequestError as exc:
                    last_error = exc
                time.sleep(0.1)
        raise AssertionError(
            f"sidecar did not become ready: {last_error}\n{self.output()}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(f"{self.base}/shutdown", headers=self.headers())
        except httpx.RequestError:
            pass
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5.0)

    def _pump_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._output_lock:
                self._output.append(line)

    def output(self) -> str:
        with self._output_lock:
            return "".join(self._output)

    # -- HTTP ------------------------------------------------------------

    def headers(self, token: Optional[str] = TOKEN) -> Dict[str, str]:
        if token is None:
            return {}
        return {"X-Hermes-Sidecar-Token": token}

    def post(
        self,
        path: str,
        *,
        json_body: Any = None,
        raw: Optional[str] = None,
        token: Optional[str] = TOKEN,
        timeout: float = 10.0,
    ) -> httpx.Response:
        with httpx.Client(timeout=timeout) as client:
            if raw is not None:
                return client.post(
                    f"{self.base}{path}",
                    content=raw.encode("utf-8"),
                    headers={
                        **self.headers(token),
                        "Content-Type": "application/json",
                    },
                )
            return client.post(
                f"{self.base}{path}", json=json_body, headers=self.headers(token)
            )

    @contextlib.contextmanager
    def watch(
        self,
        address: str,
        *,
        token: Optional[str] = TOKEN,
        read_timeout: float = 10.0,
    ) -> Iterator[httpx.Response]:
        """Open a /locations/watch NDJSON stream.

        CAUTION: bind ``resp.iter_lines()`` to a variable that lives as long
        as the connection should. httpx ties response finalization to that
        generator — dropping it lets GC close the response, and the sidecar
        (correctly) treats that as a consumer disconnect.
        """
        with httpx.Client(
            timeout=httpx.Timeout(5.0, read=read_timeout)
        ) as client:
            with client.stream(
                "POST",
                f"{self.base}/locations/watch",
                json={"address": address},
                headers=self.headers(token),
            ) as resp:
                yield resp

    # -- mock scripting ----------------------------------------------------

    def _hex(self, address: str) -> str:
        return address.encode("utf-8").hex()

    def set_get_response(self, address: str, spec: Dict[str, Any]) -> None:
        path = self.mock_dir / f"get-{self._hex(address)}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")

    def feed(self, address: str, entries: List[Dict[str, Any]]) -> None:
        path = self.mock_dir / f"feed-{self._hex(address)}.ndjson"
        with path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
            fh.flush()

    def feed_update(self, address: str, **location: Any) -> None:
        payload = {
            "address": address,
            "isLocatingInProgress": False,
            "locationType": "live",
            **location,
        }
        self.feed(address, [{"kind": "update", "location": payload}])

    def diag(self) -> Dict[str, Any]:
        path = self.mock_dir / "diag.json"
        for _ in range(100):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.02)
        raise AssertionError("mock diag.json unavailable")

    def wait_diag(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        timeout: float = 8.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.diag()
            if predicate(data):
                return data
            time.sleep(0.02)
        raise AssertionError(
            f"diag condition not met within {timeout}s: {self.diag()}"
        )

    def healthz(self) -> Dict[str, Any]:
        return self.post("/healthz", json_body={}).json()

    def wait_active_watchers(self, count: int, timeout: float = 8.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.healthz()
            locations = data.get("locations") or {}
            if locations.get("activeWatchers") == count:
                return
            time.sleep(0.05)
        raise AssertionError(
            f"activeWatchers never reached {count}: {self.healthz()}"
        )


def make_sidecar_harness(root: Path) -> SidecarHarness:
    """Stage the actual sidecar scripts + mock SDK under *root* and boot."""
    sidecar_dir = root / "sidecar"
    sidecar_dir.mkdir()

    staged = list(REPO_SIDECAR_DIR.glob("*.mjs"))
    assert staged, "sidecar scripts not found"
    for script in staged:
        shutil.copy(script, sidecar_dir / script.name)

    # Minimal @spectrum-ts/imessage dist chunk so the real runtime patch
    # (patch-spectrum-mixed-attachments.mjs) short-circuits as "already
    # patched" instead of failing the boot.
    dist = sidecar_dir / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.js").write_text(_PATCH_MARKER, encoding="utf-8")

    mock_pkg = sidecar_dir / "node_modules" / "spectrum-ts"
    (mock_pkg / "providers").mkdir(parents=True)
    (mock_pkg / "package.json").write_text(_MOCK_PACKAGE_JSON, encoding="utf-8")
    (mock_pkg / "index.mjs").write_text(_MOCK_INDEX_MJS, encoding="utf-8")
    (mock_pkg / "state.mjs").write_text(_MOCK_STATE_MJS, encoding="utf-8")
    (mock_pkg / "providers" / "imessage.mjs").write_text(
        _MOCK_IMESSAGE_MJS, encoding="utf-8"
    )

    mock_dir = root / "mock-state"
    mock_dir.mkdir()

    harness = SidecarHarness(sidecar_dir, mock_dir, _free_port())
    harness.start()
    return harness


# ---------------------------------------------------------------------------
# Shared assertions / frame helpers


def iso_ms(epoch_ms: int) -> str:
    """Epoch ms → the exact string JS ``Date.prototype.toISOString`` emits."""
    dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{epoch_ms % 1000:03d}Z"


def read_frame(lines: Iterator[str]) -> Dict[str, Any]:
    return json.loads(next(lines))


def read_non_heartbeat(lines: Iterator[str]) -> Dict[str, Any]:
    while True:
        frame = read_frame(lines)
        if frame.get("type") != "heartbeat":
            return frame


def assert_no_sdk_addresses_in_logs(harness: SidecarHarness) -> None:
    """No handle the SDK boundary ever saw may appear in sidecar logs."""
    time.sleep(0.3)
    log_text = harness.output()
    addresses = {
        a for a in harness.diag()["addresses"] if a and a != "undefined"
    }
    assert addresses, "expected the SDK mock to have seen addresses"
    for address in addresses:
        assert address not in log_text, f"handle leaked into sidecar logs"
