# Photon sidecar — read-only friend-location protocol (S8)

The long-running Photon sidecar (`index.mjs`) is the **sole owner** of the
Spectrum connection for this Hermes instance. This document specifies the
minimal read-only location protocol it exposes on top of that connection so
that a sibling local service (Hermes **Presence**) can query and watch
Find My friend locations **without launching a second Spectrum process or
gRPC client**.

Implementation: `locations.mjs` (service) + `index.mjs` (routing).
Consumer-side credential handoff: `../runtime_credentials.py`.

## Transport and authentication

- Loopback HTTP against the existing sidecar
  (`http://127.0.0.1:$PHOTON_SIDECAR_PORT`).
- Every request requires the existing shared-token header, unchanged:
  `X-Hermes-Sidecar-Token: <token>` (constant-time compare; 401 otherwise).
- Both endpoints are `POST` and take a JSON body. **The target handle
  travels in the body, never in the URL/query string** — URLs land in logs
  and proxies; handles are personal data. A query string on the path makes
  the route unmatched (404).

### Request body (both endpoints)

```json
{"address": "<E.164 phone or email>"}
```

Validation is strict:

- Body must be a JSON object with **exactly** the `address` key.
- `address` must be a string matching `^\+\d{6,15}$` (E.164) or a
  conservative email shape (≤ 64-char local part, dotted domain).
- Bodies over **4096 bytes** → `413`. Anything else malformed → `400`.
- Error responses are generic (`{"ok": false, "error": "..."}`) and never
  echo the submitted body or handle.

## `POST /locations/get`

Fetch the latest shared-location snapshot for one friend.

- `200 {"ok": true, "location": <SharedFriendLocation>}` — the SDK payload
  with full fidelity. Fields (all optional unless noted): `address`
  (required), `name`, `latitude`, `longitude`, `accuracy` (meters),
  `longAddress`, `shortAddress`, `isLocatingInProgress` (required bool),
  `locationType` (`"legacy" | "live" | "shallow" | "unknown"`, required),
  `locationTimestamp`, `expiresAt`. SDK `Date` values serialize as ISO-8601
  strings (`Date#toISOString`). Coordinates are absent while the device is
  still acquiring a fix — the sidecar never fabricates them.
- `200 {"ok": true, "location": null}` — the address is **not currently
  sharing** a location. Mapped ONLY from the exact SDK error code
  `sharedFriendLocationNotFound`. Not an error.
- `502 {"ok": false, "error": "upstream locations error"}` — any other SDK
  failure, **including a generic `NotFoundError` with a different or missing
  code** (`chatNotFound`, `addressNotFound`, …) — mapping those to `null`
  would fabricate a "not sharing" answer. The response and the sidecar log
  carry only the error class/code, never the handle or the SDK message text.
- `503 {"ok": false, "error": "locations unavailable"}` — the Spectrum app
  has no remote iMessage client (e.g. local mode).

## `POST /locations/watch`

Long-lived NDJSON stream (`Content-Type: application/x-ndjson`) of live
location updates for one friend. One JSON frame per line.

### Frames

1. **Epoch — always the first line.** Identifies this connection; all
   subsequent frames carry the same `connectionEpoch`.

   ```json
   {"type": "epoch", "connectionEpoch": "<uuid4>", "startedAtMs": 1754400000000}
   ```

2. **Update** — one per SDK location update, in order:

   ```json
   {"type": "update", "connectionEpoch": "<uuid4>",
    "sourceSequence": 1, "location": { ...SharedFriendLocation... }}
   ```

   `sourceSequence` is **connection-scoped**: it starts at 1 on every new
   connection and increments by exactly 1 per update frame. It deliberately
   replaces the SDK's process-scoped `sourceSequence` (whose absolute value
   is meaningless across reconnects; replay is not available either way).
   `location` is the SDK payload, serialized as in `/locations/get`.

3. **Heartbeat** — driven by the SDK's client-level stream-liveness signal
   (the server emits heartbeat frames on a fixed cadence even when idle):

   ```json
   {"type": "heartbeat", "connectionEpoch": "<uuid4>", "atMs": 1754400001234}
   ```

   Heartbeats do not consume sequence numbers and **never** carry a
   location. They are droppable: under consumer backpressure they are
   skipped rather than queued.

### Stream end

There is no end/error frame. When the SDK stream ends (EOF), errors, or the
sidecar shuts down, the sidecar closes the SDK subscription and ends the
HTTP response. **Presence synthesizes the subscription end from EOF.**
Upstream error details are logged class/code-only, never with the handle.

### Abort, cleanup, caps, backpressure

- When the consumer disconnects mid-stream, the sidecar immediately closes
  the SDK `TypedEventStream` (interrupting a pending pull), detaches its
  heartbeat listener, and releases the watcher slot — no leaked tasks or
  listeners. `POST /healthz` reports `locations.activeWatchers` /
  `locations.maxWatchers` for observability.
- Concurrent watchers are capped (`PHOTON_MAX_LOCATION_WATCHERS`, default
  8). Over the cap → `429 {"ok": false, "error": "too many location watchers"}`.
- Backpressure propagates: the sidecar pulls the next SDK update only after
  the previous frame flushed to the consumer socket, so a slow consumer
  pauses the upstream pull instead of growing an unbounded buffer.

## Scoped credential handoff (Presence must not parse `~/.hermes/.env`)

The Photon adapter (`adapter.py`) materializes **only** the sidecar token
into a dedicated runtime credential file when its sidecar becomes ready,
and removes it when the sidecar stops (`runtime_credentials.py`):

| Deployment | Path |
|---|---|
| Default profile | `$XDG_RUNTIME_DIR/hermes/photon-sidecar.token` |
| Named profile `<name>` | `$XDG_RUNTIME_DIR/hermes/profiles/<name>/photon-sidecar.token` |
| Custom `HERMES_HOME` | `$XDG_RUNTIME_DIR/hermes/custom-<sha256[:12]>/photon-sidecar.token` |
| No usable `XDG_RUNTIME_DIR` | `<HERMES_HOME>/run/photon-sidecar.token` |

Properties:

- Directories `0700`, file `0600` (best-effort on Windows).
- The file contains **exactly** the token — no other keys or secrets are
  ever copied there.
- Writes are atomic (`os.replace` of a same-directory temp file) so a
  reader never observes a partial token; rotation on adapter restart is a
  single atomic swap. Stop removes the file and any stale temp files.
- **Symlink-safe.** The runtime root is canonicalized once; every component
  below it is traversed with descriptor-relative `O_NOFOLLOW` opens on
  POSIX, must be a real directory owned by the current user, and the opened
  chain is verified against the canonical path before the token is swapped
  in. A symlinked `hermes`/`profiles`/profile/leaf component aborts the
  write; nothing is ever written outside the approved runtime root. Windows
  falls back to explicit per-component symlink rejection. Cleanup never
  follows symlinks either.
- **Path-bound cleanup.** The adapter records the exact path each write
  returned and clears that path on stop/reconnect — never the path the
  ambient profile scope resolves to at clear time. A secondary-profile
  sidecar stopping while the primary scope is ambient removes its own
  token and leaves the primary's untouched; a reconnect that lands under a
  different scope rotates the previously materialized token out. An adapter
  that never wrote a token clears nothing.
- The token value is never logged.

## Design notes / protocol assumptions

- **SDK access path.** The sidecar reaches the location API through the
  Spectrum app it already owns: `app.__internal.platforms.get("iMessage")
  .client` is the `RemoteClient[]` array spectrum-ts builds in cloud mode;
  entry `[0].client` is an `@photon-ai/advanced-imessage` client exposing
  `locations.get(address)` / `locations.watch(address)`. Multi-line
  dedicated deployments (multiple entries) use the **first** client — Find
  My sharing state is account-level for the project's lines. Local-mode
  installs have no remote client → `503`.
- **Heartbeat hook.** advanced-imessage only accepts a heartbeat callback
  at `createClient(...)` time, and spectrum-ts creates that client without
  one. The sidecar installs the client-wide handler post-hoc by assigning
  the resource's `_onHeartbeat` field, which the pinned SDK (0.12.x) reads
  at `watch()` call time. The signal is client-level liveness, so it is
  broadcast to every active watch connection. If a future SDK removes the
  field, heartbeat frames silently stop (updates are unaffected) — revisit
  on any spectrum-ts/advanced-imessage upgrade.
- **Read-only.** `locations.request(...)` (sending a visible Find My
  request card) is intentionally NOT exposed — this protocol never sends
  outbound messages or location requests.
