// Hermes Agent — Photon sidecar read-only friend-location protocol (S8).
//
// Serves POST /locations/get and POST /locations/watch on the sidecar's
// existing authenticated loopback server (see LOCATIONS_PROTOCOL.md for the
// full contract). The service reuses the ONE Spectrum app this sidecar
// already owns — it reaches the `@photon-ai/advanced-imessage` client that
// spectrum-ts holds internally and never creates a second Spectrum process
// or gRPC client.
//
// Privacy invariant: request targets (handles), coordinates, and location
// payloads never appear in sidecar logs or error responses. Upstream
// failures are logged class/code-only (SDK error messages can embed the
// handle).

import crypto from "node:crypto";

const E164_RE = /^\+\d{6,15}$/;
// Conservative iMessage email handle shape; both regexes are linear-time.
const EMAIL_RE = /^[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}$/;

// Location request bodies are tiny ({"address": ...}); cap them far below
// the sidecar's general 2 MiB control-body cap.
export const MAX_LOCATION_BODY_BYTES = 4096;
const MAX_ADDRESS_LENGTH = 320;
const DEFAULT_MAX_LOCATION_WATCHERS = 8;

export function isValidLocationAddress(address) {
  return (
    typeof address === "string" &&
    address.length > 0 &&
    address.length <= MAX_ADDRESS_LENGTH &&
    (E164_RE.test(address) || EMAIL_RE.test(address))
  );
}

// Resolve the advanced-imessage `locations` resource from the Spectrum app
// the sidecar already owns. In cloud/remote mode spectrum-ts stores its
// runtime client as a RemoteClient[] array ({ phone, client }); the first
// entry's client exposes the location API. Local mode (IMessageSDK) and
// unconfigured platforms have no remote client — callers surface 503.
// Multi-line dedicated deployments use the first client: Find My sharing
// state is account-level for the project's lines (see LOCATIONS_PROTOCOL.md).
export function locationsResourceFromApp(app) {
  const runtime = app?.__internal?.platforms?.get?.("iMessage");
  const client = runtime?.client;
  if (!Array.isArray(client) || client.length === 0) return null;
  const resource = client[0]?.client?.locations;
  if (
    !resource ||
    typeof resource.get !== "function" ||
    typeof resource.watch !== "function"
  ) {
    return null;
  }
  return resource;
}

// "Not currently sharing" is a normal outcome, not an error: the SDK throws
// NotFoundError with code `sharedFriendLocationNotFound`.
function isNotSharing(error) {
  return (
    error?.code === "sharedFriendLocationNotFound" ||
    error?.name === "NotFoundError"
  );
}

// Log-safe error label: class + canonical code only — never the message,
// which can embed the requested handle.
function errorLabel(error) {
  const name = error?.name || "Error";
  const code = error?.code ? `/${String(error.code)}` : "";
  return `${name}${code}`;
}

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

// Wait until the response is writable again — or gone. Racing "drain"
// against "close"/"error" (instead of once(res, "drain")) is what makes a
// consumer that dies mid-backpressure unblock the pump loop instead of
// leaking it forever.
function waitWritableOrClosed(res) {
  return new Promise((resolve) => {
    const done = () => {
      res.off("drain", done);
      res.off("close", done);
      res.off("error", done);
      resolve();
    };
    res.on("drain", done);
    res.on("close", done);
    res.on("error", done);
  });
}

async function closeQuietly(stream) {
  try {
    await stream?.close?.();
  } catch {
    /* already closed / teardown race */
  }
}

// Read and validate a location request body. Returns
// { ok: true, address } or { ok: false, status, error } — the error text is
// generic and never echoes the submitted body.
async function readLocationBody(req) {
  const tooLarge = {
    ok: false,
    status: 413,
    error: "request body too large",
  };
  const declared = Number(req.headers["content-length"]);
  if (Number.isFinite(declared) && declared > MAX_LOCATION_BODY_BYTES) {
    return tooLarge;
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_LOCATION_BODY_BYTES) return tooLarge;
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf-8");
  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    return { ok: false, status: 400, error: "invalid JSON body" };
  }
  const shapeError = {
    ok: false,
    status: 400,
    error: 'body must be {"address": "<E.164 phone or email>"}',
  };
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return shapeError;
  }
  const keys = Object.keys(body);
  if (keys.length !== 1 || keys[0] !== "address") return shapeError;
  if (!isValidLocationAddress(body.address)) return shapeError;
  return { ok: true, address: body.address };
}

export function createLocationsService({
  app,
  maxWatchers = Number(process.env.PHOTON_MAX_LOCATION_WATCHERS) ||
    DEFAULT_MAX_LOCATION_WATCHERS,
  log = console.error,
} = {}) {
  // Active watch connections; each entry is { res, alive }.
  const watchers = new Set();
  // Per-connection heartbeat fan-out callbacks.
  const heartbeatListeners = new Set();
  // Resources we've already installed the heartbeat hook on.
  const hookedResources = new WeakSet();
  let cachedResource = null;

  function resolveResource() {
    if (!cachedResource) cachedResource = locationsResourceFromApp(app);
    return cachedResource;
  }

  // advanced-imessage only accepts a heartbeat callback at createClient()
  // time, and spectrum-ts creates its clients without one. The pinned SDK
  // (0.12.x) reads the resource-level `_onHeartbeat` field at watch() call
  // time, so installing a client-wide handler here gives every active watch
  // connection the server's liveness signal. If a future SDK drops the
  // field, heartbeat frames silently stop (updates are unaffected) — see
  // LOCATIONS_PROTOCOL.md "Design notes".
  function installHeartbeatHook(resource) {
    if (hookedResources.has(resource)) return;
    const prior =
      typeof resource._onHeartbeat === "function"
        ? resource._onHeartbeat
        : null;
    try {
      resource._onHeartbeat = () => {
        if (prior) {
          try {
            prior();
          } catch {
            /* never let a foreign handler break fan-out */
          }
        }
        for (const listener of heartbeatListeners) {
          try {
            listener();
          } catch {
            /* listener cleanup races are harmless */
          }
        }
      };
    } catch {
      return; // frozen/exotic resource — heartbeats unavailable
    }
    hookedResources.add(resource);
  }

  async function handleGet(res, address) {
    const resource = resolveResource();
    if (!resource) {
      return sendJson(res, 503, { ok: false, error: "locations unavailable" });
    }
    try {
      const location = await resource.get(address);
      return sendJson(res, 200, { ok: true, location: location ?? null });
    } catch (e) {
      if (isNotSharing(e)) {
        return sendJson(res, 200, { ok: true, location: null });
      }
      log(`photon-sidecar: locations get failed (${errorLabel(e)})`);
      return sendJson(res, 502, {
        ok: false,
        error: "upstream locations error",
      });
    }
  }

  async function handleWatch(req, res, address) {
    if (watchers.size >= maxWatchers) {
      return sendJson(res, 429, {
        ok: false,
        error: "too many location watchers",
      });
    }
    const resource = resolveResource();
    if (!resource) {
      return sendJson(res, 503, { ok: false, error: "locations unavailable" });
    }

    installHeartbeatHook(resource);
    let stream;
    try {
      stream = resource.watch(address);
    } catch (e) {
      log(`photon-sidecar: locations watch open failed (${errorLabel(e)})`);
      return sendJson(res, 502, {
        ok: false,
        error: "upstream locations error",
      });
    }

    const connectionEpoch = crypto.randomUUID();
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/x-ndjson");
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Connection", "keep-alive");

    const conn = { res, alive: true };
    watchers.add(conn);

    // Heartbeats are liveness-only and droppable: skip when the socket is
    // backpressured instead of queueing behind updates.
    const onHeartbeat = () => {
      if (!conn.alive || res.writableEnded || res.writableNeedDrain) return;
      try {
        res.write(
          JSON.stringify({
            type: "heartbeat",
            connectionEpoch,
            atMs: Date.now(),
          }) + "\n"
        );
      } catch {
        /* teardown race — onAbort cleans up */
      }
    };
    heartbeatListeners.add(onHeartbeat);

    let aborted = false;
    const onAbort = () => {
      aborted = true;
      conn.alive = false;
      // Interrupt a pending stream pull so the pump loop exits promptly.
      void closeQuietly(stream);
    };
    req.on("close", onAbort);
    res.on("close", onAbort);
    res.on("error", onAbort);

    // Backpressure-aware frame writer: the next SDK update is only pulled
    // after this frame flushed (or the consumer went away).
    async function writeFrame(frame) {
      if (aborted || res.writableEnded) return;
      const flushed = res.write(JSON.stringify(frame) + "\n");
      if (!flushed) await waitWritableOrClosed(res);
    }

    try {
      await writeFrame({
        type: "epoch",
        connectionEpoch,
        startedAtMs: Date.now(),
      });
      // Connection-scoped sequence: starts at 1 for every consumer,
      // replacing the SDK's process-scoped counter (see protocol doc).
      let sourceSequence = 0;
      for await (const update of stream) {
        if (aborted) break;
        sourceSequence += 1;
        await writeFrame({
          type: "update",
          connectionEpoch,
          sourceSequence,
          location:
            update && typeof update === "object"
              ? update.location ?? null
              : null,
        });
        if (aborted) break;
      }
    } catch (e) {
      // SDK stream error: end the response — the consumer synthesizes the
      // subscription end from EOF. Class/code only; the message can carry
      // the watched handle.
      log(
        `photon-sidecar: locations watch stream error (${errorLabel(e)}) — ending stream`
      );
    } finally {
      conn.alive = false;
      watchers.delete(conn);
      heartbeatListeners.delete(onHeartbeat);
      req.off("close", onAbort);
      res.off("close", onAbort);
      res.off("error", onAbort);
      await closeQuietly(stream);
      try {
        res.end();
      } catch {
        /* already gone */
      }
    }
  }

  return {
    /** Route one authenticated POST /locations/* request. Never throws. */
    async handleRequest(req, res) {
      try {
        const parsed = await readLocationBody(req);
        if (!parsed.ok) {
          if (parsed.status === 413) res.setHeader("Connection", "close");
          return sendJson(res, parsed.status, {
            ok: false,
            error: parsed.error,
          });
        }
        if (req.url === "/locations/get") {
          return await handleGet(res, parsed.address);
        }
        return await handleWatch(req, res, parsed.address);
      } catch (e) {
        // Defensive: nothing above should throw, and we must never leak a
        // stack (it can embed the handle) to the shared handler's logger.
        log(`photon-sidecar: locations handler error (${errorLabel(e)})`);
        if (!res.headersSent) {
          return sendJson(res, 500, {
            ok: false,
            error: "internal sidecar error",
          });
        }
        try {
          res.end();
        } catch {
          /* already gone */
        }
      }
    },

    /** /healthz payload: observable watcher accounting (leak canary). */
    snapshot() {
      return { activeWatchers: watchers.size, maxWatchers };
    },

    /** Tear down every active watch connection (sidecar shutdown). */
    shutdown() {
      for (const conn of [...watchers]) {
        try {
          conn.res.destroy();
        } catch {
          /* already gone */
        }
      }
    },
  };
}
