# ADR: Discord Realtime Voice Lane (gpt-realtime-2.1) with Asynchronous Hermes Workers

Status: Proposed (plan-only; no production code yet) — rev 2 after review
Date: 2026-09-01
Scope: Discord voice only. Slack/Telegram/text paths and ordinary Discord text are unchanged.

## Objective

Voice must feel like one intelligent Jeeves. A GPT-Realtime-2.1 speech-to-speech
lane owns live conversation — natural turn-taking, clarification, spoken
progress, interruption — while full Hermes workers remain the authoritative
research/action arm with the same identity, tools, permissions, and
source-grounding requirements. **Realtime Jeeves is the sole semantic speaker**:
workers never address the user; the lane never invents authoritative facts.

## Context: the current cascaded pipeline (origin/main)

All of this exists today and stays as the fallback lane, byte-for-byte
unchanged when the flag is off:

| Stage | Where |
|---|---|
| RTP capture, NaCl+DAVE decrypt, Opus decode, per-SSRC buffers | `plugins/platforms/discord/adapter.py` — `VoiceReceiver` (`_on_packet`, `map_ssrc`, `_install_speaking_hook`) |
| Utterance segmentation (1.5 s silence), poll loop, UDP keepalive | `VoiceReceiver.check_silence` + `DiscordAdapter._voice_listen_loop` |
| STT (Whisper) + hallucination filter | `DiscordAdapter._process_voice_input` → `tools/transcription_tools.transcribe_audio`, `tools/voice_mode.is_whisper_hallucination` |
| Routing into the agent session | `gateway/run.py::_handle_voice_channel_input` — synthetic `MessageEvent(message_type=VOICE)` on the bound text channel's session, transcript dedup (`_is_duplicate_voice_transcript`), auth (`_is_user_authorized`), `[Voice]` transcript posted to the text channel |
| Voice join/leave, text-channel binding, inactivity timeout | `DiscordAdapter.join_voice_channel` / `leave_voice_channel`, `_voice_text_channels`, `_voice_sources`; gateway `_handle_voice_channel_join` / `_handle_voice_channel_leave`, `_voice_mode` |
| Spoken reply | `_should_send_voice_reply` → TTS → `play_tts` → `play_in_voice_channel` → `VoiceMixer.play_speech` (ducking) or legacy `FFmpegPCMAudio` |
| Tool-start spoken ack (canned phrases, cascaded lane only) | `gateway/run.py::voice_ack_callback` (agent `tool_start_callback`) → `DiscordAdapter.play_ack_in_voice` |
| Continuous outgoing audio, ambient bed, immediate speech drop | `plugins/platforms/discord/voice_mixer.py` — `VoiceMixer`, `MixerChild`, `stop_speech()` |
| TTS script normalization | `tools/tts_text_normalize.py::prepare_spoken_text` (used by `tools/tts_tool.py` and gateway auto-TTS) |

Existing infrastructure the design builds on:

- **Realtime WS precedent:** `plugins/google_meet/realtime/openai_client.py` —
  minimal sync OpenAI Realtime client (TTS-only). Proves the dependency surface
  (`websockets`, lazily imported) and event vocabulary (`session.update`,
  `response.audio.delta`, `response.cancel`). The new lane needs a full-duplex
  async client; the lazy-import and version-tolerant connect patterns carry over.
- **Isolated child workers:** `tools/delegate_tool.py::_build_child_agent`
  constructs a child `AIAgent` from a parent principal — inherits credentials,
  model routing, and toolsets (minus the child blocklist: `delegate_task`,
  `clarify`, `memory`, `send_message`, `cronjob`), installs the
  `delegation.subagent_auto_approve`-controlled approval callback
  (auto-deny by default), registers in `_active_subagents` with
  `interrupt_subagent(subagent_id)` for cancellation, and tracks
  `parent_session_id`/`child_session_id` lineage in the session DB. This is the
  exact machinery `delegate_task(background=true)` uses via
  `tools/async_delegation.dispatch_async_delegation`.
- **Identity sources (the real ones):** the full agent's identity is slot #1 of
  the stable prompt tier — `agent/prompt_builder.load_soul_md()` reads
  `SOUL.md` from the profile's own home (`home_override=_agent_home(agent)`,
  #50233), falling back to `DEFAULT_AGENT_IDENTITY`
  (`agent/system_prompt.py::build_system_prompt_parts`). Stable user
  preferences live in the `USER.md` profile store, rendered by
  `agent._memory_store.format_for_system_prompt("user")` in the volatile tier.
  Approval boundaries come from `delegation.subagent_auto_approve` +
  `security.*` config, enforced by callbacks — not prose.
- **Agent execution hooks:** `run_agent.py` exposes `tool_start_callback`,
  `tool_complete_callback`, `tool_progress_callback`; the gateway progress
  queue carries model-authored interim content (content bubbles). These generic
  per-run hooks are what the worker bridge taps — no new core hooks.
- **Config shape:** `hermes_cli/config.py::_PLATFORM_CONTAINER_KEYS` accepts
  `gateway.platforms.<name>.<anything>`; `gateway/config.py` resolves the
  `gateway.platforms` map into `PlatformConfig.extra`. `OPENAI_API_KEY` is
  already a known secret in `OPTIONAL_ENV_VARS`. No new env vars, no
  `DEFAULT_CONFIG` change, no `_config_version` bump.

## Decision

### D1. Feature flag and configuration

Reversible flag, default false, read at **voice-join time** (flipping it
requires only leave/rejoin, not a gateway restart):

```yaml
gateway:
  platforms:
    discord:
      voice:
        realtime:
          enabled: false            # master switch (default: off)
          model: gpt-realtime-2.1
          voice: cedar              # Arbor is ChatGPT-app-only; NOT in the
                                    # Realtime API. cedar is the chosen stand-in.
          reasoning_effort: low     # conversational lane only
          api_key_secret: OPENAI_API_KEY   # NAME of the .env secret to read
          connect_timeout_seconds: 8
          reconnect_budget_seconds: 15
          rollover_margin_seconds: 300     # start rollover at 55 min
          max_inflight_dispatches: 3       # plugin-local worker executor cap
          synthesis:
            tts_model: gpt-4o-mini-tts     # exact sourced-synthesis TTS (D13)
            voice: cedar                   # must match the realtime voice
            framing: true                  # native model may frame (no figures)
```

All keys are behavioral → config.yaml only. The one credential is read from the
`.env` secret named by `api_key_secret` (default `OPENAI_API_KEY`), through the
same scope-aware read path platform adapters already use (fail-closed under
`gateway.multiplex_profiles`, per `agent/secret_scope.py`).

Voice verification is part of the connect handshake: the lane asserts the
`session.created`/`session.updated` echo reports `voice == cedar`; a mismatch
(provider renamed/retired the voice) fails the lane loudly at join → cascaded
fallback, never a silent default voice.

### D2. Lane ownership: one consumer per guild, explicit state machine

A per-guild `VoiceLane` enum: `CASCADED | REALTIME | DEMOTING`. Decided once at
`join_voice_channel`:

- Flag off → `CASCADED`. The realtime package is **not imported** (lazy import
  behind the flag check), so disabled deployments pay no import cost and cannot
  break on a missing optional dep.
- Flag on → attempt realtime connect during join (bounded by
  `connect_timeout_seconds`). Success → `REALTIME`: `VoiceReceiver` runs in
  **frame-tap mode** (decrypted PCM frames stream to the lane; the
  silence-detection utterance path and `_voice_listen_loop` STT dispatch are
  disabled for that guild). Failure (including the D1 voice check) → log +
  telemetry, `CASCADED` exactly as today.
- Mid-session WS drop → `DEMOTING`: inbound frames buffer (bounded, ~10 s);
  reconnect within `reconnect_budget_seconds` re-enters `REALTIME` and flushes
  the buffer to the new socket; budget exhausted → lane becomes `CASCADED`, the
  buffered PCM is handed to the legacy utterance path **once** under the lane
  lock, and stays cascaded until rejoin.

This single-owner handoff guarantees **a user turn is never duplicated across
both lanes**: exactly one lane consumes inbound audio at any instant, and the
demotion handoff moves bytes, not turns, exactly once.

### D3. Module layout — everything at the Discord plugin edge

```
plugins/platforms/discord/realtime/
├── __init__.py          # lane factory; import-guarded by the config flag
├── lane.py              # RealtimeVoiceLane: per-guild state machine (D2),
│                        # turn sequencing + dispatch registry (D8),
│                        # rollover (D10)
├── transport.py         # RealtimeTransport: async WS client for
│                        # gpt-realtime-2.1 (server VAD, pcm16, tool calls,
│                        # response lifecycle, response.cancel, output
│                        # transcript events)
├── audio.py             # 48kHz stereo s16 ↔ 24kHz mono pcm16 resampling
│                        # (numpy, same "voice" extra), inbound frame pump,
│                        # outbound StreamingMixerChild bridge
├── projection.py        # compact, byte-stable Jeeves projection builder (D5)
├── tools.py             # in-lane tool defs + dispatch/cancel/recall handlers (D6)
├── worker_bridge.py     # isolated child worker sessions + hook taps →
│                        # typed WorkerEvents; reinjection pump (D7)
├── outbox.py            # deterministic result renderer + exactly-once
│                        # text-channel poster (D9)
├── exact_tts.py         # interruptible streaming OpenAI TTS for sourced
│                        # spoken_synthesis (D13) → StreamingMixerChild
├── authority_gate.py    # business-figure detector; hard block on native
│                        # responses (D13)
├── events.py            # WorkerEvent / DispatchRecord dataclasses (D7, D8)
└── telemetry.py         # counters/histograms, structured log flush (D12)
```

Changes **outside** the plugin: (a) `voice_mixer.py` gains a
`StreamingMixerChild` (a `MixerChild` sibling fed by a thread-safe frame queue
with `clear()`) so realtime audio plays through the existing continuous mixer
with ducking and barge-in clears at 20 ms granularity; (b)
`tts_text_normalize.py` gains the business-magnitude rule (D11). No core tool
schema changes, no `toolsets.py` changes, no new `run_agent.py` surface — the
model-facing tools in D6 live inside the provider session config, invisible to
every other Hermes surface.

### D4. Transport: server-side WebSocket, native speech-to-speech

`RealtimeTransport` is an asyncio client on the gateway process (server-side;
no client SDK, no WebRTC):

- `wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1`, `Authorization:
  Bearer <secret>`.
- `session.update` on connect: pcm16 in/out, server VAD with barge-in
  (`turn_detection: {type: server_vad, interrupt_response: true}`),
  `reasoning_effort: low` (2.1 session field; if the deployed API build rejects
  it, degrade by omitting — log once, don't fail the lane), `voice: cedar`
  (verified per D1), tools (D6), instructions = projection (D5).
- Inbound: `input_audio_buffer.append` frames from `audio.py`. Outbound:
  audio deltas → resample → `StreamingMixerChild` queue; **output transcript
  deltas** are consumed in parallel by the authority gate (D13) and the
  rollover transcript window (D10). Function calls → `tools.py` handlers →
  `conversation.item.create(function_call_output)` + `response.create`.
- Event-name tolerance: delta/lifecycle event names resolve through one small
  mapping table in `transport.py` (the GA API renamed several
  `response.audio.*` events); tests pin the mapping, so a provider rename is a
  one-line fix.

### D5. Session bootstrap: compact projection from the real identity sources

`projection.py::build_projection(...)` composes, in fixed order, from the
**same sources the full agent uses** — no parallel persona files:

1. **Identity:** `agent/prompt_builder.load_soul_md(home_override=<profile home>)`
   — the profile's `SOUL.md`, exactly as `build_system_prompt_parts` loads it
   (fallback `DEFAULT_AGENT_IDENTITY`). If it exceeds the projection budget
   (~4 KB total), truncate at a markdown section boundary, deterministically —
   never paraphrase.
2. **Stable user preferences:** the `USER.md` profile block via
   `MemoryStore.format_for_system_prompt("user")` — the identical renderer the
   full prompt's volatile tier uses — captured **once at join** and frozen for
   the lane's life (the full agent re-reads per session; the lane must not
   mutate instructions mid-session, so it snapshots).
3. **Authority contract** (fixed text): the lane may speak (a) what the user
   said this session, (b) in-lane read results, (c) worker results with their
   source/freshness metadata. Any other claim that sounds like a fact about
   the world or the business requires `hermes_dispatch`; while waiting it may
   only describe what it is doing. This is backstopped mechanically by D13 —
   the prompt is guidance, the gate is enforcement.
4. **Approval boundary statement** (fixed text, derived from config at join):
   whether workers auto-deny dangerous commands
   (`delegation.subagent_auto_approve=false`, the default) so the lane sets
   honest expectations ("that needs an approval I can't grant by voice").
5. **Speech style contract** (fixed text): concise synthesis aloud; detail
   goes to the text channel; speak abbreviations naturally ("38M" →
   "thirty-eight million"); acknowledgements are natural and context-specific
   — never canned phrases; no tool-name narration; no filler heartbeats.
6. **Static session context:** guild/channel names, bound text channel, user
   display name.

Explicitly excluded: the full system prompt scaffold, tool catalogs, skills
index, MEMORY.md, workspace snapshots. The projection is byte-stable for the
lane's life and across rollover (D10 reuses the join-time bytes; no re-read).
Dynamic facts enter as conversation items, never as instruction rewrites.

### D6. Two tool classes exposed to the realtime model

Defined in the provider session config only (zero core schema growth):

**Fast in-lane reads** — answered by the plugin in-process, milliseconds, no
agent run:

- `voice_context()` — who is in the channel, bound text channel, current time.
- `recall_result(dispatch_id?)` — returns completed worker results from the
  dispatch registry **only**, verbatim with `sources[]` and `fetched_at`.
  It cannot fabricate: an unknown id or empty registry returns a typed
  "no result", which combined with the authority contract and D13 forces a
  dispatch.

**Asynchronous Hermes worker dispatch:**

- `hermes_dispatch(task, spoken_ack_hint?, supersedes_dispatch_id?)` — returns
  `{dispatch_id}` immediately. The lane speaks its own natural,
  model-authored acknowledgement — there is no canned ack anywhere in this
  lane (see D11 for the WIP-patch policy).
- `hermes_cancel(dispatch_id, reason)` — cancels/redirects obsolete work.

### D7. Workers: isolated child sessions, never the bound text-channel session

**What is ruled out:** injecting dispatches as turns into the bound Discord
text-channel session. With `max_inflight_dispatches=3` that would run
concurrent model loops over one session's history — violating strict role
alternation, invalidating the per-conversation prompt cache, and muddling
cancellation ownership. The bound session belongs to the user's text
conversation; workers do not write to it.

**What happens instead:** each `hermes_dispatch` creates an **isolated child
worker session keyed by `dispatch_id`**, built and run through the existing
delegation machinery at the plugin edge:

- **Construction:** `tools/delegate_tool._build_child_agent(goal, context,
  parent_agent=<lane principal>, ...)`. The *lane principal* is a
  principal-spec object assembled at join from the same turn-route resolution
  the gateway uses for the bound session's agents (the `ctx.AIAgent(...)`
  construction block in `gateway/run.py`'s message-processing pipeline):
  identical provider/model/credentials, toolsets, `user_id`/`user_name`,
  profile home. Two deliberate overrides vs. the delegate-leaf default
  (`platform="subagent"`, `skip_context_files=True`, `skip_memory=True`):
  `load_soul_identity=True` so the worker carries the same `SOUL.md` Jeeves
  identity (this is the exact flag cron uses for the same need), and the
  worker's user-visible output contract below. `skip_memory=True` stands
  (like cron): workers do not sync memory providers; durable facts they
  produce return through events, and the user's stable preferences travel in
  the context snapshot.
- **Context snapshot:** `context` is an immutable, high-signal snapshot built
  at dispatch: the user's verbatim request, the relevant recent voice-turn
  transcript excerpts (bounded), relevant `USER.md` preference lines, and the
  bound guild/channel identifiers. Frozen at dispatch — later voice turns
  never mutate a running worker's context (cross-task isolation).
- **Lineage:** `parent_session_id` = the voice lane's session identifier;
  child sessions land in the session DB with normal
  `parent_session_id`/`child_session_id` lineage (auditable, resumable).
- **Auth principal:** dispatch happens only after the same
  `_is_user_authorized` gate the cascaded path applies to the speaking user;
  the child inherits the principal's credentials and toolsets minus the child
  blocklist — same permission envelope as any `delegate_task` child.
- **Approvals:** the standard subagent approval callback applies
  (`delegation.subagent_auto_approve`, default auto-deny). A denial surfaces
  as a `blocker` event the lane can voice; there is no voice-channel approval
  grant in v1.
- **Clarification:** leaf children have no `clarify` tool (existing
  blocklist). A worker needing input returns a structured
  needs-input outcome in its result; the bridge emits a `blocker` event; the
  voice model relays the question and a fresh dispatch carries the answer.
- **Execution & cancellation:** the bridge runs children on a plugin-local
  bounded executor (`max_inflight_dispatches`), registered in the existing
  `_active_subagents` registry so `interrupt_subagent(subagent_id)` — the same
  seam `/stop` and gateway shutdown use — implements `hermes_cancel` and
  supersession. (The durable `dispatch_async_delegation` SQLite ledger is
  deliberately not used: its completion queue re-enters results as gateway
  session turns, which is exactly the pattern being ruled out; voice events
  are meaningless without a live lane, so process-local is correct.)
- **Cleanup:** lane teardown (leave/disconnect) interrupts nothing by itself —
  live workers keep running; their **final verified results still post to the
  bound text channel through the outbox (D9)**, voice events are dropped
  (D8). Executor and registry entries are reaped on completion; gateway
  shutdown interrupts via the existing `_active_subagents` sweep.

**Workers never address the user.** The child's toolset excludes
`send_message`; its final response and interim content go only to the bridge's
hook taps (`tool_complete_callback`, content bubbles, final response), which
reduce them to typed events:

```python
@dataclass(frozen=True)
class WorkerEvent:
    dispatch_id: str
    epoch: int                # dispatch-ownership epoch, see D8
    type: Literal["started", "finding", "blocker",
                  "completed", "failed", "cancelled"]
    spoken_hint: str          # concise, user-facing; never CoT, never tool names
    detail_ref: str | None    # Discord message id of the outbox post (D9)
    sources: list[Source]     # [{title, url, fetched_at}] on finding/completed
    unsourced: bool           # True when the worker cited nothing
    ts: float
```

Event derivation rules (what prevents fake cadence):

- `started` — once, on dispatch acceptance.
- `finding` — only from model-authored interim content (content bubbles) or a
  tool result the worker explicitly surfaced; never from raw tool starts,
  never on a timer. A worker with no interim content produces silence until
  completion — by design.
- `blocker` — approval denials, needs-input outcomes, errors.
- `completed`/`failed`/`cancelled` — terminal, exactly once, with `sources`
  extracted from the worker's citations.

The **reinjection pump** (per lane) delivers events into the live realtime
conversation as `conversation.item.create` (system-role message rendering the
event) + `response.create` — only at a safe boundary: never while the user is
speaking (server VAD state), never mid-response (queue and coalesce). The
voice model phrases progress itself; the event text is data, not a script.

### D8. Turn sequencing vs. dispatch ownership — two separate mechanisms

Rev 1 conflated these; they are distinct:

- **`turn_seq`** (conversation-turn counter): bumped on every user speech
  start. Governs only the *audio plane*: which provider response is current,
  what barge-in cancels, which queued audio is stale. Barge-in
  (`speech_started`) immediately — in order — calls
  `StreamingMixerChild.clear()` (≤ one 20 ms frame to silence), sends
  `response.cancel`, clears the provider's queued output. **Barge-in and
  ordinary follow-up speech never touch dispatches.**
- **Dispatch ownership epochs**: each `DispatchRecord{dispatch_id, epoch,
  subagent_id, guild_id, text_channel_id, user_id, status}` stays live until
  an *explicit semantic revocation*: `hermes_cancel`, a dispatch carrying
  `supersedes_dispatch_id`, a `/voice` session reset, or leave. Only these
  bump the lane `epoch` for the affected records / mark them cancelled.
  Backchannel ("mm-hm", "any update?", "also add Q3 when you get a chance")
  therefore cannot stale a still-relevant worker.

Delivery rules:

- **Routing (contract 4):** an event is injected iff its `dispatch_id` is
  registered, status live, epoch current, and the lane is still bound to the
  same (guild, text channel). Exactly-once via claim-mark-complete on the
  record.
- **Stale suppression (contract 5):** events for revoked records are dropped
  and counted (`stale_events_dropped`). Revocation also interrupts the running
  child (`interrupt_subagent`), so obsolete work stops burning tokens, not
  just its audio.
- **Leave:** lane torn down, records marked voice-orphaned (voice events
  drop; outbox text delivery still completes). No audio replay on rejoin.

The follow-up-vs-correction distinction is proven by tests (slice 6): ordinary
follow-up speech during a live dispatch must not prevent its completion from
delivering; an explicit correction must.

### D9. Spoken vs. posted content split — one deterministic outbox

`outbox.py` is the only writer of worker output to Discord text:

- Renders a completed/failed result deterministically (fixed template: task,
  outcome, tables/citations verbatim from the worker, sources with
  freshness) and posts it to the bound text channel via `adapter.send`,
  exactly once per dispatch (idempotency key = `dispatch_id`).
- The event's `spoken_hint` stays plain prose ≤ ~2 sentences; anything
  structured (tables, >3-item lists, code, links, citations) lives only in
  the outbox post; the event carries `detail_ref` so the voice model can say
  "details are in the channel." The lane never reads tables aloud.
- Workers cannot post directly (no `send_message`); the outbox is the single
  deterministic renderer between worker results and the user-visible channel.

### D10. 60-minute provider session rollover — exact state, no summarizer

The provider hard-caps realtime sessions at 60 minutes. There is **no
in-tree summary mechanism fit for this** (context compression is a
conversation-history mechanism, not a provider-session restore), and a
plugin-side heuristic summarizer would be a fake — ruled out. Restore state is
**exact**:

- The lane maintains a **verbatim transcript window**: the last N turns
  (default 40, config-capped bytes) of user-input transcripts and
  assistant-output transcripts, both provided by the provider's own
  transcript events — no lossy re-derivation.
- Plus the rendered **dispatch/result registry**: live dispatches, their
  goals/status, and completed results' synopses with sources (already exact,
  D7/D8 state).

At `rollover_margin_seconds` before the cap, rollover arms and executes at the
next turn boundary (no user speech in flight, no response streaming, output
queue drained):

1. Open a new WS; send the **identical join-time projection** (D5) plus two
   conversation items: the verbatim transcript window and the registry
   rendering.
2. Swap transports under the lane lock; close the old socket.
3. Worker ownership is unaffected: records live in `lane.py`, not the provider
   session; in-flight dispatches deliver into the new session.

Older turns beyond the window are genuinely gone from the provider's context —
stated plainly rather than papered over with a pretend summary; the bound text
channel and session DB remain the durable record. No audio replay: outbound
audio is plugin-side and drained at the boundary; inbound frames buffer during
the sub-second swap. A mid-conversation cap despite the margin is handled as a
reconnect (D2) with the same restore payload.

### D11. Scripted-speech normalization & the live WIP-patch import policy

The realtime lane's audio is model-generated (the projection's style contract
covers spoken expansion). Exact **scripted** speech paths — cascaded auto-TTS
replies and `play_ack_in_voice` — go through
`tools/tts_text_normalize.py::prepare_spoken_text`, which today has no
magnitude rule. Add to `normalize_symbols_for_tts`, ordered **before** the
unit rules (the existing `(?<=\d)\s*m\b → metres` rule would otherwise eat
"38m"):

- `(?<![\w.])(\d+(?:\.\d+)?)\s*([KMBT])(?![\w])` → `\1 thousand|million|billion|trillion`
  (uppercase only, so `38M` → `38 million` but `120mm` stays millimetres), and
  currency forms `$38M` → `38 million dollars` via the magnitude expansion
  running inside the existing money-rule captures.

**Live Jeeves WIP patch policy:** the currently deployed (out-of-tree) patch
on Jeeves adds three behaviors to the cascaded path. When importing it into
this branch: **keep** the financial normalization (folds into the rule above,
with tests) and **keep** the immediate playback-stop behavior (folds into
`StreamingMixerChild.clear()` / `VoiceMixer.stop_speech()` semantics);
**drop** the canned immediate "Got it…" acknowledgement in `gateway/run.py` —
it must not be carried into the realtime lane, whose acknowledgements are
model-authored and context-aware (D6). The cascaded fallback may keep its
existing `play_ack_in_voice` canned-phrase behavior unchanged (it is
config-gated and off the realtime path).

### D12. Telemetry

`telemetry.py`: in-memory counters/histograms per lane, flushed as structured
single-line log records (agent.log, INFO) every 60 s and at teardown. Metrics:
`speech_end_to_first_audio_ms` (histogram, p50/p95), `barge_in_stop_ms`,
`worker_event_latency_ms` (event created → injected), `stale_events_dropped`,
`authority_gate_trips`, `reconnects`, `rollovers`, `session_resumes`,
`audio_out_queue_depth` (gauge, sampled). Payloads carry ids and durations
only — no raw audio, no transcripts, no secrets.

### D13. Authority: exact sourced synthesis path — never a trailing watchdog

Final invariant: **native GPT-Realtime-2.1 audio owns ordinary conversation,
clarification, contextual acknowledgements, and model-authored progress.
Authoritative worker-backed business/data answers are never delivered by
native free composition and never rely on a prompt or a trailing transcript
watchdog.** Provider transcript deltas trail audio, so zero leakage on the
native path is unprovable — therefore worker-result delivery is
**pre-classified** and uses only an exact scripted path:

1. **Exact sourced synthesis (`exact_tts.py`):** the full Hermes worker emits
   an exact `spoken_synthesis` script with its `completed` event (figures
   normalized via `prepare_spoken_text`, incl. the D11 financial-magnitude
   rule; sources named). The lane routes that script through an
   **interruptible exact streaming TTS path** — OpenAI speech/TTS
   (`POST /v1/audio/speech`, streamed, `response_format=pcm`,
   `voice: cedar` to match the realtime voice) — decoded PCM feeds the same
   `StreamingMixerChild`, so barge-in clears it identically (D8 audio plane).
   The script is spoken verbatim; the TTS engine cannot introduce figures.
2. **Continuity injection:** after the script is scheduled/played, the lane
   injects the exact spoken text into the realtime conversation as an
   assistant conversation item (`conversation.item.create`), so follow-ups
   ("wait, which month was that?") retain full context of what was actually
   said. If barge-in truncated playback, the injected item is annotated as
   interrupted at the truncation point.
3. **Pre-classification:** responses triggered by worker `completed` events
   carrying figures are classified result-delivery **before** any audio is
   requested: the lane never issues a native `response.create` for the
   figures themselves. The native model may add a short conversational frame
   around the result (injected event item + per-response instructions:
   "introduce/frame only; do not state figures — the exact result is spoken
   separately"), or the lane may skip framing entirely (config
   `synthesis_framing: true|false`, default true).
4. **Authority gate (`authority_gate.py`) — hard block, defense-in-depth:**
   every *native* response's output-transcript deltas run through a
   deterministic business-figure detector (currency, magnitude suffixes,
   percentages, large quantities). Any hit on a native response — which by
   pre-classification is never a result-delivery response — fires the
   barge-in path (`StreamingMixerChild.clear()` + `response.cancel`),
   increments `authority_gate_trips`, injects a system correction item, and
   triggers a corrective response that retracts and dispatches. **No numeric
   authoritative fragment is ever intentionally allowed through on the
   native path**; the gate exists to cancel spontaneous invention, not to
   validate result delivery (which never rides the native path).

**Residual, stated plainly:** the gate's cutoff on a spontaneously-invented
native figure is still subject to transcript-lag physics (a fragment may be
heard before the cancel). That path is a *violation being killed*, not a
sanctioned delivery mechanism — the invariant is that no authoritative answer
is ever *intentionally* routed where leakage is possible. Sourced results are
exact-TTS-only by construction. `authority_gate_trips > 0` in the supervised
canary blocks rollout. One provider-side unknown is called out for the canary:
`cedar` availability on the speech endpoint (it is a realtime-family voice);
`synthesis.voice` is config-mapped with `cedar` as the required default, and
the provider canary verifies acceptance — a rejection is a rollout blocker to
resolve, never a silent voice swap.

## Acceptance contracts → test plan (vertical TDD slices, RED → GREEN order)

Test placement: Python tests under `tests/gateway/` (matching
`test_discord_voice_mixer.py` precedent), run via `scripts/run_tests.sh`.
Slices use a `FakeRealtimeServer`/injected transport double — deterministic,
no live network. Behavior contracts, not snapshots. **Deterministic tests
prove plumbing contracts only; latency and speech-quality acceptance is
decided exclusively by the live canaries (slice L, deployment steps 4–5).**

1. **Lane gate & fallback** (contract 7 gating half) —
   `test_discord_realtime_lane_select.py`
   RED: flag off ⇒ join installs the cascaded path and `sys.modules` contains
   no `...discord.realtime` entry; flag on + connect failure or cedar-check
   failure ⇒ cascaded lane, single consumer, join still succeeds.
   GREEN: `__init__.py` factory + `lane.py` skeleton.
2. **Flag isolation across surfaces** (point-9 contract) —
   `test_discord_realtime_flag_isolation.py`
   RED: with the flag **enabled**, constructing Slack/Telegram adapters and
   processing a Discord *text* message and a Discord *cascaded-voice* turn
   (flagged guild ≠ bound guild) touches no realtime module and produces
   byte-identical outbound calls vs. a flag-off run (recorded fake adapter);
   importing the realtime package registers zero entries in
   `tools/registry.py` and leaves `toolsets.TOOLSETS` untouched; no realtime
   config key is read outside the Discord adapter. GREEN: nothing — this
   slice must pass by construction; a failure is a design leak.
   (The other half of contract 7 is the existing suite staying green with the
   flag off — CI already enforces it.)
3. **Transport bootstrap & projection stability** —
   `test_discord_realtime_transport.py`
   RED: `session.update` carries model, `voice: cedar`, `reasoning_effort:
   low`, pcm16, server VAD, the tool defs; voice-echo mismatch fails the
   connect; `build_projection` returns identical bytes across two builds and
   across a simulated rollover; projection assembles from a temp-home
   `SOUL.md` + `USER.md` fixture through the real `load_soul_md` /
   `format_for_system_prompt` helpers (E2E against a temp `HERMES_HOME`, no
   mocks of the loaders); size ceiling enforced; deterministic
   section-boundary truncation. GREEN: `transport.py`, `projection.py`.
4. **Streaming playback & barge-in plumbing** (contract 3 plumbing) —
   extend `test_discord_voice_mixer.py` + `test_discord_realtime_barge_in.py`
   RED: `StreamingMixerChild` plays queued frames through `VoiceMixer.read()`;
   after `clear()` the very next `read()` is silence (≤ 1 frame = 20 ms);
   barge-in handler ordering (clear → cancel → provider buffer clear) and
   `barge_in_stop_ms` recording. GREEN: mixer child + `audio.py` outbound
   bridge + lane barge-in handler.
5. **Inbound audio & single-consumer invariant** (contract 1 plumbing) —
   `test_discord_realtime_audio_in.py`
   RED: frame-tap forwards resampled frames as `input_audio_buffer.append`;
   the cascaded silence path is inert while lane=REALTIME (no
   `transcribe_audio` call — the never-duplicate invariant);
   `speech_end_to_first_audio_ms` measured from VAD `speech_stopped` to first
   delta enqueue; no sleeps/subprocesses in the hot path. GREEN: `audio.py`
   inbound pump.
6. **Isolated workers, typed events, exact routing, follow-up vs. correction**
   (contracts 2, 4, 5) — `test_discord_realtime_worker_bridge.py` +
   `test_discord_realtime_generations.py`
   RED: `hermes_dispatch` builds a child through the real `_build_child_agent`
   (temp-home E2E: assert the child's system prompt contains the fixture
   `SOUL.md` identity, `load_soul_identity=True`, `send_message`/`clarify`
   absent from its tools, auto-deny approval callback installed,
   `parent_session_id` lineage recorded); the bound text-channel session
   receives **zero** injected turns; a faked child run emitting a content
   bubble + final response yields exactly `started`, one `finding`, one
   `completed` with `sources` and `detail_ref`; events inject once, only for
   the matching (guild, channel, epoch); no `spoken_hint` contains a tool
   name; no interim content ⇒ no `finding` (no heartbeat); reinjection defers
   while fake VAD says the user is speaking. **Follow-up vs. correction:**
   ordinary follow-up speech (and barge-in) during a live dispatch does not
   revoke it — its completion still delivers; `hermes_cancel` /
   `supersedes_dispatch_id` / reset / leave revoke: the child is interrupted
   via `interrupt_subagent`, later events drop, `stale_events_dropped`
   increments, delivery is exactly-once under duplicated completion.
   GREEN: `worker_bridge.py`, `events.py`, `tools.py`, registry in `lane.py`.
7. **Outbox** (contract-4 text half) — `test_discord_realtime_outbox.py`
   RED: one deterministic render per completed/failed dispatch, posted once
   (idempotent under retry), to the bound channel only; structured content
   never appears in `spoken_hint`. GREEN: `outbox.py`.
8. **Reconnect, leave, 60-minute rollover** (contract 6) —
   `test_discord_realtime_rollover.py`
   RED: forced WS drop → frames buffered, reconnect within budget resumes and
   flushes once; budget exhausted → demotion hands buffered PCM to the
   cascaded path once; rollover at fake-clock 55 min swaps at a turn
   boundary, new `session.update` bytes == join-time projection, verbatim
   transcript window + registry items present (and are exact copies of the
   fed transcript events — no summarization), an in-flight dispatch completing
   after rollover delivers once into the new session, no outbound audio
   re-queued; leave keeps the worker running with outbox delivery intact
   while voice events drop. GREEN: rollover + demotion logic.
9. **Authority: exact synthesis path + hard gate** (contract 8) —
   `test_discord_realtime_authority.py`
   RED: `recall_result` returns only registry entries and a typed "no result"
   otherwise; `completed` events carry `sources`/`fetched_at` when cited and
   `unsourced: true` otherwise; sourced completions carry `spoken_synthesis`
   with normalized figures; **pre-classification:** a figure-bearing
   `completed` event routes to `exact_tts` (fake streaming TTS → PCM into
   `StreamingMixerChild`) and never triggers a native `response.create` for
   the figures; the exact spoken text is injected as an assistant
   conversation item after scheduling (interruption-annotated when barged);
   the native framing response's instructions forbid figures; the gate
   detector flags currency/magnitude/percent patterns in fake transcript
   deltas of any native response and fires clear → cancel →
   correction-inject (order asserted), incrementing `authority_gate_trips`;
   non-numeric chit-chat never trips; barge-in during exact-TTS playback
   clears the mixer child immediately.
   GREEN: `exact_tts.py` + `authority_gate.py` + event synthesis in
   `worker_bridge.py`.
10. **Scripted-speech normalization** — `test_tts_business_magnitudes.py`
    (tests/tools/) RED: `prepare_spoken_text("Revenue was $38M")` →
    "38 million dollars"; `38M` → `38 million`; `120mm` stays millimetres;
    `38m` stays metres. GREEN: D11 regex in `tts_text_normalize.py`.

**L. Live canaries (mandatory before enablement; not CI):**
- **Provider canary** — env-gated script (`HERMES_REALTIME_CANARY=1`) opening
  a real gpt-realtime-2.1 session: verifies cedar echo, `reasoning_effort`
  acceptance, event-name mapping, and captures real
  speech-end→first-audio p50/p95 over ≥ 20 scripted turns.
- **Discord supervised canary** — a real guild session with an operator:
  captures end-to-end speech-end→first-audio and barge-in→silence p50/p95
  from telemetry, exercises dispatch/follow-up/correction/rollover live, and
  watches `authority_gate_trips`. Acceptance targets (first-audio p95 < 1 s,
  barge-in p95 < 300 ms, zero unsuppressed stale deliveries, zero uncorrected
  gate trips) are judged **only** on these measurements. Mock-suite green is
  never latency acceptance.

## Deployment / rollback sequence for Jeeves

1. Merge with `enabled: false`. The realtime package is unreferenced at
   runtime (lazy import), so this deploy is inert; run the normal
   `hermes update` fleet flow.
2. Verify cascaded voice on Jeeves unchanged (join, speak, TTS reply), and
   Slack/Telegram/Discord-text spot checks (slice 2's isolation contract,
   verified live).
3. Add the `websockets` dependency to the voice extra if not already pulled in
   (pinned `>=floor,<next_major` per policy); confirm `OPENAI_API_KEY` present
   in Jeeves' profile `.env`.
4. Run the **provider canary** (slice L) from the Jeeves host; require cedar
   echo + latency numbers before proceeding.
5. Set `gateway.platforms.discord.voice.realtime.enabled: true` in Jeeves'
   config.yaml (no restart; read at join). Run the **Discord supervised
   canary** in a private guild; hold enablement in the real guild until its
   acceptance targets pass.
6. **Rollback:** set `enabled: false` (or delete the key) and leave/rejoin —
   the guild is back on the cascaded lane with zero residue. No schema,
   toolset, or core changes to unwind; the entire feature is additive files
   plus two contained diffs (`voice_mixer.py` child class,
   `tts_text_normalize.py` rule).

## Risks

- **Provider event-name / API drift** (2.1 is new; GA renamed events before).
  Mitigated by the D4 mapping table, transport tests pinning it, and the
  provider canary catching drift before enablement.
- **`reasoning_effort` field rejection** on some API builds → degrade by
  omission (D4), logged once, verified in the provider canary.
- **Authority residual (D13):** a fragment of an invented figure can be heard
  before the transcript-watchdog cutoff. Accepted as a loud, counted,
  self-correcting failure; `authority_gate_trips > 0` in the supervised
  canary blocks rollout until the projection/tooling is tuned.
- **Transcript-delta lag variance** could widen the D13 window under load;
  the canary measures actual lag, and the detector runs on deltas (not
  completed transcripts) to minimize it.
- **Latency SLOs are environment-dependent** (Discord UDP + provider RTT) —
  owned entirely by the live canaries (slice L); the unit suite bounds only
  in-process contributions.
- **Multi-speaker channels.** v1 forwards all allowed users' frames and
  attributes turns to the most-recently-active SSRC (matching the existing
  sole-allowed-member inference). Overlapping speakers will confuse
  attribution; acceptable for Jeeves' primary-operator usage, revisit if it
  bites.
- **Cost.** Realtime audio tokens are priced well above the cascaded path;
  the flag is per-deployment opt-in, the inactivity timeout bounds idle
  sessions, and the lane counts provider-session minutes in telemetry.
- **Demotion double-speech window.** If the provider dies mid-response while
  a queued TTS fallback also fires, both could sound. The lane-owner lock +
  the rule that only the owning lane may enqueue mixer speech closes this;
  slice 8 tests the handoff.
- **Worker prompt divergence.** Workers deliberately run a leaner prompt
  (SOUL.md identity + snapshot, no MEMORY.md sync) than the bound text
  session. A worker could lack a memory-resident fact the text agent would
  have had; mitigated by the dispatch snapshot carrying relevant preference
  lines, and honest `blocker`/needs-input outcomes rather than guessing.
