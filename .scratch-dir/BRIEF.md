# Task: Per-channel flat-reply mode for the Slack platform plugin

## Context

Hermes' Slack adapter (`plugins/platforms/slack/adapter.py`) supports a global
`reply_in_thread` boolean in `platforms.slack.extra`:

- `reply_in_thread: true` (default): every top-level channel ping forks a thread;
  sessions are keyed per thread root.
- `reply_in_thread: false`: replies go flat (top-level) into the channel; the whole
  channel shares ONE session keyed `(platform, channel_id, None)`.

The user wants this PER-CHANNEL, pattern-driven: channels whose NAME matches a
configured glob pattern (e.g. `ghost-*`) behave flat (as if reply_in_thread=false);
all other channels keep the current global setting's behavior.

## Config surface (add)

```yaml
platforms:
  slack:
    extra:
      reply_in_thread: true            # existing global default, unchanged semantics
      flat_channel_patterns:           # NEW: list of fnmatch globs against channel NAME
        - "ghost-*"
```

- `flat_channel_patterns` (list[str], default `[]`). A channel whose name matches ANY
  pattern is treated as flat-mode regardless of the global `reply_in_thread`.
- Matching uses `fnmatch.fnmatch` on the channel name (lowercase both sides), no `#` prefix.
- DMs / IMs are NEVER affected by patterns (they have their own session semantics).
- When the list is empty (default), behavior is byte-identical to today — this must hold
  for every existing test.

## Implementation requirements

1. **One decision helper** on the adapter, e.g.:
   ```python
   def _channel_flat_mode(self, channel_id: str) -> bool:
       """True if replies to this channel should be flat (no thread) and the
       channel shares one session. Patterns override the global reply_in_thread."""
   ```
   - Resolves channel name via Slack API `conversations_info`, cached in a dict
     (`self._channel_name_cache: Dict[str, str]`) — one API call per channel per process
     lifetime. On API failure, fall back to the global setting (fail-safe: never crash
     message handling; log at debug).
   - Multi-workspace aware: use the right team client (see `self._team_clients` /
     `_client_for_channel` patterns already in the adapter).
   - Note the adapter already caches channel→team in `self._channel_team`; follow the
     existing caching idioms.

2. **Wire it into BOTH existing decision points** (they must stay in sync — this is the
   invariant the current code comments stress):
   - Inbound session keying in `_handle_slack_message` (~line 2780–2820): the branch
     `elif self.config.extra.get("reply_in_thread", True): thread_ts = ts` becomes
     per-channel: if `_channel_flat_mode(channel)`, take the flat path (`thread_ts = None`).
     Genuine thread replies (event.thread_ts != ts) STAY threaded — a user deliberately
     opening a thread in a flat channel keeps a thread-scoped conversation, exactly like
     reply_in_thread=false does today.
   - Outbound `_resolve_thread_ts` (~line 1630–1660): the guard
     `if not self.config.extra.get("reply_in_thread", True)` needs the per-channel
     equivalent. NOTE: this method currently doesn't receive the channel — you will need
     to thread the channel_id through from its call sites (send() and streaming variants,
     ~lines 1382, 1455). Keep the signature change minimal and default-safe
     (`channel_id: Optional[str] = None` → None means use global behavior only).

3. **The `in_channel` continuable-cron warning** (`_warn_if_in_channel_without_flat`,
   ~line 1594–1624) should recognize that flat_channel_patterns can satisfy the pairing
   requirement — soften the warning when patterns are configured (mention them in the
   log message) rather than trying to resolve channel names at config-load time.

4. **Async correctness**: conversations_info call must follow the adapter's existing
   sync/async client usage patterns (check how the adapter calls Slack Web API elsewhere —
   mirror it exactly).

## Tests (extend `tests/gateway/test_slack.py` — follow its existing fixtures/mocks)

- Pattern match → flat: top-level message in `ghost-foo` with `flat_channel_patterns: ["ghost-*"]`
  and global `reply_in_thread: true` → session key has thread_id None, outbound reply has
  no thread_ts.
- Non-matching channel → threaded (global default true): unchanged legacy behavior.
- Genuine thread reply inside a flat-pattern channel → stays threaded (session keyed to
  thread root; reply carries thread_ts).
- Empty patterns → byte-identical behavior to today (existing tests must pass untouched).
- Channel-name API failure → falls back to global setting, no exception.
- Multiple patterns; case-insensitivity (`Ghost-ABC` matches `ghost-*`).
- DM unaffected by patterns.

## Constraints

- Work ONLY in `plugins/platforms/slack/` + its tests. Zero core changes.
- Follow the repo's AGENTS.md: no new env vars, config.yaml only; no change-detector
  tests; preserve existing comment style (the adapter has heavy explanatory comments —
  match them for the new branches, including WHY patterns override the global).
- Do NOT rename or alter semantics of the existing `reply_in_thread` flag.
- Run the Slack gateway tests (`python -m pytest tests/gateway/test_slack.py -q`) and
  make them pass. If the venv lacks deps, `pip install -e .[dev]` or install
  slack_bolt/slack_sdk + pytest as needed.
- Commit to the CURRENT branch (`arman1/slack-flat-channel-patterns`) with a
  conventional-commit message, and PUSH the branch. Do NOT open a PR.
- Git author: Arman <11728969+ashah360@users.noreply.github.com>

## Done condition

- Helper + both wiring points implemented with explanatory comments.
- All new tests pass; full test_slack.py suite green.
- Branch pushed with the work committed.
- Final summary: files changed, test results (actual output), and any deviations from
  this brief with rationale.
