import json
import subprocess
from pathlib import Path


SIDECAR_MODULE = (
    Path(__file__).parents[4]
    / "plugins"
    / "platforms"
    / "photon"
    / "sidecar"
    / "conversation-actions.mjs"
)


def test_sidecar_exact_targets_replies_reactions_and_atomic_groups(tmp_path):
    script = tmp_path / "contract.mjs"
    script.write_text(
        f"""
import assert from "node:assert/strict";
import {{
  resolveMessageTarget,
  setExactReaction,
  sendExactReply,
  sendImageGroup,
}} from {json.dumps(SIDECAR_MODULE.as_uri())};

const cachedTarget = {{ id: "cached" }};
let getCalls = 0;
const recoveredTarget = {{
  id: "recovered",
  reacted: [],
  replies: [],
  async react(emoji) {{
    this.reacted.push(emoji);
    return {{ id: `reaction-${{emoji}}` }};
  }},
  async reply(...items) {{
    this.replies.push(items);
    return items.map((_, index) => ({{ id: `reply-${{index}}` }}));
  }},
}};
const space = {{
  async getMessage(id) {{
    getCalls += 1;
    return id === "persisted-guid" ? recoveredTarget : undefined;
  }},
}};

assert.equal(
  (await resolveMessageTarget(space, "cached-guid", new Map([["cached-guid", cachedTarget]]))).target,
  cachedTarget,
);
assert.equal(getCalls, 0);
assert.equal(
  (await resolveMessageTarget(space, "persisted-guid", new Map())).target,
  recoveredTarget,
);
assert.equal(getCalls, 1);

for (const emoji of ["❤️", "🫡"]) {{
  const reaction = await setExactReaction({{
    space,
    messageId: "persisted-guid",
    emoji,
    knownMessages: new Map(),
  }});
  assert.equal(reaction.found, true);
}}
assert.deepEqual(recoveredTarget.reacted, ["❤️", "🫡"]);

const reply = await sendExactReply({{
  space,
  messageId: "persisted-guid",
  knownMessages: new Map(),
  items: [{{ type: "text", text: "threaded" }}],
  text: (value) => ({{ kind: "text", value }}),
  attachment: () => {{ throw new Error("not used"); }},
}});
assert.deepEqual(reply.messageIds, ["reply-0"]);
assert.deepEqual(recoveredTarget.replies[0], [{{ kind: "text", value: "threaded" }}]);

const missing = await sendExactReply({{
  space,
  messageId: "missing-guid",
  knownMessages: new Map(),
  items: [{{ type: "text", text: "must not send" }}],
  text: (value) => value,
  attachment: () => null,
}});
assert.equal(missing.found, false);
assert.equal(recoveredTarget.replies.length, 1);

const visibleSends = [];
const order = [];
const grouped = await sendImageGroup({{
  space: {{
    async send(content) {{
      visibleSends.push(content);
      return {{
        id: "parent",
        content: {{
          type: "group",
          items: content.items.map((_, index) => ({{ id: `child-${{index}}` }})),
        }},
      }};
    }},
  }},
  images: [
    {{ path: "/tmp/one.png", name: "one.png", mimeType: "image/png" }},
    {{ path: "/tmp/two.jpg", name: "two.jpg", mimeType: "image/jpeg" }},
  ],
  caption: "compare",
  stat: async (path) => ({{ isFile: () => true }}),
  attachment: (path) => (order.push(path), {{ kind: "attachment", path }}),
  text: (value) => ({{ kind: "text", value }}),
  group: (...items) => ({{ type: "group", items }}),
}});
assert.deepEqual(order, ["/tmp/one.png", "/tmp/two.jpg"]);
assert.deepEqual(grouped, {{
  parentMessageId: "parent",
  childMessageIds: ["child-0", "child-1", "child-2"],
  partCount: 3,
}});
assert.equal(visibleSends.length, 1);

let failedVisibleSends = 0;
await assert.rejects(() => sendImageGroup({{
  space: {{ async send() {{ failedVisibleSends += 1; }} }},
  images: [{{ path: "ok.png" }}, {{ path: "bad.png" }}],
  stat: async (path) => {{
    if (path === "bad.png") throw new Error("upload preparation failed");
    return {{ isFile: () => true }};
  }},
  attachment: (path) => path,
  text: (value) => value,
  group: (...items) => items,
}}), /upload preparation failed/);
assert.equal(failedVisibleSends, 0);
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["node", str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
