import fs from "node:fs/promises";

export function createPerSpaceSerializer() {
  const tails = new Map();
  return async function serialize(spaceId, operation) {
    const previous = tails.get(spaceId) ?? Promise.resolve();
    const current = previous.catch(() => undefined).then(operation);
    tails.set(spaceId, current);
    try {
      return await current;
    } finally {
      if (tails.get(spaceId) === current) tails.delete(spaceId);
    }
  };
}

function requireString(value, name) {
  if (typeof value !== "string" || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return value;
}

function attachmentOptions(item) {
  const options = {};
  if (item.name) options.name = item.name;
  if (item.mimeType) options.mimeType = item.mimeType;
  return Object.keys(options).length ? options : undefined;
}

async function validateFiles(items, stat = fs.stat) {
  await Promise.all(
    items.map(async (item) => {
      const path = requireString(item.path, "attachment path");
      const info = await stat(path);
      if (!info?.isFile?.()) {
        throw new TypeError("attachment path must identify a file");
      }
    })
  );
}

export async function resolveMessageTarget(space, messageId, knownMessages) {
  requireString(messageId, "messageId");
  const cached = knownMessages?.get(messageId);
  if (cached) return { target: cached, source: "cache" };

  const target = await space.getMessage(messageId);
  if (!target) return { target: null, source: "space" };
  return { target, source: "space" };
}

export async function setExactReaction({
  space,
  messageId,
  emoji,
  knownMessages,
}) {
  requireString(emoji, "emoji");
  const { target, source } = await resolveMessageTarget(
    space,
    messageId,
    knownMessages
  );
  if (!target) return { found: false, source, handle: null };
  const handle = await target.react(emoji);
  if (!handle) throw new Error("reactions are not supported for this target");
  return { found: true, source, handle };
}

export async function sendExactReply({
  space,
  messageId,
  knownMessages,
  items,
  text,
  attachment,
  stat = fs.stat,
}) {
  if (!Array.isArray(items) || items.length === 0) {
    throw new TypeError("reply items are required");
  }
  const attachments = items.filter((item) => item?.type === "attachment");
  await validateFiles(attachments, stat);

  const builders = items.map((item) => {
    if (item?.type === "text") {
      return text(requireString(item.text, "reply text"));
    }
    if (item?.type === "attachment") {
      return attachment(
        requireString(item.path, "attachment path"),
        attachmentOptions(item)
      );
    }
    throw new TypeError("reply items must be text or attachment");
  });

  const { target, source } = await resolveMessageTarget(
    space,
    messageId,
    knownMessages
  );
  if (!target) return { found: false, source, messageIds: [] };

  const sent = await target.reply(...builders);
  const messages = Array.isArray(sent) ? sent : sent ? [sent] : [];
  if (messages.length !== builders.length) {
    throw new Error("reply operation did not return every sent message");
  }
  return {
    found: true,
    source,
    messageIds: messages.map((message) => message.id).filter(Boolean),
  };
}

export async function sendImageGroup({
  space,
  images,
  caption,
  attachment,
  text,
  group,
  stat = fs.stat,
}) {
  if (!Array.isArray(images) || images.length < 2 || images.length > 5) {
    throw new TypeError("image groups require 2 to 5 images");
  }
  await validateFiles(images, stat);

  const builders = images.map((image) =>
    attachment(
      requireString(image.path, "image path"),
      attachmentOptions(image)
    )
  );
  if (typeof caption === "string" && caption.trim()) {
    builders.push(text(caption.trim()));
  }

  const sent = await space.send(group(...builders));
  if (!sent?.id || sent?.content?.type !== "group") {
    throw new Error("group operation did not return a multipart message");
  }
  const children = Array.isArray(sent.content.items) ? sent.content.items : [];
  if (children.length !== builders.length || children.some((item) => !item?.id)) {
    throw new Error("group operation did not return every child message");
  }
  return {
    parentMessageId: sent.id,
    childMessageIds: children.map((item) => item.id),
    partCount: children.length,
  };
}

export async function sendTextBatch({ space, chunks, buildContent }) {
  if (!Array.isArray(chunks) || chunks.length < 2) {
    throw new TypeError("text batches require at least two chunks");
  }
  if (chunks.some((chunk) => typeof chunk !== "string" || !chunk)) {
    throw new TypeError("text batch chunks must be non-empty strings");
  }

  const messageIds = [];
  for (let index = 0; index < chunks.length; index += 1) {
    try {
      const message = await space.send(buildContent(chunks[index]));
      if (!message?.id) {
        throw new Error("text chunk did not return a message identifier");
      }
      messageIds.push(message.id);
    } catch (error) {
      return {
        complete: false,
        messageIds,
        failedIndex: index,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
  return { complete: true, messageIds };
}
