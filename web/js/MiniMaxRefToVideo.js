import { app } from "../../scripts/app.js";

/* ================================================================
   MiniMaxRefToVideo.js  (手动引用模式)
   【台词】 + 空格/回车 → 绿色台词块 → <d>[Chinese] 台词。</d>
   切镜N  + 空格/回车 → 蓝色切镜块 → [Shot N] At MM:SS.mmm,
   @图片N + 空格/回车 → 橙色引用块 → <Picture N>  (对应 ref_image_N-1)
   @视频N + 空格/回车 → 橙色引用块 → <Video N>    (对应 ref_video_N-1)
   @音频N + 空格/回车 → 橙色引用块 → <Audio N>    (对应 ref_audio_N-1)
   无音乐 → non_diegetic_music:\nN/A
   无字幕 → 防字幕约束句
   ================================================================ */

const NODE_CLASS = "MiniMaxRefToVideo";
const PROMPT_DOC_PROP = "mmr_prompt_doc";
const DIALOGUE_CLASS = "mmr-dialogue-block";
const SHOT_CHIP_CLASS = "mmr-shot-chip";
const MENTION_CHIP_CLASS = "mmr-mention-chip";
const CHIP_SELECTOR = `.${MENTION_CHIP_CLASS}, .${SHOT_CHIP_CLASS}`;
const CARET_SENTINEL = "\u200B";
const PROMPT_HISTORY_LIMIT = 80;

const SHOT_TRIGGER_RE = /切镜\s*(\d+(?:\.\d+)?)$/;
const FALLBACK_SHOT_RE = /切镜\s*(\d+(?:\.\d+)?)\s*[，,]?\s*/g;
const BRACKET_DIALOGUE_RE = /【([^】]*)】/g;
const MENTION_TRIGGER_RE = /@(图片|视频|音频)(\d+)$/;

const KEYWORD_RULES = [
  {
    re: /不要背景音乐|无背景音乐|不要音乐|无音乐|无\s*BGM|不要\s*BGM/g,
    guard: /non_diegetic_music/i,
    replacement: "non_diegetic_music:\nN/A",
  },
  {
    re: /不要字幕|无字幕|不要出现字幕/g,
    guard: /no subtitles|无任何字幕/i,
    replacement: "画面严格保持干净，无任何字幕、屏幕文字、说明文字或水印。",
  },
];

const MENTION_TYPE_MAP = { "图片": "image", "视频": "video", "音频": "audio" };
const MENTION_TAG_MAP = { image: "Picture", video: "Video", audio: "Audio" };
const MENTION_ICON_MAP = { image: "🖼", video: "🎬", audio: "🎵" };

let installed = false;
let patchedPrompt = false;

/* ================================================================
   工具函数
   ================================================================ */

function isTarget(node) {
  return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "") === NODE_CLASS;
}

function getWidget(node, name) {
  return node?.widgets?.find((w) => w?.name === name) || null;
}

function padLeft(v, size) {
  return String(v).padStart(size, "0");
}

function formatShotTime(totalSeconds) {
  let ms = Math.max(0, Math.round(Number(totalSeconds) * 1000));
  const minutes = Math.floor(ms / 60000);
  ms -= minutes * 60000;
  const seconds = Math.floor(ms / 1000);
  ms -= seconds * 1000;
  return `${padLeft(minutes, 2)}:${padLeft(seconds, 2)}.${padLeft(ms, 3)}`;
}

function wrapDialogueTag(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return "";
  const withPunct = /[.?!。？!]$/.test(trimmed) ? trimmed : `${trimmed}。`;
  if (/^\[[^\]]+\]/.test(withPunct)) return `<d>${withPunct}</d>`;
  return `<d>[Chinese] ${withPunct}</d>`;
}

function postProcessPromptText(text) {
  let result = String(text || "");
  result = result.replace(BRACKET_DIALOGUE_RE, (m, inner) => wrapDialogueTag(inner));
  for (const rule of KEYWORD_RULES) {
    if (rule.guard.test(result)) {
      result = result.replace(rule.re, "");
      continue;
    }
    let first = true;
    result = result.replace(rule.re, () => {
      const v = first ? rule.replacement : "";
      first = false;
      return v;
    });
  }
  return result;
}

/* ================================================================
   DOM 辅助
   ================================================================ */

function makeCaretSentinel() {
  return document.createTextNode(CARET_SENTINEL);
}

function isCaretSentinelText(node) {
  return node?.nodeType === Node.TEXT_NODE && String(node.textContent || "").includes(CARET_SENTINEL);
}

function isOnlyCaretSentinelText(node) {
  return node?.nodeType === Node.TEXT_NODE && stripCaretSentinels(node.textContent) === "";
}

function stripCaretSentinels(value) {
  return String(value ?? "").replaceAll(CARET_SENTINEL, "");
}

function appendTextWithBreaks(container, value) {
  String(value || "").split("\n").forEach((part, i) => {
    if (i) container.append(document.createElement("br"));
    if (part) container.append(document.createTextNode(part));
  });
}

function setCaretAtNode(node, offset = 0) {
  const sel = window.getSelection?.();
  if (!sel || !node) return;
  const range = document.createRange();
  range.setStart(node, offset);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}

function setCaretAtEndOfNode(node) {
  if (!node) return;
  const sel = window.getSelection?.();
  if (!sel) return;
  const range = document.createRange();
  let target = node;
  while (target?.lastChild) target = target.lastChild;
  if (target?.nodeType === Node.TEXT_NODE) {
    range.setStart(target, target.textContent.length);
  } else if (target?.parentNode && target !== node) {
    range.setStartAfter(target);
  } else {
    range.setStart(node, node.childNodes.length);
  }
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}

function editorText(editor) {
  let result = "";
  const visit = (node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      result += String(node.textContent || "").replaceAll(CARET_SENTINEL, "");
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    if (node.classList?.contains(MENTION_CHIP_CLASS) || node.classList?.contains(SHOT_CHIP_CLASS)) {
      result += node.dataset.token || "";
      return;
    }
    if (node.tagName === "BR") { result += "\n"; return; }
    const block = ["DIV", "P"].includes(node.tagName);
    if (block && result && !result.endsWith("\n")) result += "\n";
    for (const child of node.childNodes || []) visit(child);
  };
  for (const child of editor.childNodes || []) visit(child);
  return result;
}

function insertPlainText(editor, text) {
  if (document.execCommand?.("insertText", false, text)) return;
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  range.deleteContents();
  const node = document.createTextNode(text);
  range.insertNode(node);
  range.setStartAfter(node);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}

function insertEditorLineBreak(editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount) return false;
  const range = sel.getRangeAt(0);
  if (!editor.contains(range.commonAncestorContainer)) return false;
  range.deleteContents();
  const br = document.createElement("br");
  const marker = document.createTextNode(CARET_SENTINEL);
  const frag = document.createDocumentFragment();
  frag.append(br, marker);
  range.insertNode(frag);
  const caret = document.createRange();
  caret.setStart(marker, marker.textContent.length);
  caret.collapse(true);
  sel.removeAllRanges();
  sel.addRange(caret);
  return true;
}

/* ================================================================
   台词块（绿色 + 💬）
   ================================================================ */

function isDialogueBlock(node) {
  return node?.nodeType === Node.ELEMENT_NODE && node.classList?.contains(DIALOGUE_CLASS);
}

function makeDialogueBlock(value = "") {
  const block = document.createElement("span");
  block.className = DIALOGUE_CLASS;
  block.spellcheck = false;
  block.dataset.dialogue = "true";
  appendTextWithBreaks(block, value);
  if (!String(value || "")) block.append(makeCaretSentinel());
  return block;
}

function dialogueBlockText(block) {
  return editorText(block);
}

function dialogueBlockAtSelection(editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount) return null;
  const container = sel.getRangeAt(0).startContainer;
  const element = container.nodeType === Node.ELEMENT_NODE ? container : container.parentElement;
  const block = element?.closest?.(`.${DIALOGUE_CLASS}`);
  return block && editor.contains(block) ? block : null;
}

function dialogueBoundary(block, side) {
  if (!block?.parentNode) return null;
  const sibling = side === "before" ? block.previousSibling : block.nextSibling;
  if (isCaretSentinelText(sibling)) return sibling;
  const marker = makeCaretSentinel();
  block.parentNode.insertBefore(marker, side === "before" ? block : block.nextSibling);
  return marker;
}

function exitDialogueBlock(node, editor, block) {
  const marker = dialogueBoundary(block, "after");
  if (!marker) return false;
  const text = String(marker.textContent || "");
  const idx = text.indexOf(CARET_SENTINEL);
  editor.focus({ preventScroll: true });
  setCaretAtNode(marker, idx >= 0 ? idx + CARET_SENTINEL.length : text.length);
  return true;
}

function insertDialogueBlockAtSelection(node, editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !editor) return false;
  const range = sel.getRangeAt(0);
  if (!editor.contains(range.commonAncestorContainer)) return false;
  if (dialogueBlockAtSelection(editor)) return false;
  range.deleteContents();
  const before = makeCaretSentinel();
  const block = makeDialogueBlock("");
  const after = makeCaretSentinel();
  const frag = document.createDocumentFragment();
  frag.append(before, block, after);
  range.insertNode(frag);
  editor.focus({ preventScroll: true });
  setCaretAtEndOfNode(block);
  return true;
}

function removeDialogueBlock(block) {
  if (!block?.parentNode) return false;
  const parent = block.parentNode;
  const before = block.previousSibling;
  const after = block.nextSibling;
  let marker = isCaretSentinelText(before) ? before : null;
  if (!marker) {
    marker = makeCaretSentinel();
    parent.insertBefore(marker, block);
  }
  block.remove();
  if (after !== marker && isOnlyCaretSentinelText(after)) after.remove();
  setCaretAtNode(marker, marker.textContent.length);
  return true;
}

function convertBracketsAtCaret(node, editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return false;
  const caret = sel.getRangeAt(0);
  const container = caret.startContainer;
  if (container.nodeType !== Node.TEXT_NODE || !editor.contains(container)) return false;
  if (container.parentElement?.closest?.(`.${DIALOGUE_CLASS}`)) return false;

  const textBefore = container.textContent.slice(0, caret.startOffset);
  const match = textBefore.match(/【([^】]*)】$/);
  if (!match) return false;

  const content = match[1];
  const startOffset = caret.startOffset - match[0].length;
  container.deleteData(startOffset, match[0].length);

  const range = document.createRange();
  range.setStart(container, startOffset);
  range.collapse(true);

  const before = makeCaretSentinel();
  const block = makeDialogueBlock(content);
  const after = makeCaretSentinel();
  const frag = document.createDocumentFragment();
  frag.append(before, block, after);
  range.insertNode(frag);

  setCaretAtNode(after, after.textContent.length);
  syncPromptFromEditor(node);
  pushPromptHistory(node);
  return true;
}

function convertLooseBrackets(node, editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return;
  const caret = sel.getRangeAt(0);
  const container = caret.startContainer;
  if (container.nodeType !== Node.TEXT_NODE || !editor.contains(container)) return;
  const insideDialogue = container.parentElement?.closest?.(`.${DIALOGUE_CLASS}`);
  const before = container.textContent.slice(0, caret.startOffset);
  if (insideDialogue && before.endsWith("】")) {
    container.deleteData(caret.startOffset - 1, 1);
    exitDialogueBlock(node, editor, insideDialogue);
    syncPromptFromEditor(node);
  }
}

/* ================================================================
   切镜块（蓝色 + ✂）
   ================================================================ */

function makeShotChip(secondsValue) {
  const seconds = Number(secondsValue) || 0;
  const chip = document.createElement("span");
  chip.className = SHOT_CHIP_CLASS;
  chip.contentEditable = "false";
  chip.dataset.seconds = String(seconds);
  chip.dataset.token = `切镜${seconds}`;

  const icon = document.createElement("span");
  icon.className = "mmr-chip-icon";
  icon.textContent = "✂";

  const label = document.createElement("span");
  label.className = "mmr-shot-chip-label";
  label.textContent = formatShotTime(seconds);

  chip.append(icon, label);
  chip.title = `切镜 → [Shot N] At ${formatShotTime(seconds)}`;

  chip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const sel = window.getSelection?.();
    if (!sel) return;
    const range = document.createRange();
    const rect = chip.getBoundingClientRect();
    const before = event.clientX < rect.left + rect.width / 2;
    before ? range.setStartBefore(chip) : range.setStartAfter(chip);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
  });
  return chip;
}

function getShotTriggerRange(editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return null;
  const caret = sel.getRangeAt(0);
  const container = caret.startContainer;
  if (container.nodeType !== Node.TEXT_NODE || !editor.contains(container)) return null;
  if (container.parentElement?.closest?.(`.${DIALOGUE_CLASS}`)) return null;
  const before = container.textContent.slice(0, caret.startOffset);
  const match = before.match(SHOT_TRIGGER_RE);
  if (!match) return null;
  const range = document.createRange();
  range.setStart(container, caret.startOffset - match[0].length);
  range.setEnd(container, caret.startOffset);
  return { range, seconds: Number(match[1]) };
}

function validateShotChips(editor) {
  const chips = editor?.querySelectorAll?.(`.${SHOT_CHIP_CLASS}`) || [];
  let previous = -Infinity;
  for (const chip of chips) {
    const seconds = Number(chip.dataset.seconds);
    chip.classList.toggle("is-warning", Number.isFinite(seconds) && seconds <= previous);
    if (Number.isFinite(seconds)) previous = seconds;
  }
}

/* ================================================================
   引用块（淡橙色 + 🖼/🎬/🎵）— 手动模式
   ================================================================ */

function isMentionChip(node) {
  return node?.nodeType === Node.ELEMENT_NODE
    && (node.classList?.contains(MENTION_CHIP_CLASS) || node.classList?.contains(SHOT_CHIP_CLASS));
}

function makeMentionChip(option) {
  const chip = document.createElement("span");
  chip.className = MENTION_CHIP_CLASS;
  chip.contentEditable = "false";
  chip.dataset.token = option.token || option.tag || "";
  chip.dataset.label = option.label || "";
  chip.dataset.ordinal = String(option.ordinal || "");
  chip.dataset.mediaType = option.type || "image";
  chip.title = option.tag || "";

  const icon = document.createElement("span");
  icon.className = "mmr-chip-icon";
  icon.textContent = MENTION_ICON_MAP[option.type] || "🖼";

  const label = document.createElement("span");
  label.className = "mmr-mention-chip-label";
  label.textContent = `@${option.label || ""}`;

  chip.append(icon, label);

  chip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const sel = window.getSelection?.();
    if (!sel) return;
    const range = document.createRange();
    const rect = chip.getBoundingClientRect();
    const before = event.clientX < rect.left + rect.width / 2;
    before ? range.setStartBefore(chip) : range.setStartAfter(chip);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
  });
  return chip;
}

function convertMentionAtCaret(node, editor) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return false;
  const caret = sel.getRangeAt(0);
  const container = caret.startContainer;
  if (container.nodeType !== Node.TEXT_NODE || !editor.contains(container)) return false;
  if (container.parentElement?.closest?.(`.${DIALOGUE_CLASS}`)) return false;

  const textBefore = container.textContent.slice(0, caret.startOffset);
  const match = textBefore.match(MENTION_TRIGGER_RE);
  if (!match) return false;

  const type = MENTION_TYPE_MAP[match[1]];
  const ordinal = parseInt(match[2], 10);
  const tag = `<${MENTION_TAG_MAP[type]} ${ordinal}>`;
  const token = `@${match[1]}${match[2]}`;

  const startOffset = caret.startOffset - match[0].length;
  container.deleteData(startOffset, match[0].length);

  const range = document.createRange();
  range.setStart(container, startOffset);
  range.collapse(true);

  const before = makeCaretSentinel();
  const chip = makeMentionChip({ type, ordinal, tag, token, label: `${match[1]}${match[2]}` });
  const after = makeCaretSentinel();
  const frag = document.createDocumentFragment();
  frag.append(before, chip, after);
  range.insertNode(frag);

  setCaretAtNode(after, after.textContent.length);
  syncPromptFromEditor(node);
  pushPromptHistory(node);
  return true;
}

/* ================================================================
   序列化 / 反序列化
   ================================================================ */

function serializeEditorDoc(editor) {
  const parts = [];
  const pushText = (text) => {
    const value = String(text || "").replaceAll(CARET_SENTINEL, "");
    if (!value) return;
    if (parts.at(-1)?.type === "text") parts[parts.length - 1].text += value;
    else parts.push({ type: "text", text: value });
  };
  const visit = (item) => {
    if (item.nodeType === Node.TEXT_NODE) { pushText(item.textContent); return; }
    if (item.nodeType !== Node.ELEMENT_NODE) return;
    if (isDialogueBlock(item)) {
      parts.push({ type: "dialogue", text: dialogueBlockText(item) });
      return;
    }
    if (item.classList?.contains(SHOT_CHIP_CLASS)) {
      parts.push({ type: "shot", seconds: Number(item.dataset.seconds) || 0 });
      return;
    }
    if (item.classList?.contains(MENTION_CHIP_CLASS)) {
      parts.push({
        type: "mention",
        token: item.dataset.token || "",
        label: item.dataset.label || "",
        ordinal: Number(item.dataset.ordinal) || null,
        mediaType: item.dataset.mediaType || "image",
      });
      return;
    }
    if (item.tagName === "BR") { pushText("\n"); return; }
    const block = ["DIV", "P"].includes(item.tagName);
    if (block && parts.length && !(parts.at(-1)?.type === "text" && parts.at(-1).text.endsWith("\n"))) pushText("\n");
    for (const child of item.childNodes || []) visit(child);
  };
  for (const child of editor.childNodes || []) visit(child);
  return {
    version: 1,
    text: parts.map((p) => {
      if (p.type === "mention") return p.token;
      if (p.type === "dialogue") return `<d>${p.text || ""}</d>`;
      if (p.type === "shot") return `切镜${p.seconds}`;
      return p.text;
    }).join(""),
    parts,
  };
}

function appendDialogueBlock(container, value = "") {
  container.append(makeCaretSentinel(), makeDialogueBlock(value), makeCaretSentinel());
}

function appendPromptTextWithDialogueBlocks(container, value) {
  const source = String(value || "");
  const pattern = /<d>([\s\S]*?)<\/d>/gi;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(source))) {
    appendTextWithBreaks(container, source.slice(cursor, match.index));
    appendDialogueBlock(container, match[1]);
    cursor = match.index + match[0].length;
  }
  appendTextWithBreaks(container, source.slice(cursor));
}

function renderEditorFromNode(node, force = false) {
  const editor = node?.__mmrEditor;
  const widget = getWidget(node, "prompt");
  if (!editor || !widget || (document.activeElement === editor && !force)) return;
  const doc = node.properties?.[PROMPT_DOC_PROP];
  editor.textContent = "";
  if (!Array.isArray(doc?.parts)) {
    appendPromptTextWithDialogueBlocks(editor, String(widget.value || ""));
    return;
  }
  for (const part of doc.parts) {
    if (part?.type === "dialogue") {
      appendDialogueBlock(editor, String(part.text || ""));
      continue;
    }
    if (part?.type === "shot") {
      editor.append(makeShotChip(Number(part.seconds) || 0));
      continue;
    }
    if (part?.type === "mention") {
      editor.append(makeMentionChip({
        type: part.mediaType || "image",
        ordinal: part.ordinal,
        tag: part.token || "",
        token: part.token || "",
        label: part.label || "",
      }));
      continue;
    }
    appendTextWithBreaks(editor, part?.text || "");
  }
  validateShotChips(editor);
}

function syncPromptFromEditor(node, markDirty = true) {
  const editor = node?.__mmrEditor;
  const widget = getWidget(node, "prompt");
  if (!editor || !widget || node.__mmrEditorSyncing) return;
  node.__mmrEditorSyncing = true;
  try {
    const doc = serializeEditorDoc(editor);
    widget.value = doc.text;
    if (widget._state) widget._state.value = doc.text;
    node.properties ||= {};
    node.properties[PROMPT_DOC_PROP] = doc;
    validateShotChips(editor);
    if (markDirty) {
      node.setDirtyCanvas?.(true, true);
      app.graph?.setDirtyCanvas?.(true, true);
      app.graph?.change?.();
    }
  } finally {
    node.__mmrEditorSyncing = false;
  }
}

/* ================================================================
   buildRuntimePrompt — 核心输出
   ================================================================ */

function buildRuntimePrompt(node) {
  const promptWidget = getWidget(node, "prompt");
  const fallback = String(promptWidget?.value || "");
  const doc = node?.properties?.[PROMPT_DOC_PROP];
  if (!Array.isArray(doc?.parts)) return postProcessPromptText(fallback);

  let shotIndex = 1;
  const emitShot = (seconds) => {
    shotIndex += 1;
    return `[Shot ${shotIndex}] At ${formatShotTime(seconds)}, `;
  };

  const pieces = doc.parts.map((part) => {
    if (part?.type === "dialogue") return wrapDialogueTag(part.text);
    if (part?.type === "shot") return emitShot(Number(part.seconds) || 0);
    if (part?.type === "mention") {
      const prefix = MENTION_TAG_MAP[part.mediaType] || "Picture";
      return `<${prefix} ${part.ordinal}>`;
    }
    return String(part?.text || "").replace(FALLBACK_SHOT_RE, (m, s) => emitShot(Number(s)));
  });

  return postProcessPromptText(pieces.join(""));
}

/* ================================================================
   撤销 / 重做
   ================================================================ */

function clonePromptDoc(doc) {
  const source = doc && typeof doc === "object" ? doc : {};
  return {
    version: 1,
    text: String(source.text || ""),
    parts: Array.isArray(source.parts) ? source.parts.map((p) => ({ ...p })) : [],
  };
}

function promptDocKey(doc) {
  return JSON.stringify(clonePromptDoc(doc));
}

function ensurePromptHistory(node) {
  const editor = node?.__mmrEditor;
  if (!editor) return null;
  if (node.__mmrPromptHistory) return node.__mmrPromptHistory;
  const doc = clonePromptDoc(serializeEditorDoc(editor));
  node.__mmrPromptHistory = { undo: [{ doc }], redo: [], lastKey: promptDocKey(doc), applying: false };
  return node.__mmrPromptHistory;
}

function resetPromptHistory(node) {
  node.__mmrPromptHistory = null;
  ensurePromptHistory(node);
}

function pushPromptHistory(node) {
  const history = ensurePromptHistory(node);
  const editor = node?.__mmrEditor;
  if (!history || !editor || history.applying) return;
  const doc = clonePromptDoc(serializeEditorDoc(editor));
  const key = promptDocKey(doc);
  if (key === history.lastKey) return;
  history.undo.push({ doc });
  if (history.undo.length > PROMPT_HISTORY_LIMIT) history.undo.shift();
  history.redo = [];
  history.lastKey = key;
}

function isPromptUndoRedoEvent(event) {
  if (!(event?.ctrlKey || event?.metaKey)) return false;
  const key = String(event.key || "").toLowerCase();
  const code = String(event.code || "");
  return key === "z" || key === "y" || code === "KeyZ" || code === "KeyY";
}

function setEditorCaretAtEnd(editor) {
  if (!editor) return;
  const sel = window.getSelection?.();
  if (!sel) return;
  const range = document.createRange();
  range.selectNodeContents(editor);
  range.collapse(false);
  sel.removeAllRanges();
  sel.addRange(range);
}

function applyPromptHistoryEntry(node, entry) {
  const history = node?.__mmrPromptHistory;
  const editor = node?.__mmrEditor;
  const widget = getWidget(node, "prompt");
  if (!history || !editor || !entry?.doc || !widget) return false;
  history.applying = true;
  try {
    const doc = clonePromptDoc(entry.doc);
    node.properties ||= {};
    node.properties[PROMPT_DOC_PROP] = doc;
    widget.value = doc.text;
    if (widget._state) widget._state.value = doc.text;
    renderEditorFromNode(node, true);
    syncPromptFromEditor(node, false);
    history.lastKey = promptDocKey(doc);
  } finally {
    history.applying = false;
  }
  editor.focus();
  setEditorCaretAtEnd(editor);
  return true;
}

function handlePromptHistoryKeydown(node, event) {
  if (!isPromptUndoRedoEvent(event)) return false;
  event.preventDefault?.();
  event.stopPropagation?.();
  event.stopImmediatePropagation?.();
  const history = ensurePromptHistory(node);
  if (!history) return true;
  const key = String(event.key || "").toLowerCase();
  const isRedo = key === "y" || String(event.code || "") === "KeyY" || (key === "z" && event.shiftKey);
  if (isRedo) {
    const entry = history.redo.pop();
    if (!entry) return true;
    history.undo.push(entry);
    applyPromptHistoryEntry(node, entry);
    return true;
  }
  if (history.undo.length <= 1) return true;
  const current = history.undo.pop();
  if (current) history.redo.push(current);
  applyPromptHistoryEntry(node, history.undo[history.undo.length - 1]);
  return true;
}

/* ================================================================
   粘贴处理
   ================================================================ */

function appendPastedText(fragment, text) {
  String(text || "").split("\n").forEach((part, i) => {
    if (i) fragment.append(document.createElement("br"));
    if (part) fragment.append(document.createTextNode(part));
  });
}

function insertTextWithMentionChips(node, editor, text) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !editor.contains(sel.anchorNode)) return false;
  const range = sel.getRangeAt(0);
  const value = String(text || "");
  if (!value) return false;
  range.deleteContents();
  const fragment = document.createDocumentFragment();

  const SPECIAL = /【[^】]*】|切镜\s*\d+(?:\.\d+)?|@(?:图片|视频|音频)\d+/g;
  let lastIndex = 0;
  let match;
  while ((match = SPECIAL.exec(value))) {
    if (match.index > lastIndex) appendPastedText(fragment, value.slice(lastIndex, match.index));
    const token = match[0];
    fragment.append(document.createTextNode(CARET_SENTINEL));
    if (token.startsWith("【")) {
      fragment.append(makeDialogueBlock(token.slice(1, -1)));
    } else if (token.startsWith("@")) {
      const m = token.match(/@(图片|视频|音频)(\d+)/);
      if (m) {
        const type = MENTION_TYPE_MAP[m[1]];
        const ordinal = parseInt(m[2], 10);
        const tag = `<${MENTION_TAG_MAP[type]} ${ordinal}>`;
        fragment.append(makeMentionChip({ type, ordinal, tag, token, label: `${m[1]}${m[2]}` }));
      }
    } else {
      const numeric = token.match(/\d+(?:\.\d+)?/);
      fragment.append(makeShotChip(Number(numeric ? numeric[0] : 0)));
    }
    fragment.append(document.createTextNode(CARET_SENTINEL));
    lastIndex = match.index + token.length;
  }
  if (lastIndex < value.length) appendPastedText(fragment, value.slice(lastIndex));

  const caretMarker = document.createTextNode(CARET_SENTINEL);
  fragment.append(caretMarker);
  range.insertNode(fragment);
  const caret = document.createRange();
  caret.setStart(caretMarker, caretMarker.textContent.length);
  caret.collapse(true);
  sel.removeAllRanges();
  sel.addRange(caret);
  return true;
}

/* ================================================================
   删除处理
   ================================================================ */

function removeChip(chip, direction = "backward") {
  if (!chip?.parentNode) return null;
  const marker = makeCaretSentinel();
  chip.parentNode.insertBefore(marker, direction === "backward" ? chip : chip.nextSibling);
  chip.remove();
  return marker;
}

function deleteChipNearCaret(editor, node, direction) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return false;
  const range = sel.getRangeAt(0);
  const editorNode = range.startContainer;
  if (!editor.contains(editorNode)) return false;
  const directChip = editorNode.nodeType === Node.ELEMENT_NODE
    ? editorNode.closest?.(CHIP_SELECTOR)
    : editorNode.parentElement?.closest?.(CHIP_SELECTOR);
  if (directChip && editor.contains(directChip)) {
    const marker = removeChip(directChip, direction);
    setCaretAtNode(marker, marker.textContent.length);
    return true;
  }
  return false;
}

function backspaceDialogueBoundary(editor, node) {
  const sel = window.getSelection?.();
  if (!sel || !sel.rangeCount || !sel.isCollapsed) return false;
  const activeBlock = dialogueBlockAtSelection(editor);
  if (activeBlock) {
    if (!dialogueBlockText(activeBlock)) {
      const removed = removeDialogueBlock(activeBlock);
      return removed;
    }
    return false;
  }
  return false;
}

/* ================================================================
   编辑器创建
   ================================================================ */

function hideOriginalPromptWidget(widget) {
  if (!widget) return;
  if (!widget.__mmrPromptHidden) {
    widget.__mmrPromptHidden = true;
    widget.__mmrOriginalType = widget.type;
    widget.__mmrOriginalComputeSize = widget.computeSize;
  }
  widget.hidden = true;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function ensurePromptEditor(node) {
  if (node.__mmrEditor) return;
  if (typeof document === "undefined" || typeof node.addDOMWidget !== "function") return;
  const widget = getWidget(node, "prompt");
  if (!widget) return;
  hideOriginalPromptWidget(widget);

  const wrap = document.createElement("div");
  wrap.className = "mmr-prompt-editor-wrap";
  wrap.style.minHeight = "0px";

  const editor = document.createElement("div");
  editor.className = "comfy-multiline-input mmr-prompt-editor";
  editor.contentEditable = "true";
  editor.__mmrPromptNode = node;
  editor.tabIndex = 0;
  editor.setAttribute("role", "textbox");
  editor.setAttribute("aria-label", "prompt");
  editor.dataset.placeholder = "【】台词 | 切镜3.5 | @图片1 @视频1 @音频1 → 按空格/回车确认";
  editor.spellcheck = false;

  editor.addEventListener("beforeinput", (event) => {
    if (node.__mmrDialogueHashHandled) {
      node.__mmrDialogueHashHandled = false;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
      return;
    }
    if (event.inputType === "insertText" && event.data === "#") {
      if (insertDialogueBlockAtSelection(node, editor)) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        syncPromptFromEditor(node);
        pushPromptHistory(node);
        return;
      }
    }
    if (event.inputType === "insertText" && event.data === "】") {
      const activeBlock = dialogueBlockAtSelection(editor);
      if (activeBlock) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        exitDialogueBlock(node, editor, activeBlock);
        syncPromptFromEditor(node);
        pushPromptHistory(node);
        return;
      }
    }
  });

  editor.addEventListener("input", (event) => {
    syncPromptFromEditor(node);
    if (event?.isComposing || event?.inputType === "insertCompositionText" || node.__mmrPromptComposing) return;
    if (!node.__mmrPromptComposing) convertLooseBrackets(node, editor);
    pushPromptHistory(node);
  });

  editor.addEventListener("compositionstart", () => { node.__mmrPromptComposing = true; });
  editor.addEventListener("compositionend", () => {
    node.__mmrPromptComposing = false;
    syncPromptFromEditor(node);
    pushPromptHistory(node);
  });

  editor.addEventListener("keydown", (event) => {
    if (isPromptUndoRedoEvent(event)) handlePromptHistoryKeydown(node, event);
  }, true);

  editor.addEventListener("keydown", (event) => {
    /* 回车/空格：先【台词】，再 @引用，再 切镜 */
    if ((event.key === " " || event.key === "Enter") && !node.__mmrPromptComposing) {
      if (convertBracketsAtCaret(node, editor)) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (convertMentionAtCaret(node, editor)) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
    }

    if ((event.key === " " || event.key === "Enter") && !node.__mmrPromptComposing && !dialogueBlockAtSelection(editor)) {
      const trigger = getShotTriggerRange(editor);
      if (trigger) {
        event.preventDefault();
        event.stopPropagation();
        trigger.range.deleteContents();
        const before = document.createTextNode(CARET_SENTINEL);
        const chip = makeShotChip(trigger.seconds);
        const after = document.createTextNode(CARET_SENTINEL);
        const frag = document.createDocumentFragment();
        frag.append(before, chip, after);
        trigger.range.insertNode(frag);
        const sel = window.getSelection?.();
        if (sel) {
          const caret = document.createRange();
          caret.setStart(after, after.textContent.length);
          caret.collapse(true);
          sel.removeAllRanges();
          sel.addRange(caret);
        }
        if (event.key === " ") insertPlainText(editor, " ");
        else insertEditorLineBreak(editor);
        syncPromptFromEditor(node);
        pushPromptHistory(node);
        return;
      }
    }

    if (event.key === "#" && !event.ctrlKey && !event.metaKey && !event.altKey && insertDialogueBlockAtSelection(node, editor)) {
      event.preventDefault();
      event.stopPropagation();
      node.__mmrDialogueHashHandled = true;
      setTimeout(() => { node.__mmrDialogueHashHandled = false; }, 0);
      syncPromptFromEditor(node);
      pushPromptHistory(node);
      return;
    }

    const dialogue = dialogueBlockAtSelection(editor);
    if (event.key === "Enter" && dialogue && !event.shiftKey) {
      event.preventDefault();
      event.stopPropagation();
      exitDialogueBlock(node, editor, dialogue);
      syncPromptFromEditor(node);
      pushPromptHistory(node);
      return;
    }
    if (event.key === "Enter" && dialogue && event.shiftKey && insertEditorLineBreak(editor)) {
      event.preventDefault();
      event.stopPropagation();
      syncPromptFromEditor(node);
      pushPromptHistory(node);
      return;
    }

    if (event.key === "Backspace" && (
      backspaceDialogueBoundary(editor, node)
      || deleteChipNearCaret(editor, node, "backward")
    )) {
      event.preventDefault();
      syncPromptFromEditor(node);
      pushPromptHistory(node);
    } else if (event.key === "Delete" && deleteChipNearCaret(editor, node, "forward")) {
      event.preventDefault();
      syncPromptFromEditor(node);
      pushPromptHistory(node);
    } else if (event.key === "Enter" && insertEditorLineBreak(editor)) {
      event.preventDefault();
      syncPromptFromEditor(node);
      pushPromptHistory(node);
    }
    event.stopPropagation();
  });

  editor.addEventListener("paste", (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();
    insertTextWithMentionChips(node, editor, event.clipboardData?.getData("text/plain") || "");
    syncPromptFromEditor(node);
    pushPromptHistory(node);
  });

  editor.addEventListener("blur", () => {
    syncPromptFromEditor(node);
  });

  wrap.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
  });

  wrap.append(editor);
  node.__mmrEditor = editor;
  node.__mmrEditorWrap = wrap;
  renderEditorFromNode(node);
  resetPromptHistory(node);

  const domWidget = node.addDOMWidget("mmr_prompt_editor", "mmr_prompt_editor", wrap, {
    getValue: () => String(getWidget(node, "prompt")?.value || ""),
    setValue: (value) => {
      const promptWidget = getWidget(node, "prompt");
      if (promptWidget) promptWidget.value = String(value || "");
      renderEditorFromNode(node);
    },
    margin: 10,
    serialize: false,
    getMinHeight: () => 50,
    afterResize: () => {
      node.setDirtyCanvas?.(true, true);
    },
  });

  if (!domWidget) {
    wrap.remove();
    node.__mmrEditor = null;
    node.__mmrEditorWrap = null;
    return;
  }

  node.__mmrDomWidget = domWidget;
  domWidget.serialize = false;

  const domIndex = node.widgets?.indexOf(domWidget) ?? -1;
  const promptIndex = node.widgets?.indexOf(widget) ?? -1;
  if (domIndex >= 0 && promptIndex >= 0 && domIndex !== promptIndex + 1) {
    node.widgets.splice(domIndex, 1);
    const nextPromptIndex = node.widgets.indexOf(widget);
    node.widgets.splice(nextPromptIndex + 1, 0, domWidget);
  }

  node.setDirtyCanvas?.(true, true);
}

/* ================================================================
   graphToPrompt 补丁
   ================================================================ */

function patchGraphToPrompt() {
  if (patchedPrompt || typeof app.graphToPrompt !== "function") return;
  patchedPrompt = true;
  const original = app.graphToPrompt;
  app.graphToPrompt = async function graphToPromptWithMMREditor() {
    const promptData = await original.apply(this, arguments);
    const output = promptData?.output || {};
    for (const node of app.graph?._nodes || []) {
      if (!isTarget(node)) continue;
      const promptNode = output[String(node.id)];
      if (!promptNode) continue;
      promptNode.inputs ||= {};
      if (node.__mmrEditor) syncPromptFromEditor(node, false);
      promptNode.inputs.prompt = buildRuntimePrompt(node);
    }
    return promptData;
  };
}

/* ================================================================
   样式
   ================================================================ */

function installStyles() {
  const style = document.createElement("style");
  style.textContent = `
.mmr-prompt-editor-wrap {
  position: relative; display: block; width: 100%; height: 100%;
  min-width: 0; min-height: 0; max-height: 100%;
  box-sizing: border-box; padding: 0; overflow: hidden;
}
.mmr-prompt-editor {
  --mmr-text-size: 12px;
  display: block; width: 100%; height: 100%;
  min-width: 0; min-height: 0; max-height: 100%;
  box-sizing: border-box; padding: 4px;
  overflow-y: auto; overflow-x: hidden; overscroll-behavior: contain;
  white-space: pre-wrap; overflow-wrap: anywhere;
  border: 0; outline: none; resize: none;
  background-color: #222; color: #ddd; caret-color: #ddd;
  font-family: Consolas, "Courier New", monospace;
  font-size: var(--mmr-text-size); font-weight: 400;
  line-height: 1.4; letter-spacing: 0;
}
.mmr-prompt-editor:empty::before {
  content: attr(data-placeholder);
  color: rgba(255,255,255,.35); pointer-events: none;
}

/* 台词块：绿色 + 💬 图标 */
.mmr-dialogue-block {
  display: inline; margin: 0 1px; padding: 2px 5px; vertical-align: 1px;
  border-radius: 4px;
  background: rgba(80, 200, 120, .16); color: #d4f5dd;
  box-shadow: inset 0 0 0 1px rgba(80, 200, 120, .3);
  font-family: Consolas, "Courier New", monospace;
  font-size: var(--mmr-text-size); line-height: calc(1em + 6px);
  white-space: pre-wrap; user-select: text; cursor: text; outline: none;
}
.mmr-dialogue-block::before {
  content: "💬 ";
  font-size: 0.9em; opacity: 0.75;
}
.mmr-dialogue-block:focus {
  background: rgba(80, 200, 120, .22);
  box-shadow: inset 0 0 0 1px rgba(80, 200, 120, .42);
}

/* 切镜块：蓝色 + ✂ 图标 */
.mmr-shot-chip {
  display: inline; margin: 0 2px; padding: 2px 6px; vertical-align: 1px;
  border-radius: 4px;
  background: rgba(90, 169, 240, .16); color: #9ccaff;
  box-shadow: inset 0 0 0 1px rgba(90, 169, 240, .38);
  font-family: Consolas, monospace; font-size: var(--mmr-text-size);
  line-height: calc(1em + 6px); white-space: nowrap;
  user-select: none; cursor: default;
}
.mmr-shot-chip.is-warning {
  background: rgba(255,110,110,.14); color: #ffb4a8;
  box-shadow: inset 0 0 0 1px rgba(255,110,110,.55);
}

/* 引用块：淡橙色 + 🖼/🎬/🎵 图标 */
.mmr-mention-chip {
  display: inline; margin: 0 2px; padding: 2px 6px; vertical-align: 1px;
  border-radius: 4px;
  background: rgba(255, 178, 102, .18); color: #ffd9a8;
  box-shadow: inset 0 0 0 1px rgba(255, 178, 102, .4);
  font-family: Consolas, monospace; font-size: var(--mmr-text-size);
  line-height: calc(1em + 6px); white-space: nowrap;
  user-select: none; cursor: default;
}

/* 小图标通用样式 */
.mmr-chip-icon {
  display: inline-block; margin-right: 3px; font-size: 0.9em; opacity: 0.8;
}
`;
  document.head.append(style);
}

/* ================================================================
   节点安装
   ================================================================ */

function installNode(nodeType, nodeData) {
  if (nodeData?.name !== NODE_CLASS) return;
  if (nodeType.prototype.__mmrNodeInstalled) return;
  nodeType.prototype.__mmrNodeInstalled = true;

  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function onNodeCreatedMMR() {
    const result = originalCreated?.apply(this, arguments);
    this.properties ||= {};
    ensurePromptEditor(this);
    return result;
  };

  const originalConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function onConfigureMMR(info) {
    const result = originalConfigure?.apply(this, arguments);
    if (info?.properties?.[PROMPT_DOC_PROP]) {
      this.properties ||= {};
      this.properties[PROMPT_DOC_PROP] = info.properties[PROMPT_DOC_PROP];
    }
    renderEditorFromNode(this);
    resetPromptHistory(this);
    ensurePromptEditor(this);
    return result;
  };

  const originalSerialize = nodeType.prototype.onSerialize;
  nodeType.prototype.onSerialize = function onSerializeMMR(info) {
    if (this.__mmrEditor) syncPromptFromEditor(this, false);
    const result = originalSerialize?.apply(this, arguments);
    if (info && this.properties?.[PROMPT_DOC_PROP]) {
      info.properties ||= {};
      info.properties[PROMPT_DOC_PROP] = this.properties[PROMPT_DOC_PROP];
    }
    return result;
  };

  const originalRemoved = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function onRemovedMMR() {
    this.__mmrEditorWrap?.remove?.();
    this.__mmrEditor = null;
    this.__mmrEditorWrap = null;
    this.__mmrDomWidget = null;
    return originalRemoved?.apply(this, arguments);
  };
}

/* ================================================================
   扩展注册
   ================================================================ */

app.registerExtension({
  name: "MiniMaxRefToVideo",
  setup() {
    if (installed) return;
    installed = true;
    patchGraphToPrompt();
    installStyles();
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    installNode(nodeType, nodeData);
  },
});