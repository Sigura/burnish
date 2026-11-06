const ENDPOINT = "http://127.0.0.1:8765/refine";
const MODE = "all";
const MENU_ID = "burnish-selection";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Burnish selection",
    contexts: ["selection", "editable"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_ID) burnish(tab.id, info.frameId);
});

chrome.commands.onCommand.addListener((command, tab) => {
  if (command === MENU_ID) burnish(tab.id);
});

// The functions below run inside the page, so they can't use anything from this file's scope.

function captureSelection() {
  let el = document.activeElement;
  while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement;
  const isField =
    el && (el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && typeof el.selectionStart === "number"));
  if (isField) {
    window.__burnishSaved = { kind: "field", el, start: el.selectionStart, end: el.selectionEnd };
    return { text: el.value.slice(el.selectionStart, el.selectionEnd), focused: document.hasFocus() };
  }
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return { text: "", focused: document.hasFocus() };
  const anchor = selection.anchorNode;
  const anchorEl = anchor && (anchor.nodeType === Node.ELEMENT_NODE ? anchor : anchor.parentElement);
  const root = anchorEl && anchorEl.closest('[contenteditable=""], [contenteditable="true"], [role="textbox"]');
  window.__burnishSaved = { kind: "range", range: selection.getRangeAt(0).cloneRange(), root };
  return { text: selection.toString(), focused: document.hasFocus() };
}

async function applyResult(text) {
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.append(area);
      area.select();
      const copied = document.execCommand("copy");
      area.remove();
      if (!copied) return "failed";
    }
    return "copied";
  }

  const saved = window.__burnishSaved;
  if (saved.kind === "field") {
    saved.el.focus();
    saved.el.setSelectionRange(saved.start, saved.end);
    // insertText keeps undo history and is the only path React-controlled fields notice.
    if (!document.execCommand("insertText", false, text)) {
      saved.el.setRangeText(text, saved.start, saved.end, "end");
      saved.el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
    }
    return "replaced";
  }
  if (!saved.root) return copy();

  saved.root.focus();
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(saved.range);
  const needle = text.trim().split("\n")[0];
  const landed = () => saved.root.innerText.includes(needle);
  if (document.execCommand("insertText", false, text) && landed()) return "replaced";

  // Some rich editors ignore execCommand but do handle paste events.
  const data = new DataTransfer();
  data.setData("text/plain", text);
  saved.root.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
  return landed() ? "replaced" : copy();
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.textContent = message;
  toast.style.cssText =
    "position:fixed;right:16px;bottom:16px;z-index:2147483647;padding:8px 12px;border-radius:6px;" +
    "background:#222;color:#fff;font:13px sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.4)";
  document.documentElement.append(toast);
  setTimeout(() => toast.remove(), 4000);
}

async function toast(tabId, frameId, message) {
  await chrome.scripting.executeScript({
    target: { tabId, frameIds: [frameId] },
    func: showToast,
    args: [message],
  });
}

async function burnish(tabId, frameId) {
  const target = frameId === undefined ? { tabId, allFrames: true } : { tabId, frameIds: [frameId] };
  const frames = await chrome.scripting.executeScript({ target, func: captureSelection });
  const withText = frames.filter((f) => f.result && f.result.text.trim());
  const chosen = withText.find((f) => f.result.focused) || withText[0];
  if (!chosen) {
    await toast(tabId, frameId ?? 0, "Burnish: select some text first");
    return;
  }

  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: chosen.result.text, mode: MODE }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    const [applied] = await chrome.scripting.executeScript({
      target: { tabId, frameIds: [chosen.frameId] },
      func: applyResult,
      args: [body.text],
    });
    const outcome = {
      replaced: `Burnish: done (${body.model})`,
      copied: `Burnish: can't edit here, result copied (${body.model})`,
      failed: "Burnish: couldn't replace or copy the result",
    }[applied.result];
    await toast(tabId, chosen.frameId, outcome);
  } catch (e) {
    await toast(tabId, chosen.frameId, `Burnish: ${e.message}. Is the backend running?`);
  }
}
