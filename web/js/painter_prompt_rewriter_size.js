import { app } from "../../scripts/app.js";

const NODE_TYPE = "PainterPromptRewriter";
const PROP_KEY = "__painter_prompt_rewriter_size";
const STORAGE_KEY = "PainterPromptRewriter:lastNodeSize";

let applyingSavedSize = false;
let configuringGraph = false;

function isValidSize(size) {
  return (
    Array.isArray(size) &&
    size.length === 2 &&
    Number.isFinite(size[0]) &&
    Number.isFinite(size[1]) &&
    size[0] >= 80 &&
    size[1] >= 40
  );
}

function normalizeSize(size) {
  if (!size) return null;

  let w, h;

  if (Array.isArray(size)) {
    [w, h] = size;
  } else {
    w = size[0] ?? size.w ?? size.width;
    h = size[1] ?? size.h ?? size.height;
  }

  w = Math.round(Number(w));
  h = Math.round(Number(h));

  const out = [w, h];
  return isValidSize(out) ? out : null;
}

function readGlobalSize() {
  try {
    return normalizeSize(JSON.parse(localStorage.getItem(STORAGE_KEY)));
  } catch {
    return null;
  }
}

let saveTimer = null;

function writeGlobalSize(size) {
  const s = normalizeSize(size);
  if (!s) return;

  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    } catch {}
  }, 250);
}

function clampToMinSize(node, size) {
  const s = normalizeSize(size);
  if (!s) return null;

  try {
    if (typeof node.computeSize === "function") {
      const min = normalizeSize(node.computeSize());
      if (min) {
        return [
          Math.max(s[0], min[0]),
          Math.max(s[1], min[1]),
        ];
      }
    }
  } catch {}

  return s;
}

function persistCurrentSize(node, updateGlobal = false) {
  if (applyingSavedSize) return;

  const s = normalizeSize(node.size);
  if (!s) return;

  node.properties = node.properties || {};
  const prev = normalizeSize(node.properties[PROP_KEY]);

  node.properties[PROP_KEY] = s;

  if (updateGlobal) {
    if (!prev || prev[0] !== s[0] || prev[1] !== s[1]) {
      writeGlobalSize(s);
    }
  }
}

function applySize(node, size) {
  const s = clampToMinSize(node, size);
  if (!s) return;

  applyingSavedSize = true;

  try {
    if (typeof node.setSize === "function") {
      node.setSize(s);
    } else {
      node.size = s;
    }
  } finally {
    applyingSavedSize = false;
  }

  const real = normalizeSize(node.size) || s;

  node.properties = node.properties || {};
  node.properties[PROP_KEY] = real;

  if (typeof node.setDirtyCanvas === "function") {
    node.setDirtyCanvas(true, true);
  }
}

function restoreSize(node, info, allowGlobal) {
  info = info || node?.__pprConfigureInfo;

  let saved =
    normalizeSize(info?.properties?.[PROP_KEY]) ||
    normalizeSize(node?.properties?.[PROP_KEY]);

  if (!saved && info) {
    saved = normalizeSize(info.size);
  }

  if (!saved && allowGlobal) {
    saved = readGlobalSize();
  }

  if (!saved) return;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      applySize(node, saved);
    });
  });
}

function isTargetNode(node) {
  if (!node) return false;

  return (
    node.type === NODE_TYPE ||
    node.constructor?.title === NODE_TYPE ||
    node.constructor?.nodeData?.name === NODE_TYPE ||
    node.constructor?.type === NODE_TYPE
  );
}

function patchNode(node) {
  if (!node || node.__pprSizeMemoryPatched) return;

  node.__pprSizeMemoryPatched = true;
  node.properties = node.properties || {};

  const oldSetSize = node.setSize;
  node.setSize = function (...args) {
    const ret = oldSetSize ? oldSetSize.apply(this, args) : undefined;
    persistCurrentSize(this, false);
    return ret;
  };

  const oldOnResize = node.onResize;
  node.onResize = function (...args) {
    persistCurrentSize(this, true);
    return oldOnResize ? oldOnResize.apply(this, args) : undefined;
  };

  const oldOnMouseUp = node.onMouseUp;
  node.onMouseUp = function (...args) {
    persistCurrentSize(this, true);
    return oldOnMouseUp ? oldOnMouseUp.apply(this, args) : undefined;
  };

  const oldSerialize = node.serialize;
  node.serialize = function (...args) {
    persistCurrentSize(this, false);

    const data = oldSerialize ? oldSerialize.apply(this, args) : {};
    data.properties = this.properties;

    return data;
  };
}

function getGraphNodes() {
  const g = app.graph;
  if (!g) return [];

  if (Array.isArray(g.nodes)) return g.nodes;
  if (Array.isArray(g._nodes)) return g._nodes;

  try {
    return [...g.nodes];
  } catch {
    return [];
  }
}

function forEachTargetNode(fn) {
  try {
    const nodes = getGraphNodes();
    for (const node of nodes) {
      if (isTargetNode(node)) {
        fn(node);
      }
    }
  } catch {}
}

app.registerExtension({
  name: "PainterPromptRewriter.SizeMemory",

  async beforeConfigureGraph() {
    configuringGraph = true;
  },

  async afterConfigureGraph() {
    configuringGraph = false;

    forEachTargetNode((node) => {
      patchNode(node);
      restoreSize(node, node.__pprConfigureInfo || node, false);
    });
  },

  beforeGraphSerialize() {
    forEachTargetNode((node) => {
      patchNode(node);
      persistCurrentSize(node, false);
    });
  },

  nodeCreated(node) {
    if (!isTargetNode(node)) return;

    patchNode(node);

    const oldConfigure = node.configure;

    node.configure = function (...args) {
      const info = args?.[0] || {};
      this.__pprConfigureInfo = info;

      const ret = oldConfigure ? oldConfigure.apply(this, args) : undefined;

      restoreSize(this, info, false);

      return ret;
    };

    if (!configuringGraph) {
      restoreSize(node, null, true);
    }
  },

  loadedGraphNode(node) {
    if (!isTargetNode(node)) return;

    patchNode(node);
    restoreSize(node, node.__pprConfigureInfo || node, false);
  },
});