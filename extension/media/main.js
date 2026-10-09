/* Vistorch webview: neural-net flow graph + debugger controls. */
(function () {
  "use strict";
  const vscode = acquireVsCodeApi();

  // ------------------------------------------------------------------ state
  const NODE_W = 210, NODE_H = 78, X_GAP = 280, Y_GAP = 116;
  const MAX_CALLS_KEPT = 4000;

  const S = {
    hello: null,
    nodes: new Map(),        // id -> node meta
    nodeSeqs: new Map(),     // id -> [seq,...] ascending
    edges: [],               // [{from,to}]
    edgeSet: new Set(),
    events: new Map(),       // seq -> call event (ring)
    minSeq: 0, maxSeq: -1,
    paused: null,            // paused message
    finished: null,
    error: null,
    selected: null,
    breakpoints: new Set(),
    viewSeq: null,           // null = follow live
    follow: true,
    filter: "",
    transform: { x: 60, y: 40, k: 1 },
    layout: new Map(),       // id -> {x,y,layer,order}
    log: [],
  };

  const $ = (sel) => document.querySelector(sel);
  const svg = $("#canvas");
  const wrap = $("#canvas-wrap");
  const SVGNS = "http://www.w3.org/2000/svg";
  let renderQueued = false;
  let layoutDirty = true;

  // ------------------------------------------------------------------ utils
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function fmtShape(t) {
    if (!t || t.kind !== "tensor") return t ? (t.repr || t.kind) : "?";
    return "[" + t.shape.join(",") + "]";
  }
  function fmtNum(v) {
    if (v == null) return "-";
    if (Math.abs(v) >= 1000) return v.toExponential(2);
    return String(Math.round(v * 1e4) / 1e4);
  }
  function firstTensor(summary) {
    if (!summary) return null;
    if (summary.kind === "tensor") return summary;
    if (summary.items && summary.items.length) {
      for (const it of summary.items) {
        const t = firstTensor(it.value !== undefined ? it.value : it);
        if (t) return t;
      }
    }
    return null;
  }
  function shapeSig(summary) {
    const t = firstTensor(summary);
    return t ? t.shape.join("x") + t.dtype : "";
  }
  function ioText(summary, maxLen) {
    if (!summary) return "-";
    if (summary.kind === "tensor") return fmtShape(summary);
    if (summary.items) {
      const parts = summary.items.map((i) => ioText(i.value !== undefined ? i.value : i, 1e9));
      let s = parts.join(", ");
      if (summary.truncated) s += ",…";
      if (s.length > maxLen) s = s.slice(0, maxLen - 1) + "…";
      return s;
    }
    return summary.repr || summary.kind;
  }

  function eventForNodeAt(nodeId, seqLimit) {
    const seqs = S.nodeSeqs.get(nodeId);
    if (!seqs || !seqs.length) return null;
    const limit = seqLimit == null ? Infinity : seqLimit;
    let lo = 0, hi = seqs.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (seqs[mid] <= limit) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans >= 0 ? S.events.get(seqs[ans]) || null : null;
  }
  function prevEventForNode(nodeId, seq) {
    const seqs = S.nodeSeqs.get(nodeId);
    if (!seqs) return null;
    for (let i = seqs.length - 1; i >= 0; i--) {
      if (seqs[i] < seq) return S.events.get(seqs[i]) || null;
    }
    return null;
  }

  // ------------------------------------------------------------------ layout
  function computeLayout() {
    const ids = [...S.nodes.keys()];
    const indeg = new Map(), out = new Map(), inc = new Map();
    ids.forEach((id) => { indeg.set(id, 0); out.set(id, []); inc.set(id, []); });
    S.edges.forEach((e) => {
      if (!S.nodes.has(e.from) || !S.nodes.has(e.to)) return;
      out.get(e.from).push(e.to);
      inc.get(e.to).push(e.from);
      indeg.set(e.to, indeg.get(e.to) + 1);
    });

    // longest-path layering (Kahn; cycles fall back gracefully)
    const layer = new Map();
    const queue = ids.filter((id) => indeg.get(id) === 0);
    const done = new Set(queue);
    queue.forEach((id) => layer.set(id, 0));
    let guard = ids.length * 4 + 10;
    const q = [...queue];
    while (q.length && guard-- > 0) {
      const u = q.shift();
      for (const v of out.get(u)) {
        const nl = Math.max(layer.get(v) || 0, layer.get(u) + 1);
        if (nl !== layer.get(v)) { layer.set(v, nl); q.push(v); }
        if (!done.has(v)) {
          done.add(v);
          layer.set(v, nl);
          q.push(v);
        }
      }
    }
    ids.forEach((id) => { if (!layer.has(id)) layer.set(id, 0); });

    // group by layer, initial order by first seq
    const firstSeq = new Map();
    ids.forEach((id) => {
      const seqs = S.nodeSeqs.get(id);
      firstSeq.set(id, seqs && seqs.length ? seqs[0] : 1e9);
    });
    const layers = new Map();
    ids.forEach((id) => {
      const l = layer.get(id);
      if (!layers.has(l)) layers.set(l, []);
      layers.get(l).push(id);
    });
    layers.forEach((arr) => arr.sort((a, b) => firstSeq.get(a) - firstSeq.get(b)));

    // barycenter sweeps to reduce crossings
    const pos = new Map();
    const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
    for (let sweep = 0; sweep < 6; sweep++) {
      sortedLayers.forEach((l) => layers.get(l).forEach((id, i) => pos.set(id, i)));
      const pass = (neighbors, layerList) => {
        layerList.forEach((l) => {
          const arr = layers.get(l);
          arr.sort((a, b) => bary(neighbors.get(a), pos) - bary(neighbors.get(b), pos));
          arr.forEach((id, i) => pos.set(id, i));
        });
      };
      pass(inc, sortedLayers.slice(1));
      pass(out, sortedLayers.slice(0, -1).reverse());
    }

    S.layout.clear();
    sortedLayers.forEach((l) => {
      const arr = layers.get(l);
      arr.forEach((id, i) => {
        S.layout.set(id, {
          x: 40 + l * X_GAP,
          y: 40 + i * Y_GAP,
          layer: l, order: i,
        });
      });
    });
    layoutDirty = false;
  }
  function bary(list, pos) {
    if (!list || !list.length) return 1e9;
    let s = 0, n = 0;
    list.forEach((id) => { if (pos.has(id)) { s += pos.get(id); n++; } });
    return n ? s / n : 1e9;
  }

  // ------------------------------------------------------------------ render
  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => { renderQueued = false; render(); });
  }

  function el(name, attrs, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, cls, text) {
    const t = el("text", { x, y, class: cls }, parent);
    t.textContent = text;
    return t;
  }

  function render() {
    if (layoutDirty) computeLayout();
    renderToolbar();
    renderScrub();
    renderSidebar();
    renderGraph();
  }

  function renderGraph() {
    svg.innerHTML = "";
    const defs = el("defs", {}, svg);
    const filter = el("filter", { id: "glow", x: "-40%", y: "-40%", width: "180%", height: "180%" }, defs);
    el("feDropShadow", { dx: 0, dy: 0, stdDeviation: 4, "flood-color": "#dcdcaa", "flood-opacity": 0.8 }, filter);
    const marker = el("marker", {
      id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse",
    }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#888" }, marker);

    const vp = el("g", {
      transform: `translate(${S.transform.x},${S.transform.y}) scale(${S.transform.k})`,
    }, svg);

    const activeId = S.paused && S.follow ? S.paused.nodeId : null;

    // edges
    S.edges.forEach((e) => {
      const a = S.layout.get(e.from), b = S.layout.get(e.to);
      if (!a || !b) return;
      const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2;
      const x2 = b.x, y2 = b.y + NODE_H / 2;
      const dx = Math.max(40, (x2 - x1) / 2);
      const hot = activeId && (e.from === activeId || e.to === activeId) ||
        S.selected && (e.from === S.selected || e.to === S.selected);
      el("path", {
        d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`,
        class: "edge" + (hot ? " hot" : ""),
        "marker-end": "url(#arrow)",
      }, vp);
    });

    // nodes
    const filt = S.filter.toLowerCase();
    S.nodes.forEach((node, id) => {
      const p = S.layout.get(id);
      if (!p) return;
      const ev = eventForNodeAt(id, S.viewSeq == null ? undefined : S.viewSeq);
      const g = el("g", { class: "node", "data-id": id, transform: `translate(${p.x},${p.y})` }, vp);
      let cls = "node";
      if (!node.builtin) cls += " custom";
      if (id === S.selected) cls += " selected";
      if (id === activeId) cls += " active";
      if (filt) cls += id.toLowerCase().includes(filt) ? " match" : " dimmed";
      g.setAttribute("class", cls);

      el("rect", { class: "body", width: NODE_W, height: NODE_H, rx: 6 }, g);
      txt(g, 10, 17, "title", trim(node.name, 26));
      txt(g, 10, 31, "cls", node.className + (node.extra ? "  " + trim(node.extra, 22) : ""));
      if (ev) {
        const prev = prevEventForNode(id, ev.seq);
        const inStr = trim("← " + ev.inputs.map((i) => ioText(i, 26)).join(", "), 34);
        const outStr = trim("→ " + ioText(ev.outputs, 34), 34);
        const changed = prev && shapeSig(prev.outputs) !== shapeSig(ev.outputs);
        txt(g, 10, 46, "io", inStr);
        txt(g, 10, 59, "io" + (changed ? " changed" : ""), outStr);
        txt(g, 10, 72, "meta", `#${ev.seq}  ${ev.timeMs}ms` + (changed ? "  ⚠ shape changed" : ""));
      } else {
        txt(g, 10, 46, "io", "(not executed yet)");
        if (node.numParams) txt(g, 10, 59, "meta", `params ${fmtNum(node.numParams)}`);
      }
      if (S.breakpoints.has(id)) {
        el("circle", { class: "bp", cx: NODE_W - 12, cy: 12, r: 5 }, g);
      }
      g.addEventListener("click", (e) => { e.stopPropagation(); select(id); });
      g.addEventListener("dblclick", (e) => { e.stopPropagation(); toggleBreakpoint(id); });
    });
  }

  function trim(s, n) {
    s = String(s || "");
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  // ------------------------------------------------------------------ toolbar
  function renderToolbar() {
    const bar = $("#toolbar");
    const paused = !!S.paused && !S.finished;
    bar.innerHTML = "";
    const mk = (label, title, mode, primary) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.title = title;
      if (primary) b.className = "primary";
      b.disabled = !paused;
      b.onclick = () => {
        vscode.postMessage({ type: "resume", mode });
        S.paused = null;
        setStatus("running…");
        renderToolbar();
      };
      bar.appendChild(b);
      return b;
    };
    mk("▶ Continue", "run until next breakpoint", "continue", true);
    mk("↓ Step", "pause at the very next module call", "step");
    mk("⤼ Step Over", "next call at same or shallower depth", "step_over");
    mk("↑ Step Out", "next call at shallower depth", "step_out");
    bar.appendChild(sep());
    const restart = document.createElement("button");
    restart.textContent = "⟳ Restart";
    restart.onclick = () => vscode.postMessage({ type: "restart" });
    bar.appendChild(restart);
    const stop = document.createElement("button");
    stop.textContent = "■ Stop";
    stop.onclick = () => vscode.postMessage({ type: "stop" });
    bar.appendChild(stop);
    bar.appendChild(sep());
    const fit = document.createElement("button");
    fit.textContent = "⛶ Fit";
    fit.onclick = fitView;
    bar.appendChild(fit);

    const status = document.createElement("div");
    status.id = "status";
    bar.appendChild(status);
    const search = document.createElement("input");
    search.id = "search";
    search.placeholder = "filter nodes…";
    search.value = S.filter;
    search.oninput = () => { S.filter = search.value; scheduleRender(); };
    bar.appendChild(search);

    if (S.paused) {
      setStatus(`⏸ paused at ${S.paused.nodeId}  (#${S.paused.seq}, ${S.paused.reason})`, true);
    } else if (S.finished) {
      setStatus(`✔ finished — ${S.finished.totalCalls} module calls`);
    } else if (S.error) {
      setStatus("✖ script error (see sidebar)");
    } else if (S.hello) {
      setStatus(`running… torch ${S.hello.torch}`);
    }
  }
  function sep() {
    const d = document.createElement("div");
    d.className = "sep";
    return d;
  }
  function setStatus(text, pausedCls) {
    const elx = $("#status");
    if (!elx) return;
    elx.innerHTML = pausedCls ? `<span class="paused">${esc(text)}</span>` : esc(text);
  }

  // ------------------------------------------------------------------ scrub
  function renderScrub() {
    const bar = $("#scrubbar");
    bar.innerHTML = "";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = "scrub";
    slider.min = S.minSeq;
    slider.max = Math.max(S.maxSeq, 0);
    slider.value = S.viewSeq == null ? slider.max : S.viewSeq;
    slider.oninput = () => {
      S.viewSeq = Number(slider.value);
      S.follow = false;
      scheduleRender();
    };
    bar.appendChild(slider);
    const label = document.createElement("div");
    label.id = "scrublabel";
    label.textContent = S.viewSeq == null
      ? `LIVE (#${S.maxSeq})`
      : `history #${S.viewSeq} / ${S.maxSeq}`;
    bar.appendChild(label);
    const live = document.createElement("button");
    live.textContent = "⇥ LIVE";
    live.title = "jump back to the latest event";
    live.onclick = () => { S.viewSeq = null; S.follow = true; scheduleRender(); };
    bar.appendChild(live);
  }

  // ------------------------------------------------------------------ sidebar
  function select(id) {
    S.selected = id;
    scheduleRender();
  }
  function toggleBreakpoint(id) {
    if (S.breakpoints.has(id)) S.breakpoints.delete(id);
    else S.breakpoints.add(id);
    vscode.postMessage({ type: "setBreakpoints", breakpoints: [...S.breakpoints] });
    scheduleRender();
  }

  function renderSidebar() {
    const side = $("#sidebar");
    side.innerHTML = "";
    if (S.error) {
      const d = document.createElement("div");
      d.className = "section";
      d.innerHTML = `<h3 style="color:#f48771">Error</h3><pre>${esc(S.error.traceback || S.error.message)}</pre>`;
      side.appendChild(d);
    }
    if (!S.selected || !S.nodes.has(S.selected)) {
      const d = document.createElement("div");
      d.className = "empty";
      d.innerHTML = "点击图中的节点查看张量详情<br>双击节点设置/取消断点<br>滚轮缩放，拖拽平移";
      side.appendChild(d);
    } else {
      const node = S.nodes.get(S.selected);
      const ev = eventForNodeAt(S.selected, S.viewSeq == null ? undefined : S.viewSeq);
      side.appendChild(nodeSection(node, ev));
      if (ev) side.appendChild(ioSection(ev));
      if (node.source) side.appendChild(sourceSection(node));
    }
    if (S.log.length) side.appendChild(logSection());
  }

  function nodeSection(node, ev) {
    const d = document.createElement("div");
    d.className = "section";
    const bpOn = S.breakpoints.has(node.id);
    d.innerHTML = `
      <h3>${esc(node.id)}</h3>
      <div class="kv"><b>class</b><span>${esc(node.className)}</span></div>
      ${node.extra ? `<div class="kv"><b>config</b><span>${esc(node.extra)}</span></div>` : ""}
      <div class="kv"><b>params</b><span>${fmtNum(node.numParams)} (${fmtNum(node.trainableParams)} trainable)</span></div>
      ${ev ? `<div class="kv"><b>last call</b><span>#${ev.seq} · ${ev.timeMs} ms · depth ${ev.depth}</span></div>` : ""}
      <div class="kv"><b>defined</b><span class="dim">${esc(shortPath(node.file))}:${node.line || 1}</span></div>
      <div class="kv"><b>called</b><span class="dim">${esc(shortPath(node.callFile))}:${node.callLine || 1}</span></div>
      <div style="margin-top:6px">
        <button data-act="def">→ 跳到定义</button>
        <button data-act="call">→ 跳到调用处</button>
        <button data-act="bp">${bpOn ? "✖ 取消断点" : "● 设置断点"}</button>
      </div>`;
    d.querySelector('[data-act="def"]').onclick = () =>
      vscode.postMessage({ type: "jump", file: node.file, line: node.line });
    d.querySelector('[data-act="call"]').onclick = () =>
      vscode.postMessage({ type: "jump", file: node.callFile, line: node.callLine });
    d.querySelector('[data-act="bp"]').onclick = () => toggleBreakpoint(node.id);
    return d;
  }

  function shortPath(p) {
    if (!p) return "?";
    const parts = p.split(/[\\/]/);
    return parts.length > 2 ? "…/" + parts.slice(-2).join("/") : p;
  }

  function tensorTable(t, caption) {
    return `<table>
      <tr><th colspan="2">${esc(caption)} ${esc(fmtShape(t))} ${esc(t.dtype)}</th></tr>
      <tr><td>device</td><td>${esc(t.device)}</td></tr>
      <tr><td>requires_grad</td><td>${t.requiresGrad}</td></tr>
      <tr><td>mean ± std</td><td>${fmtNum(t.mean)} ± ${fmtNum(t.std)}</td></tr>
      <tr><td>min / max</td><td>${fmtNum(t.min)} / ${fmtNum(t.max)}</td></tr>
      ${t.norm != null ? `<tr><td>L2 norm</td><td>${fmtNum(t.norm)}</td></tr>` : ""}
    </table>`;
  }

  function summaryHtml(s, caption) {
    if (!s) return "";
    if (s.kind === "tensor") return tensorTable(s, caption);
    if (s.items) {
      return s.items.map((it, i) =>
        summaryHtml(it.value !== undefined ? it.value : it,
          `${caption}[${it.key !== undefined ? esc(it.key) : i}]`)
      ).join("");
    }
    return `<div class="kv"><b>${esc(caption)}</b><span>${esc(s.repr || s.kind)}</span></div>`;
  }

  function ioSection(ev) {
    const d = document.createElement("div");
    d.className = "section";
    let html = `<h3>调用 #${ev.seq} 输入 / 输出</h3>`;
    ev.inputs.forEach((inp, i) => {
      html += summaryHtml(inp, inp.name || `in${i}`);
    });
    html += summaryHtml(ev.outputs, "out");
    d.innerHTML = html;
    return d;
  }

  function sourceSection(node) {
    const d = document.createElement("div");
    d.className = "section";
    d.innerHTML = `<h3>源码（${esc(node.className)}）</h3><pre>${esc(node.source)}</pre>`;
    return d;
  }

  function logSection() {
    const d = document.createElement("div");
    d.className = "section";
    d.innerHTML = `<h3>脚本输出</h3><pre>${esc(S.log.slice(-30).join(""))}</pre>`;
    return d;
  }

  // ------------------------------------------------------------------ pan/zoom
  let drag = null;
  svg.addEventListener("mousedown", (e) => {
    drag = { x: e.clientX, y: e.clientY, tx: S.transform.x, ty: S.transform.y };
    svg.classList.add("panning");
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    S.transform.x = drag.tx + (e.clientX - drag.x);
    S.transform.y = drag.ty + (e.clientY - drag.y);
    scheduleRender();
  });
  window.addEventListener("mouseup", () => { drag = null; svg.classList.remove("panning"); });
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const k0 = S.transform.k;
    const k1 = Math.min(3, Math.max(0.1, k0 * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
    const wx = (mx - S.transform.x) / k0, wy = (my - S.transform.y) / k0;
    S.transform.k = k1;
    S.transform.x = mx - wx * k1;
    S.transform.y = my - wy * k1;
    scheduleRender();
  }, { passive: false });
  svg.addEventListener("click", () => { S.selected = null; scheduleRender(); });

  function fitView() {
    if (!S.layout.size) return;
    let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    S.layout.forEach((p) => {
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + NODE_W); maxY = Math.max(maxY, p.y + NODE_H);
    });
    const r = wrap.getBoundingClientRect();
    const k = Math.min(1.5, Math.min(r.width / (maxX - minX + 80), r.height / (maxY - minY + 80)));
    S.transform.k = Math.max(0.1, k);
    S.transform.x = (r.width - (maxX - minX) * S.transform.k) / 2 - minX * S.transform.k;
    S.transform.y = (r.height - (maxY - minY) * S.transform.k) / 2 - minY * S.transform.k;
    scheduleRender();
  }

  function centerOn(nodeId) {
    const p = S.layout.get(nodeId);
    if (!p) return;
    const r = wrap.getBoundingClientRect();
    S.transform.x = r.width / 2 - (p.x + NODE_W / 2) * S.transform.k;
    S.transform.y = r.height / 2 - (p.y + NODE_H / 2) * S.transform.k;
  }

  // ------------------------------------------------------------------ banner
  let bannerEl = null;
  function banner(text, isErr) {
    if (bannerEl) bannerEl.remove();
    bannerEl = document.createElement("div");
    bannerEl.className = "banner" + (isErr ? " error" : "");
    bannerEl.textContent = text;
    wrap.appendChild(bannerEl);
    if (!isErr) setTimeout(() => { if (bannerEl) { bannerEl.remove(); bannerEl = null; } }, 4000);
  }

  // ------------------------------------------------------------------ messages
  window.addEventListener("message", (e) => {
    const m = e.data;
    switch (m.type) {
      case "hello":
        S.hello = m;
        $("#hint").textContent =
          `${m.script}  ·  torch ${m.torch}${m.cuda ? " · CUDA" : " · CPU"} — 双击节点打断点，然后 Continue`;
        break;
      case "graph":
        let added = false;
        m.nodes.forEach((n) => { if (!S.nodes.has(n.id)) { S.nodes.set(n.id, n); added = true; } });
        m.edges.forEach((ed) => {
          const k = ed.from + "→" + ed.to;
          if (!S.edgeSet.has(k)) { S.edgeSet.add(k); S.edges.push(ed); added = true; }
        });
        if (added) layoutDirty = true;
        break;
      case "call": {
        S.maxSeq = Math.max(S.maxSeq, m.seq);
        S.events.set(m.seq, m);
        if (!S.nodeSeqs.has(m.nodeId)) S.nodeSeqs.set(m.nodeId, []);
        S.nodeSeqs.get(m.nodeId).push(m.seq);
        if (S.events.size > MAX_CALLS_KEPT) {
          const drop = S.events.keys().next().value;
          S.events.delete(drop);
          S.minSeq = drop + 1;
        }
        break;
      }
      case "paused":
        S.paused = m;
        S.viewSeq = m.seq;
        S.follow = true;
        S.selected = m.nodeId;
        if (m.reason === "breakpoint") banner(`● 断点命中: ${m.nodeId}`);
        if (layoutDirty) computeLayout();
        centerOn(m.nodeId);
        break;
      case "stdout":
        S.log.push(m.text);
        if (S.log.length > 200) S.log.shift();
        break;
      case "error":
        S.error = m;
        banner("脚本异常：" + (m.message || ""), true);
        break;
      case "finished":
        S.finished = m;
        S.paused = null;
        banner(m.stopped ? "已停止" : `运行完成：${m.totalCalls} 次模块调用`);
        break;
      case "sessionEnd":
        S.paused = null;
        if (!S.finished) banner(`进程退出 (code ${m.code})`, m.code !== 0);
        break;
    }
    scheduleRender();
  });

  // ------------------------------------------------------------------ boot
  $("#hint").textContent = "等待 tracer 连接…";
  vscode.postMessage({ type: "ready" });
})();
