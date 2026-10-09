#!/usr/bin/env node
/**
 * Protocol smoke-test for the Vistorch python tracer.
 *
 * Acts as a minimal "extension": opens a TCP server, spawns
 * `python -m vistorch examples/demo_cnn.py`, then drives the debugger:
 * entry pause -> several steps -> breakpoint on Net.fc -> continue -> stop.
 *
 * Usage:  node test/mock_client.js [python-binary] [script]
 */
const net = require("net");
const { spawn } = require("child_process");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PYTHON = process.argv[2] || "python3";
const SCRIPT = process.argv[3] || path.join(ROOT, "examples", "demo_cnn.py");

let failures = 0;
function check(cond, label, extra) {
  if (cond) console.log(`  ok   ${label}`);
  else {
    failures++;
    console.log(`  FAIL ${label}${extra ? " — " + extra : ""}`);
  }
}

const server = net.createServer();
server.listen(0, "127.0.0.1", () => {
  const port = server.address().port;
  console.log(`mock extension listening on ${port}`);
  const env = Object.assign({}, process.env, {
    VISTORCH_PORT: String(port),
    PYTHONPATH: path.join(ROOT, "extension", "python"),
  });
  const child = spawn(PYTHON, ["-m", "vistorch", SCRIPT], {
    env,
    cwd: ROOT,
    stdio: ["ignore", "inherit", "inherit"],
  });
  child.on("exit", (code) => console.log(`tracer exited code=${code}`));

  server.on("connection", (sock) => drive(sock, () => {
    child.kill("SIGKILL");
    server.close();
    setTimeout(() => process.exit(failures ? 1 : 0), 300);
  }));
});

function drive(sock, done) {
  let buf = "";
  const nodes = new Map();
  const edges = [];
  const calls = [];
  let stdoutSeen = false;
  const waiters = [];

  function send(obj) { sock.write(JSON.stringify(obj) + "\n"); }
  function waitFor(pred, label, timeoutMs = 15000) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("timeout waiting " + label)), timeoutMs);
      waiters.push({ pred, resolve: (m) => { clearTimeout(t); resolve(m); } });
    });
  }

  sock.on("data", (d) => {
    buf += d.toString();
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      let msg; try { msg = JSON.parse(line); } catch { continue; }
      if (msg.type === "graph") {
        msg.nodes.forEach((n) => nodes.set(n.id, n));
        edges.push(...msg.edges);
      } else if (msg.type === "call") {
        calls.push(msg);
      } else if (msg.type === "stdout") {
        stdoutSeen = true;
      }
      for (let k = waiters.length - 1; k >= 0; k--) {
        if (waiters[k].pred(msg)) { const w = waiters.splice(k, 1)[0]; w.resolve(msg); }
      }
    }
  });
  sock.on("close", () => console.log("socket closed"));

  (async () => {
    console.log("--- phase 1: handshake & entry pause");
    const hello = await waitFor((m) => m.type === "hello", "hello");
    check(!!hello.torch, "hello reports torch version", JSON.stringify(hello));
    const p0 = await waitFor((m) => m.type === "paused", "entry pause");
    check(p0.reason === "entry", "first pause reason is entry", p0.reason);
    check(nodes.size > 0, "graph nodes discovered before entry pause");
    const root = [...nodes.values()].find((n) => n.className === "Net");
    check(!!root, "root module Net discovered");
    check(root && root.callFile && root.callFile.endsWith("demo_cnn.py"),
      "call site recorded for Net", root && root.callFile);
    const conv = [...nodes.values()].find((n) => n.className === "Conv2d");
    check(conv && /torch/.test(conv.file || ""), "Conv2d maps into torch source");
    check(conv && typeof conv.source === "string" && conv.source.length > 0,
      "source snippet captured for Conv2d");

    console.log("--- phase 2: stepping");
    send({ cmd: "resume", mode: "step" });
    const p1 = await waitFor((m) => m.type === "paused" && m.reason === "step", "step pause 1");
    send({ cmd: "resume", mode: "step" });
    const p2 = await waitFor((m) => m.type === "paused" && m.seq > p1.seq, "step pause 2");
    check(p2.seq === p1.seq + 1, "step advances exactly one call", `${p1.seq}->${p2.seq}`);
    const ev = calls.find((c) => c.seq === p1.seq);
    check(ev && Array.isArray(ev.inputs) && ev.outputs, "call events carry io summaries");
    const t = ev && [...ev.inputs, ev.outputs].flatMap((x) => x.kind === "tensor" ? [x] : (x.items || []))
      .find((x) => x.kind === "tensor");
    check(t && Array.isArray(t.shape) && typeof t.mean === "number",
      "tensor summary has shape+stats", t && JSON.stringify(t && t.shape));

    console.log("--- phase 3: step_over / step_out sanity");
    send({ cmd: "resume", mode: "step_over" });
    const p3 = await waitFor((m) => m.type === "paused" && m.seq > p2.seq, "step_over pause");
    check(p3.depth <= p2.depth, "step_over stays at same/shallower depth",
      `${p2.depth}->${p3.depth}`);

    console.log("--- phase 4: breakpoint + continue");
    send({ cmd: "set_breakpoints", breakpoints: ["Net.fc"] });
    send({ cmd: "resume", mode: "continue" });
    const pb = await waitFor((m) => m.type === "paused" && m.reason === "breakpoint", "breakpoint pause", 20000);
    check(pb.nodeId === "Net.fc", "breakpoint hits Net.fc", pb.nodeId);
    const edgeToFc = edges.find((e) => e.to === "Net.fc");
    check(!!edgeToFc, "data-flow edge into Net.fc exists", JSON.stringify(edges.slice(-3)));
    check(edges.length > 5, "graph edges collected (" + edges.length + ")");

    console.log("--- phase 5: run to completion");
    send({ cmd: "set_breakpoints", breakpoints: [] });
    send({ cmd: "resume", mode: "continue" });
    const fin = await waitFor((m) => m.type === "finished", "finished", 30000);
    check(fin.stopped === false, "finished after natural completion");
    check(fin.exitCode === 0, "exit code 0", String(fin.exitCode));
    check(stdoutSeen, "script stdout was forwarded");

    console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL CHECKS PASSED");
    done();
  })().catch((e) => { console.error("driver error:", e.message); failures++; done(); });
}
