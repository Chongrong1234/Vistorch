#!/usr/bin/env node
/**
 * Static-analyzer checks + static/runtime node-id compatibility.
 *
 * 1. analyze examples/ with root class Net (pure stdlib python, no torch)
 * 2. trace demo_cnn.py to completion (torch python)
 * 3. assert every runtime node id was predicted by the static analyzer
 *    (so breakpoints placed on the static graph always hit)
 *
 * Usage: node test/static_check.js [torch-python]
 */
const { spawnSync, spawn } = require("child_process");
const net = require("net");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const TORCH_PY = process.argv[2] || "python3";
const PYPATH = path.join(ROOT, "extension", "python");

let failures = 0;
const check = (c, label, extra) => {
  if (c) console.log(`  ok   ${label}`);
  else { failures++; console.log(`  FAIL ${label}${extra ? " — " + extra : ""}`); }
};

// ---------------------------------------------------------------- phase 1
console.log("--- phase 1: static analysis of examples/");
const res = spawnSync("python3", ["-m", "vistorch.static",
  path.join(ROOT, "examples"), "--root-class", "Net"],
  { env: Object.assign({}, process.env, { PYTHONPATH: PYPATH }), maxBuffer: 1 << 24 });
check(res.status === 0, "analyzer exits 0", res.stderr && res.stderr.toString().slice(0, 300));
const data = JSON.parse(res.stdout.toString());
const ids = new Set(data.nodes.map((n) => n.id));

check(data.rootClass === "Net", "root class Net");
check(data.entries.some((e) => e.className === "Net" && e.script.endsWith("demo_cnn.py")),
  "entry point detected in demo_cnn.py");
["Net", "Net.stem", "Net.stem.conv", "Net.body", "Net.body.0", "Net.body.2.act",
  "Net.pool", "Net.dropout", "Net.fc"].forEach((id) =>
  check(ids.has(id), `predicted node ${id}`));
const ek = new Set(data.edges.map((e) => e.from + "→" + e.to));
check(ek.has("Net.stem→Net.body.0"), "flow edge stem→body.0");
check(ek.has("Net.stem.conv→Net.stem.bn"), "execution order conv→bn (not reversed)");
check(ek.has("Net.pool→Net.dropout") && ek.has("Net.dropout→Net.fc"), "tail chain");
const stem = data.nodes.find((n) => n.id === "Net.stem");
check(stem && /Block/.test(stem.source) && stem.file.endsWith("demo_cnn.py"),
  "source snippet + file for custom class");
check(data.nodes.every((n) => n.predicted === true), "all static nodes flagged predicted");

// ---------------------------------------------------------------- phase 2
console.log("--- phase 2: trace demo_cnn and compare runtime ids");
const server = net.createServer();
server.listen(0, "127.0.0.1", () => {
  const port = server.address().port;
  const child = spawn(TORCH_PY, ["-m", "vistorch", path.join(ROOT, "examples", "demo_cnn.py")], {
    env: Object.assign({}, process.env, {
      VISTORCH_PORT: String(port), PYTHONPATH: PYPATH,
    }),
    stdio: ["ignore", "ignore", "inherit"],
  });
  server.on("connection", (sock) => {
    const runtimeIds = new Set();
    let buf = "";
    sock.on("data", (d) => {
      buf += d.toString();
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        let m; try { m = JSON.parse(line); } catch { continue; }
        if (m.type === "graph") m.nodes.forEach((n) => runtimeIds.add(n.id));
        if (m.type === "paused")
          sock.write(JSON.stringify({ cmd: "resume", mode: "continue" }) + "\n");
        if (m.type === "finished") {
          const missing = [...runtimeIds].filter((id) => !ids.has(id));
          check(missing.length === 0,
            `all ${runtimeIds.size} runtime node ids predicted statically`,
            missing.join(", "));
          check(runtimeIds.size === ids.size,
            `static count == runtime count (${ids.size} vs ${runtimeIds.size})`);
          console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL CHECKS PASSED");
          child.kill("SIGKILL");
          server.close();
          process.exit(failures ? 1 : 0);
        }
      }
    });
  });
});
