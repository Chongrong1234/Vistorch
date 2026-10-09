#!/usr/bin/env node
/**
 * Generic driver: runs a script under the tracer straight to completion
 * (auto-continues every pause), then prints graph statistics.
 *
 * Usage: node test/run_to_end.js <python> <script> [extraEnvJson]
 */
const net = require("net");
const { spawn } = require("child_process");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PYTHON = process.argv[2] || "python3";
const SCRIPT = process.argv[3];
const EXTRA_ENV = process.argv[4] ? JSON.parse(process.argv[4]) : {};
if (!SCRIPT) { console.error("usage: node test/run_to_end.js <python> <script>"); process.exit(2); }

const server = net.createServer();
server.listen(0, "127.0.0.1", () => {
  const port = server.address().port;
  const env = Object.assign({}, process.env, EXTRA_ENV, {
    VISTORCH_PORT: String(port),
    PYTHONPATH: path.join(ROOT, "extension", "python"),
  });
  const child = spawn(PYTHON, ["-m", "vistorch", SCRIPT], {
    env, cwd: path.dirname(SCRIPT), stdio: ["ignore", "inherit", "inherit"],
  });
  child.on("exit", (code) => console.log(`tracer exited code=${code}`));

  server.on("connection", (sock) => {
    const nodes = new Set();
    const edges = new Set();
    let calls = 0, pauses = 0, stdout = "";
    let buf = "";
    sock.on("data", (d) => {
      buf += d.toString();
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        let m; try { m = JSON.parse(line); } catch { continue; }
        if (m.type === "graph") {
          m.nodes.forEach((n) => nodes.add(n.id));
          m.edges.forEach((e) => edges.add(e.from + "→" + e.to));
        } else if (m.type === "call") {
          calls++;
        } else if (m.type === "paused") {
          pauses++;
          sock.write(JSON.stringify({ cmd: "resume", mode: "continue" }) + "\n");
        } else if (m.type === "stdout") {
          stdout += m.text;
        } else if (m.type === "error") {
          console.error("SCRIPT ERROR:\n" + m.traceback);
        } else if (m.type === "finished") {
          console.log(`finished: calls=${calls} nodes=${nodes.size} edges=${edges.size} pauses=${pauses} exit=${m.exitCode}`);
          console.log("script stdout:", stdout.trim());
          child.kill("SIGKILL");
          server.close();
          process.exit(m.exitCode === 0 && nodes.size > 5 && edges.size > 5 ? 0 : 1);
        }
      }
    });
  });
});
