/**
 * Vistorch VSCode extension host (zero-dependency, CommonJS).
 *
 * Flow:
 *   1. "Vistorch: Trace Current Python File" opens the graph webview (Beside).
 *   2. When the webview signals `ready`, we start a TCP server on an ephemeral
 *      localhost port and spawn `<python> -m vistorch <script>` with
 *      PYTHONPATH pointing at the bundled tracer and VISTORCH_PORT set.
 *   3. Socket messages are forwarded to the webview; webview commands
 *      (resume / set_breakpoints / stop) are written back to the socket.
 *   4. `jump` messages open the source file at the requested line in column 1.
 */
const vscode = require("vscode");
const net = require("net");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

let output;
let session = null;
let statusItem;

function activate(context) {
  output = vscode.window.createOutputChannel("Vistorch");
  statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  context.subscriptions.push(
    output,
    statusItem,
    vscode.commands.registerCommand("vistorch.traceCurrentFile", (uri) =>
      startTrace(context, uri)
    ),
    vscode.commands.registerCommand("vistorch.analyzeProject", (uri) =>
      analyzeProject(context, uri)
    ),
    vscode.commands.registerCommand("vistorch.stopTrace", () =>
      session ? session.stop() : undefined
    ),
    vscode.commands.registerCommand("vistorch.restartTrace", () =>
      session ? session.restart() : undefined
    )
  );
}

function deactivate() {
  if (session) session.dispose();
}

async function startTrace(context, uri) {
  let file = uri && uri.fsPath;
  if (!file && vscode.window.activeTextEditor) {
    file = vscode.window.activeTextEditor.document.uri.fsPath;
  }
  if (!file || !file.endsWith(".py")) {
    const picked = await vscode.window.showOpenDialog({
      filters: { Python: ["py"] },
      canSelectMany: false,
      title: "Select a python script to trace",
    });
    if (!picked || !picked.length) return;
    file = picked[0].fsPath;
  }

  if (session) session.dispose();
  session = new TraceSession(context, file);
  session.start();
}

/**
 * Static mode: analyze a project root with the bundled AST analyzer (no code
 * executed), open the graph webview showing the PREDICTED network, let the
 * user place breakpoints, then Run an entry script under the live tracer.
 */
async function analyzeProject(context, uri) {
  let folder = uri && uri.fsPath;
  if (!folder) {
    const ws = vscode.workspace.workspaceFolders;
    if (ws && ws.length === 1) {
      folder = ws[0].uri.fsPath;
    } else {
      const picked = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectMany: false,
        title: "选择要分析的工程根目录",
      });
      if (!picked || !picked.length) return;
      folder = picked[0].fsPath;
    }
  }

  const data = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification,
      title: "Vistorch: 正在静态分析工程…" },
    async () => runStaticAnalyzer(context, await resolvePython(), folder)
  );
  if (!data) return;
  if (data.error) {
    vscode.window.showWarningMessage(`Vistorch 静态分析: ${data.error}`);
  }
  output.appendLine(
    `[vistorch] static: root=${folder} classes=${data.classes} ` +
    `nodes=${data.nodes.length} edges=${data.edges.length} root=${data.rootClass}`
  );

  if (session) session.dispose();
  session = new TraceSession(context, null, data);
  session.start();
}

function runStaticAnalyzer(context, python, folder) {
  return new Promise((resolve) => {
    const bundled = path.join(context.extensionPath, "python");
    const env = Object.assign({}, process.env, {
      PYTHONPATH: process.env.PYTHONPATH
        ? bundled + path.delimiter + process.env.PYTHONPATH
        : bundled,
    });
    const child = spawn(python, ["-m", "vistorch.static", folder], { env });
    let out = "", err = "";
    child.stdout.on("data", (d) => (out += d.toString()));
    child.stderr.on("data", (d) => (err += d.toString()));
    child.on("error", (e) => {
      vscode.window.showErrorMessage(
        `Vistorch: 无法启动 python "${python}": ${e.message}`);
      resolve(null);
    });
    child.on("exit", (code) => {
      if (code !== 0) {
        output.appendLine(`[vistorch] static analyzer failed: ${err}`);
        vscode.window.showErrorMessage("Vistorch 静态分析失败，详见输出面板");
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(out.trim().split("\n").pop()));
      } catch {
        output.appendLine(`[vistorch] bad analyzer output: ${out.slice(0, 500)}`);
        resolve(null);
      }
    });
  });
}

class TraceSession {
  constructor(context, script, staticData) {
    this.context = context;
    this.script = script || null;
    this.staticData = staticData || null;
    this.server = null;
    this.socket = null;
    this.child = null;
    this.panel = null;
    this.buf = "";
    this.webviewReady = false;
    this.pending = [];
    this.disposed = false;
  }

  start() {
    this.createPanel();
  }

  // ------------------------------------------------------------------ webview
  createPanel() {
    const extPath = this.context.extensionPath;
    const media = vscode.Uri.file(path.join(extPath, "media"));
    this.panel = vscode.window.createWebviewPanel(
      "vistorchGraph",
      this.script
        ? `Vistorch: ${path.basename(this.script)}`
        : `Vistorch: ${this.staticData.rootClass} (静态图)`,
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [media],
      }
    );
    this.panel.iconPath = vscode.Uri.file(
      path.join(extPath, "media", "icon.svg")
    );
    const js = this.panel.webview.asWebviewUri(
      vscode.Uri.file(path.join(extPath, "media", "main.js"))
    );
    const css = this.panel.webview.asWebviewUri(
      vscode.Uri.file(path.join(extPath, "media", "style.css"))
    );
    const nonce = String(Math.random()).slice(2);
    this.panel.webview.html = buildHtml(this.panel.webview, js, css, nonce);

    this.panel.webview.onDidReceiveMessage((msg) => this.onWebviewMessage(msg));
    this.panel.onDidDispose(() => this.dispose());
  }

  onWebviewMessage(msg) {
    switch (msg.type) {
      case "ready":
        if (!this.webviewReady) {
          this.webviewReady = true;
          if (this.staticData) {
            this.post(Object.assign({ type: "staticGraph" }, this.staticData));
          }
          if (this.script) {
            this.launchTracer();
          }
        }
        break;
      case "run":
        this.pickEntryAndRun();
        break;
      case "resume":
        this.sendCmd({ cmd: "resume", mode: msg.mode });
        break;
      case "setBreakpoints":
        this.sendCmd({ cmd: "set_breakpoints", breakpoints: msg.breakpoints });
        break;
      case "stop":
        this.stop();
        break;
      case "restart":
        this.restart();
        break;
      case "jump":
        this.jumpTo(msg.file, msg.line);
        break;
      case "log":
        output.appendLine(`[webview] ${msg.text}`);
        break;
    }
  }

  post(msg) {
    if (!this.panel) return;
    if (!this.webviewReady) {
      this.pending.push(msg);
      return;
    }
    this.panel.webview.postMessage(msg);
  }

  /** Static mode: let the user pick an entry script, then start the tracer. */
  async pickEntryAndRun() {
    if (this.child) return; // already running
    const entries = (this.staticData && this.staticData.entries) || [];
    let file = null;
    const items = entries.map((e) => ({
      label: `$(play) ${e.className}`,
      description: `${path.basename(e.script)}:${e.line}`,
      detail: e.script,
      script: e.script,
    }));
    items.push({ label: "$(file) 手动选择脚本…", script: null });
    const picked = await vscode.window.showQuickPick(items, {
      title: "选择入口脚本开始断点调试",
    });
    if (!picked) return;
    if (picked.script) {
      file = picked.script;
    } else {
      const dlg = await vscode.window.showOpenDialog({
        filters: { Python: ["py"] },
        canSelectMany: false,
        title: "选择入口脚本",
      });
      if (!dlg || !dlg.length) return;
      file = dlg[0].fsPath;
    }
    this.script = file;
    if (this.panel) this.panel.title = `Vistorch: ${path.basename(file)}`;
    this.launchTracer();
  }

  // ------------------------------------------------------------------ tracer
  async launchTracer() {
    const python = await resolvePython();
    const cfg = vscode.workspace.getConfiguration("vistorch");
    const extPath = this.context.extensionPath;
    const bundled = path.join(extPath, "python");

    this.server = net.createServer((sock) => this.onSocket(sock));
    await new Promise((res) => this.server.listen(0, "127.0.0.1", res));
    const port = this.server.address().port;

    const env = Object.assign({}, process.env, {
      VISTORCH_PORT: String(port),
      VISTORCH_STATS: cfg.get("stats") ? "1" : "0",
      VISTORCH_MAX_CALLS: String(cfg.get("maxCalls") || 0),
      PYTHONPATH: process.env.PYTHONPATH
        ? bundled + path.delimiter + process.env.PYTHONPATH
        : bundled,
      PYTHONUNBUFFERED: "1",
    });

    const args = ["-m", "vistorch"];
    if (!cfg.get("pauseOnEntry")) args.push("--no-entry-pause");
    args.push(this.script);
    const extra = (cfg.get("scriptArgs") || "").trim();
    if (extra) args.push(...extra.split(/\s+/));

    const folder = vscode.workspace.getWorkspaceFolder(
      vscode.Uri.file(this.script)
    );
    const cwd = folder ? folder.uri.fsPath : path.dirname(this.script);

    output.appendLine(`[vistorch] ${python} ${args.join(" ")}`);
    output.appendLine(`[vistorch] cwd=${cwd} port=${port}`);
    this.child = spawn(python, args, { env, cwd });
    this.child.stdout.on("data", (d) => output.append(d.toString()));
    this.child.stderr.on("data", (d) => output.append(d.toString()));
    this.child.on("error", (err) => {
      vscode.window.showErrorMessage(
        `Vistorch: failed to start python "${python}": ${err.message}`
      );
      this.post({ type: "sessionEnd", code: -1, error: err.message });
    });
    this.child.on("exit", (code) => {
      output.appendLine(`[vistorch] tracer exited code=${code}`);
      this.post({ type: "sessionEnd", code });
      setStatus("");
    });
  }

  onSocket(sock) {
    this.socket = sock;
    sock.setNoDelay(true);
    sock.on("data", (d) => {
      this.buf += d.toString();
      let i;
      while ((i = this.buf.indexOf("\n")) >= 0) {
        const line = this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        if (!line.trim()) continue;
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          continue;
        }
        this.onTracerMessage(msg);
      }
    });
    sock.on("close", () => {
      this.socket = null;
    });
    sock.on("error", () => {});
  }

  onTracerMessage(msg) {
    if (msg.type === "paused") {
      setStatus(`$(debug-pause) Vistorch: paused at ${msg.nodeId} (${msg.reason})`);
    } else if (msg.type === "finished") {
      setStatus(`$(check) Vistorch: finished (${msg.totalCalls} calls)`);
    } else if (msg.type === "error") {
      setStatus("$(error) Vistorch: script error");
      output.appendLine(msg.traceback || msg.message || "");
    } else if (msg.type === "hello") {
      output.appendLine(
        `[vistorch] connected: torch ${msg.torch}, script=${msg.script}`
      );
    }
    this.post(msg);
  }

  sendCmd(obj) {
    if (this.socket) {
      try {
        this.socket.write(JSON.stringify(obj) + "\n");
      } catch {}
    }
  }

  // ------------------------------------------------------------------ actions
  async jumpTo(file, line) {
    if (!file) return;
    if (!fs.existsSync(file)) {
      vscode.window.showWarningMessage(`Vistorch: file not found: ${file}`);
      return;
    }
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
    const ln = Math.max(0, (line || 1) - 1);
    const range = new vscode.Range(ln, 0, ln, 0);
    await vscode.window.showTextDocument(doc, {
      viewColumn: vscode.ViewColumn.One,
      selection: range,
    });
  }

  stop() {
    this.sendCmd({ cmd: "stop" });
    setTimeout(() => {
      if (this.child && !this.child.killed) this.child.kill();
    }, 1500);
  }

  restart() {
    const script = this.script;
    const staticData = this.staticData;
    this.dispose();
    session = new TraceSession(this.context, script, staticData);
    session.start();
  }

  dispose() {
    this.disposed = true;
    try { this.sendCmd({ cmd: "stop" }); } catch {}
    if (this.child && !this.child.killed) {
      try { this.child.kill(); } catch {}
    }
    if (this.server) { try { this.server.close(); } catch {} }
    if (this.panel) {
      const p = this.panel;
      this.panel = null;
      p.dispose();
    }
    setStatus("");
    if (session === this) session = null;
  }
}

async function resolvePython() {
  const cfg = vscode.workspace.getConfiguration("vistorch");
  const fromCfg = cfg.get("pythonPath");
  if (fromCfg) return fromCfg;
  const pyCfg = vscode.workspace.getConfiguration("python");
  const def = pyCfg.get("defaultInterpreterPath");
  if (def) return def;
  // best effort: ask the official Python extension
  try {
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (ext) {
      const api = ext.isActive ? ext.exports : await ext.activate();
      const exec = api?.settings?.getExecutionDetails?.()?.execCommand;
      if (exec && exec.length) return exec[0];
      const envPath = api?.environments?.getActiveEnvironmentPath?.()?.path;
      if (envPath) return envPath;
    }
  } catch {}
  return process.platform === "win32" ? "python" : "python3";
}

function setStatus(text) {
  if (!text) {
    statusItem.hide();
    return;
  }
  statusItem.text = text;
  statusItem.tooltip = "Vistorch trace session";
  statusItem.show();
}

function buildHtml(webview, js, css, nonce) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data:;">
<link rel="stylesheet" href="${css}">
<title>Vistorch</title>
</head>
<body>
  <div id="app">
    <div id="toolbar"></div>
    <div id="main">
      <div id="canvas-wrap"><svg id="canvas"></svg><div id="hint"></div></div>
      <div id="sidebar"></div>
    </div>
    <div id="scrubbar"></div>
  </div>
  <script nonce="${nonce}" src="${js}"></script>
</body>
</html>`;
}

module.exports = { activate, deactivate };
