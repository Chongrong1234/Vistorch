# Vistorch — PyTorch 张量流可视化调试器（VSCode 插件）

左边写代码，右边实时画出神经网络的**数据流动图**：每一层什么时候被调用、
输入/输出张量的 shape、dtype、mean/std/min/max 一目了然；像调试器一样
**单步（Step）/ 步过（Step Over）/ 步出（Step Out）/ 断点（Breakpoint）**
地走完整个前向传播，点击任意节点即可**跳转到对应的代码实现**。

## 功能

- 🔀 **实时网络流图**：运行脚本时动态构建，边来自真实的张量流向
  （按输出张量身份追踪生产者 → 消费者，残差连接也能正确连线）
- 🐞 **调试器式步进**：Continue / Step / Step Over / Step Out / Restart / Stop，
  双击图中节点设置断点，Continue 会跑到断点处暂停
- 📐 **张量详情**：shape、dtype、device、requires_grad、mean±std、min/max、L2 范数；
  同一层多次调用 shape 发生变化时高亮提示（⚠ shape changed）
- 🔗 **代码跳转**：每个节点记录「类定义位置」和「调用位置」，
  内置层（nn.Conv2d 等）跳到你的调用处，自定义层跳到 forward 实现；
  侧栏还直接内嵌显示该模块的源码
- 🕘 **历史回放**：底部滑块可以拖回任意一次模块调用，查看当时的张量状态
- 🧩 **零侵入**：不需要改你的代码，插件用补丁方式 Hook `nn.Module.__call__`

## 架构

```
┌────────────────────────── VSCode ──────────────────────────┐
│  extension.js (宿主)          media/ (Webview 流图+调试控制) │
│   · 启动 TCP server            · SVG 分层布局 / 缩放平移      │
│   · spawn python -m vistorch   · 步进按钮 / 断点 / 历史滑块   │
│   · 转发消息 ◄── JSONL/TCP ──► · 点击节点 → 跳转源码          │
└──────────────▲─────────────────────────────────────────────┘
               │ 127.0.0.1:<ephemeral port>
┌──────────────┴─────────────────────────────────────────────┐
│  python/vistorch (随插件内置, 经 PYTHONPATH 注入, 无需安装)   │
│   · 补丁 torch.nn.Module.__call__                           │
│   · 记录输入/输出张量摘要 + 源码位置 (inspect)                │
│   · 状态机: continue / step / step_over / step_out + 断点集   │
│   · 暂停时阻塞用户线程, 等待扩展端 resume 命令                │
└─────────────────────────────────────────────────────────────┘
```

## 安装与使用

### 开发模式（推荐现在就用）

1. 克隆本仓库：`git clone https://github.com/Chongrong1234/Vistorch.git`
2. 用 VSCode 打开仓库根目录
3. 按 `F5`（已内置 `.vscode/launch.json`）启动 Extension Development Host
4. 在新窗口里打开你的训练脚本（比如 `examples/demo_cnn.py`）
5. 命令面板执行 **`Vistorch: Trace Current Python File`**
   （或编辑器右上角的运行按钮 / 右键菜单）
6. 右侧弹出流图：先按 `▶ Continue` 跑到断点，或 `↓ Step` 逐层走

### 打包安装

```bash
npm i -g @vscode/vsce
cd extension && vsce package        # 生成 vistorch-0.1.0.vsix
code --install-extension vistorch-0.1.0.vsix
```

### 配置项（Settings）

| 配置 | 默认 | 说明 |
|---|---|---|
| `vistorch.pythonPath` | `""` | 带 PyTorch 的解释器；留空则自动探测（`python.defaultInterpreterPath` → Python 插件 → PATH） |
| `vistorch.scriptArgs` | `""` | 传给被追踪脚本的额外参数 |
| `vistorch.stats` | `true` | 是否统计 mean/std/min/max（超大模型可关掉提速） |
| `vistorch.maxCalls` | `0` | 超过 N 次模块调用自动停止（0 不限） |
| `vistorch.pauseOnEntry` | `true` | 第一次模块调用即暂停，方便从头单步 |

### 命令行单独使用 tracer

```bash
PYTHONPATH=extension/python VISTORCH_PORT=8977 \
  python -m vistorch examples/demo_cnn.py
# 另起一个 TCP server 监听 8977 即可收到 JSONL 事件流（协议见下）
```

## 协议（JSON Lines over TCP）

python → 扩展：`hello` / `graph`（节点与边的增量）/ `call`（一次模块调用的
io 摘要）/ `paused`（阻塞等待恢复）/ `stdout` / `error` / `finished`

扩展 → python：`{cmd:"resume", mode:"continue|step|step_over|step_out"}` /
`{cmd:"set_breakpoints", breakpoints:[nodeId...]}` / `{cmd:"stop"}`

详见 `extension/python/vistorch/tracer.py` 模块 docstring。

## 开发自测

```bash
# 协议冒烟测试（步进/断点/源码定位/张量摘要，需一个有 torch 的 python）
node test/mock_client.js /path/to/python examples/demo_cnn.py

# 跑通真实模型 GS-MTRNet（约 140 个节点、130 条边）
node test/run_to_end.js /path/to/python examples/demo_gsmtrnet.py
```

## 已知限制 / 路线图

- 边按「张量对象身份」推断：纯函数式算子（`torch.add`、`F.relu`）不产生节点，
  但其两端模块间的连线仍然正确；计划中：可选 Hook `torch.nn.functional` 常用算子
- `torch.compile` / JIT trace 的模型暂不生效（绕过 `__call__`）
- DataLoader worker 子进程自动跳过（不会重复上报）
- 计划中：反向传播梯度范数列、张量直方图、导出 trace 为文件分享

## 目录结构

```
extension/            VSCode 插件（零依赖 plain JS，免构建）
  package.json        插件清单（命令/配置项/菜单）
  extension.js        宿主：TCP server、spawn tracer、消息转发、代码跳转
  media/              Webview：main.js(SVG流图+调试控制) style.css icon.svg
  python/vistorch/    内置 tracer 包（__main__ 入口、协议、Hook、源码定位）
examples/             demo_cnn.py（教学小网络） demo_gsmtrnet.py（真实研究模型）
test/                 mock_client.js（协议冒烟） run_to_end.js（端到端）
```

## License

MIT
