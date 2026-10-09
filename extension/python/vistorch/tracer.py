"""Core live tracer: patches ``nn.Module.__call__`` and speaks JSONL/TCP.

Protocol (one JSON object per line):

python -> extension
    {"type":"hello",   "pid", "script", "torch", "vistorch"}
    {"type":"graph",   "nodes":[...], "edges":[{"from","to"}]}   # deltas
    {"type":"call",    "seq", "nodeId", "depth", "inputs", "outputs", "timeMs"}
    {"type":"paused",  "seq", "nodeId", "depth", "reason"}       # blocks!
    {"type":"stdout",  "stream", "text"}
    {"type":"error",   "message", "traceback"}
    {"type":"finished","totalCalls", "stopped", "exitCode"}

extension -> python
    {"cmd":"resume", "mode":"continue"|"step"|"step_over"|"step_out"}
    {"cmd":"set_breakpoints", "breakpoints":[nodeId, ...]}
    {"cmd":"stop"}
"""
from __future__ import annotations

import itertools
import json
import os
import queue
import socket
import sys
import threading
import time

import torch
import torch.nn as nn

from . import __version__, sourcemap, tensors

_ORIG_CALL = nn.Module.__call__
_TRACER = None  # global singleton


class TraceStopped(Exception):
    """Raised inside user code when the user presses Stop."""


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
class _Connection:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._wlock = threading.Lock()
        self.dead = False

    def send(self, obj: dict) -> bool:
        if self.dead:
            return False
        try:
            data = (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            with self._wlock:
                self.sock.sendall(data)
            return True
        except OSError:
            self.dead = True
            return False

    def reader_loop(self, on_command):
        buf = b""
        while not self.dead:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    on_command(json.loads(line.decode("utf-8")))
                except Exception:
                    pass
        self.dead = True
        on_command({"cmd": "stop"})  # extension vanished -> unblock user code


# --------------------------------------------------------------------------- #
# stdout tee
# --------------------------------------------------------------------------- #
class _Tee:
    def __init__(self, wrapped, tracer, stream):
        self._wrapped = wrapped
        self._tracer = tracer
        self._stream = stream

    def write(self, text):
        if text and text.strip():
            self._tracer.emit({"type": "stdout", "stream": self._stream, "text": text})
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# --------------------------------------------------------------------------- #
# module registry / graph
# --------------------------------------------------------------------------- #
class _Registry:
    """Assigns stable node ids and builds the flow-graph incrementally."""

    def __init__(self, tracer):
        self.tracer = tracer
        self.by_module = {}          # id(module) -> node dict
        self.names = set()
        self.edges = set()           # (producer_node, consumer_node)
        self.producer = {}           # id(tensor) -> node_id  (bounded)
        self._anon_counter = itertools.count(1)

    # -- nodes ---------------------------------------------------------------
    def node_for(self, module: nn.Module, parent_frame) -> str:
        key = id(module)
        node = self.by_module.get(key)
        if node is not None:
            return node["id"]

        cls = type(module)
        class_name = cls.__name__
        if parent_frame is not None:
            attr = self._find_attr(parent_frame.module, module)
            if attr:
                name = f"{parent_frame.node_id}.{attr}"
            else:
                name = f"{parent_frame.node_id}.{class_name.lower()}#{next(self._anon_counter)}"
        else:
            name = class_name
        name = self._dedupe(name)

        file, line, source = sourcemap.class_location(cls)
        call = sourcemap.find_call_site()
        builtin = sourcemap.is_torch_builtin(cls)
        n_params = 0
        n_train = 0
        try:
            for p in module.parameters(recurse=False):
                n_params += p.numel()
                n_train += int(p.requires_grad) * p.numel()
        except Exception:
            pass
        try:
            extra = module.extra_repr()
        except Exception:
            extra = ""

        node = {
            "id": name,
            "name": name.split(".")[-1],
            "className": class_name,
            "builtin": builtin,
            "extra": extra[:200],
            "file": file,
            "line": line,
            "callFile": call[0] if call else None,
            "callLine": call[1] if call else None,
            "source": source,
            "numParams": n_params,
            "trainableParams": n_train,
        }
        self.by_module[key] = node
        self.tracer.emit({"type": "graph", "nodes": [node], "edges": []})
        return name

    @staticmethod
    def _find_attr(parent: nn.Module, child: nn.Module):
        try:
            for n, c in parent.named_children():
                if c is child:
                    return n
        except Exception:
            pass
        return None

    def _dedupe(self, name: str) -> str:
        if name not in self.names:
            self.names.add(name)
            return name
        i = 2
        while f"{name}#{i}" in self.names:
            i += 1
        name = f"{name}#{i}"
        self.names.add(name)
        return name

    # -- edges (tensor-identity data flow) ------------------------------------
    def edges_from_inputs(self, args, kwargs, consumer: str):
        new_edges = []
        seen = set()
        for value in itertools.chain(args, kwargs.values()):
            for t in tensors.iter_tensors(value):
                prod = self.producer.get(id(t))
                if prod and prod != consumer and (prod, consumer) not in self.edges:
                    self.edges.add((prod, consumer))
                    if (prod, consumer) not in seen:
                        seen.add((prod, consumer))
                        new_edges.append({"from": prod, "to": consumer})
        if new_edges:
            self.tracer.emit({"type": "graph", "nodes": [], "edges": new_edges})

    def register_outputs(self, out, node_id: str):
        if len(self.producer) > 200_000:  # bound memory
            self.producer.clear()
        for t in tensors.iter_tensors(out):
            self.producer[id(t)] = node_id


class _Frame:
    __slots__ = ("module", "node_id")

    def __init__(self, module, node_id):
        self.module = module
        self.node_id = node_id


# --------------------------------------------------------------------------- #
# tracer
# --------------------------------------------------------------------------- #
class Tracer:
    def __init__(self, conn: _Connection, pause_on_entry: bool = True):
        self.conn = conn
        self.pid = os.getpid()
        self.registry = _Registry(self)
        self.resume_q: queue.Queue = queue.Queue()
        self.breakpoints = set()
        self._bp_lock = threading.Lock()
        self.mode = "continue"
        self.pause_depth = 0
        self.pause_on_entry = pause_on_entry
        self._entered = False
        self._stop = False
        self._local = threading.local()
        self._seq = itertools.count(0)
        self.total_calls = 0
        self.max_calls = int(os.environ.get("VISTORCH_MAX_CALLS", "0") or "0")

    # -- lifecycle -----------------------------------------------------------
    @classmethod
    def connect(cls, host: str, port: int, pause_on_entry: bool = True) -> "Tracer":
        tracer = cls(_Connection(host, port), pause_on_entry)
        thread = threading.Thread(
            target=tracer.conn.reader_loop, args=(tracer._on_command,), daemon=True
        )
        thread.start()
        return tracer

    def install(self):
        global _TRACER
        _TRACER = self
        nn.Module.__call__ = _wrapped_call
        sys.stdout = _Tee(sys.stdout, self, "stdout")
        sys.stderr = _Tee(sys.stderr, self, "stderr")

    def uninstall(self):
        global _TRACER
        nn.Module.__call__ = _ORIG_CALL
        _TRACER = None

    # -- io ------------------------------------------------------------------
    def emit(self, obj: dict):
        self.conn.send(obj)

    def _on_command(self, cmd: dict):
        c = cmd.get("cmd")
        if c == "resume":
            self.resume_q.put(cmd.get("mode", "continue"))
        elif c == "set_breakpoints":
            with self._bp_lock:
                self.breakpoints = set(cmd.get("breakpoints", []))
        elif c == "stop":
            self._stop = True
            self.resume_q.put(None)

    # -- hook ----------------------------------------------------------------
    def should_trace(self) -> bool:
        return (
            not self.conn.dead
            and not self._stop
            and os.getpid() == self.pid  # dataloader workers etc.
        )

    def call(self, module: nn.Module, args, kwargs):
        local = self._local
        if getattr(local, "in_hook", False):
            return _ORIG_CALL(module, *args, **kwargs)
        local.in_hook = True
        try:
            stack = getattr(local, "stack", None)
            if stack is None:
                stack = local.stack = []
            depth = len(stack)
            parent = stack[-1] if stack else None
            node_id = self.registry.node_for(module, parent)
            self.registry.edges_from_inputs(args, kwargs, node_id)
            inputs = tensors.summarize_call(args, kwargs)

            stack.append(_Frame(module, node_id))
            local.in_hook = False
            t0 = time.perf_counter()
            try:
                out = _ORIG_CALL(module, *args, **kwargs)
            finally:
                local.in_hook = True
                stack.pop()
            elapsed = (time.perf_counter() - t0) * 1000.0

            outputs = tensors.summarize(out)
            self.registry.register_outputs(out, node_id)

            seq = next(self._seq)
            self.total_calls = seq + 1
            self.emit(
                {
                    "type": "call",
                    "seq": seq,
                    "nodeId": node_id,
                    "depth": depth,
                    "inputs": inputs,
                    "outputs": outputs,
                    "timeMs": round(elapsed, 3),
                }
            )
            self._maybe_pause(node_id, seq, depth)
            if self.max_calls and seq + 1 >= self.max_calls:
                raise TraceStopped(f"max calls reached ({self.max_calls})")
            return out
        finally:
            local.in_hook = False

    # -- debugger state machine ----------------------------------------------
    def _maybe_pause(self, node_id: str, seq: int, depth: int):
        reason = None
        if self.pause_on_entry and not self._entered:
            reason = "entry"
        self._entered = True

        mode = self.mode
        if mode == "step":
            reason = reason or "step"
        elif mode == "step_over" and depth <= self.pause_depth:
            reason = reason or "step"
        elif mode == "step_out" and depth < self.pause_depth:
            reason = reason or "step"
        if reason is None:
            with self._bp_lock:
                if node_id in self.breakpoints:
                    reason = "breakpoint"
        if reason is None:
            return

        self.emit({"type": "paused", "seq": seq, "nodeId": node_id,
                   "depth": depth, "reason": reason})
        # block until the extension tells us how to resume
        while True:
            got = self.resume_q.get()
            if got is None or self._stop or self.conn.dead:
                raise TraceStopped("trace stopped by user")
            self.mode = got
            self.pause_depth = depth
            return


def _wrapped_call(module: nn.Module, *args, **kwargs):
    tracer = _TRACER
    if tracer is None or not tracer.should_trace():
        return _ORIG_CALL(module, *args, **kwargs)
    return tracer.call(module, args, kwargs)
