"""Static project analyzer: predict the module tree WITHOUT running any code.

Parses every ``.py`` file under a root directory with ``ast``, finds
``nn.Module`` subclasses, reads their ``__init__`` (``self.x = SomeLayer(...)``
and ``nn.Sequential(...)`` containers) and their ``forward`` call order, then
expands a predicted instance graph whose node ids use the SAME naming scheme
as the runtime tracer (``RootClass.child.attr``), so breakpoints placed on the
static graph map directly onto runtime nodes.

Pure stdlib — torch is NOT required for analysis.

CLI:
    python -m vistorch.static ROOT_DIR [--root-class NAME] [--exclude DIR ...]

Prints one JSON object: {"nodes": [...], "edges": [...],
                         "entries": [...], "rootClass": "...", "classes": N}
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

_MAX_SOURCE_LINES = 80
_DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "venv", ".venv", "env", ".env", "dist", "build",
    "site-packages", "archive", "legacy",
}


# --------------------------------------------------------------------------- #
# model classes
# --------------------------------------------------------------------------- #
class Child:
    __slots__ = ("attr", "cls", "line", "container", "items")

    def __init__(self, attr, cls, line, container=False, items=None):
        self.attr = attr          # attribute name on the parent
        self.cls = cls            # constructor class name ("" for containers)
        self.line = line
        self.container = container       # nn.Sequential / ModuleList
        self.items = items or []         # container: list of (cls, line)


class ModuleClass:
    __slots__ = ("name", "file", "line", "children", "forward_calls", "source")

    def __init__(self, name, file, line):
        self.name = name
        self.file = file
        self.line = line
        self.children = []         # Child, in __init__ assignment order
        self.forward_calls = []    # attr names, in forward source order
        self.source = ""


def _ctor_name(call: ast.Call):
    """Resolve a constructor call to a short class name, or None."""
    f = call.func
    if isinstance(f, ast.Attribute):
        parts = []
        node = f
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        # nn.Conv2d / torch.nn.Conv2d -> Conv2d ; layers.Conv2d -> Conv2d
        return parts[-1] if parts else None
    if isinstance(f, ast.Name):
        return f.id
    return None


_CONTAINER_NAMES = {"Sequential", "ModuleList"}
_TORCH_MODULE_SUFFIXES = ("Module",)


class _ModuleVisitor(ast.NodeVisitor):
    """First pass: collect candidate module classes per file."""

    def __init__(self, file, known_names):
        self.file = file
        self.known = known_names   # names already known to be modules
        self.found = []

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Attribute):
                bases.append(b.attr)
            elif isinstance(b, ast.Name):
                bases.append(b.id)
        is_module = any(
            base in self.known or base.endswith(_TORCH_MODULE_SUFFIXES)
            for base in bases
        )
        if not is_module:
            return
        mc = ModuleClass(node.name, self.file, node.lineno)
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                self._parse_init(stmt, mc)
            elif isinstance(stmt, ast.FunctionDef) and stmt.name == "forward":
                self._parse_forward(stmt, mc)
        self.found.append(mc)

    # -- __init__: self.attr = Layer(...) / nn.Sequential(...) --------------
    def _parse_init(self, fn: ast.FunctionDef, mc: ModuleClass):
        seen = set()
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            tgt = stmt.targets[0]
            if not (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                continue
            if not isinstance(stmt.value, ast.Call):
                continue
            attr = tgt.attr
            if attr in seen:
                continue
            seen.add(attr)
            cls = _ctor_name(stmt.value) or ""
            if cls in _CONTAINER_NAMES:
                items = []
                for arg in stmt.value.args:
                    if isinstance(arg, ast.Call):
                        items.append((_ctor_name(arg) or "?", arg.lineno))
                mc.children.append(Child(attr, cls, stmt.lineno,
                                         container=True, items=items))
            elif cls:
                mc.children.append(Child(attr, cls, stmt.lineno))

    # -- forward: self.attr(...) call order ----------------------------------
    def _parse_forward(self, fn: ast.FunctionDef, mc: ModuleClass):
        # Post-order traversal of Call nodes == actual evaluation order:
        # self.act(self.bn(self.conv(x)))  ->  conv, bn, act
        calls = []

        class V(ast.NodeVisitor):
            def visit_Call(self, node):
                self.generic_visit(node)  # arguments evaluate first
                f = node.func
                if (isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "self"):
                    calls.append(f.attr)

        V().visit(fn)
        mc.forward_calls = calls


def _read_source(file, line, end_line):
    try:
        with open(file, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    end = min(end_line or (line + _MAX_SOURCE_LINES), line + _MAX_SOURCE_LINES)
    return "".join(lines[line - 1:end])


# --------------------------------------------------------------------------- #
# project scan
# --------------------------------------------------------------------------- #
def scan_project(root, excludes=None):
    excludes = set(_DEFAULT_EXCLUDES) | set(excludes or [])
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excludes
                       and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                py_files.append(os.path.join(dirpath, fn))
    py_files.sort()

    # pass 1: class names that look like modules (nn.Module base heuristic)
    trees = {}
    known = set()
    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                trees[f] = ast.parse(fh.read(), filename=f)
        except (SyntaxError, ValueError):
            continue
    for f, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    base = b.attr if isinstance(b, ast.Attribute) else \
                        b.id if isinstance(b, ast.Name) else ""
                    if base.endswith(_TORCH_MODULE_SUFFIXES):
                        known.add(node.name)

    # pass 2: full parse with known names (catches project-internal bases)
    classes = {}
    for f, tree in trees.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                v = _ModuleVisitor(f, known)
                v.visit_ClassDef(node)
                for mc in v.found:
                    mc.source = _read_source(f, mc.line,
                                             getattr(node, "end_lineno", None))
                    classes.setdefault(mc.name, mc)

    # entry points: if __name__ == "__main__" blocks instantiating a class
    entries = []
    for f, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            seg = node.test
            is_main = False
            if isinstance(seg, ast.Compare) and isinstance(seg.left, ast.Name) \
                    and seg.left.id == "__name__":
                is_main = True
            if not is_main:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                    name = _ctor_name(sub.value)
                    if name in classes:
                        entries.append({"script": f, "className": name,
                                        "line": sub.lineno})
    return classes, entries


# --------------------------------------------------------------------------- #
# graph expansion (instance tree with runtime-compatible ids)
# --------------------------------------------------------------------------- #
_BUILTIN_HINT = True  # unknown ctor names are treated as torch builtins


def expand(root_class, classes):
    nodes = {}
    edges = set()

    def add_node(nid, name, cls_name, file, line, source, builtin):
        if nid in nodes:
            return nodes[nid]
        nodes[nid] = {
            "id": nid,
            "name": name,
            "className": cls_name,
            "builtin": builtin,
            "predicted": True,
            "extra": "",
            "file": file,
            "line": line,
            "callFile": None,
            "callLine": None,
            "source": source,
            "numParams": 0,
            "trainableParams": 0,
        }
        return nodes[nid]

    def expand_class(prefix, cls_name, stack):
        """Expand a project class at id=prefix. Returns (firstLeaf, lastLeaf)
        connector endpoints for chaining, or None when unknown."""
        mc = classes.get(cls_name)
        if mc is None or cls_name in stack:
            return None
        stack = stack | {cls_name}
        child_ids = {}
        for ch in mc.children:
            cid = f"{prefix}.{ch.attr}"
            if ch.container:
                add_node(cid, ch.attr, ch.cls, mc.file, ch.line, "", True)
                prev_leaf = None
                first = last = None
                for i, (sub_cls, sub_line) in enumerate(ch.items):
                    sid = f"{cid}.{i}"
                    sub_mc = classes.get(sub_cls)
                    add_node(sid, str(i), sub_cls,
                             sub_mc.file if sub_mc else mc.file,
                             sub_mc.line if sub_mc else sub_line,
                             sub_mc.source if sub_mc else "",
                             sub_mc is None)
                    if sub_mc is not None:
                        expand_class(sid, sub_cls, stack)
                    edges.add((cid, sid))          # container -> each item
                    if prev_leaf is not None:
                        edges.add((prev_leaf, sid))
                    if first is None:
                        first = sid
                    prev_leaf = sid
                    last = sid
                child_ids[ch.attr] = (cid, first, last)
            else:
                sub_mc = classes.get(ch.cls)
                add_node(cid, ch.attr, ch.cls,
                         sub_mc.file if sub_mc else mc.file,
                         sub_mc.line if sub_mc else ch.line,
                         sub_mc.source if sub_mc else "",
                         sub_mc is None)
                if sub_mc is not None:
                    expand_class(cid, ch.cls, stack)
                child_ids[ch.attr] = (cid, cid, cid)

        # forward-call order -> flow edges between children
        prev = None
        for attr in mc.forward_calls:
            info = child_ids.get(attr)
            if info is None:
                continue
            cid, first, last = info
            if prev is not None:
                edges.add((prev, first))           # previous out -> this in
            prev = last
        return child_ids

    add_node(root_class, root_class, root_class,
             classes[root_class].file, classes[root_class].line,
             classes[root_class].source, False)
    expand_class(root_class, root_class, frozenset())
    return list(nodes.values()), [{"from": a, "to": b} for a, b in sorted(edges)]


def pick_root(classes, entries):
    if entries:
        return entries[0]["className"]
    if not classes:
        return None
    return max(classes.values(), key=lambda mc: len(mc.children)).name


def analyze(root, root_class=None, excludes=None):
    classes, entries = scan_project(root, excludes)
    if root_class is None:
        root_class = pick_root(classes, entries)
    if root_class is None or root_class not in classes:
        return {"nodes": [], "edges": [], "entries": entries,
                "rootClass": root_class, "classes": len(classes),
                "error": f"root class not found: {root_class!r}; "
                         f"candidates: {sorted(classes)[:20]}"}
    nodes, edges = expand(root_class, classes)
    return {"nodes": nodes, "edges": edges, "entries": entries,
            "rootClass": root_class, "classes": len(classes),
            "rootDir": os.path.abspath(root)}


def main():
    ap = argparse.ArgumentParser(prog="vistorch.static")
    ap.add_argument("root")
    ap.add_argument("--root-class", default=None)
    ap.add_argument("--exclude", action="append", default=[])
    ns = ap.parse_args()
    result = analyze(ns.root, ns.root_class, ns.exclude)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
