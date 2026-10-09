"""Map nn.Module instances to source-code locations.

For every module we capture two locations:

* ``file`` / ``line`` — where the module *class* is defined (its ``forward``).
* ``callFile`` / ``callLine`` — where the module was *invoked* the first time,
  i.e. the first stack frame outside of torch / vistorch internals.  For
  built-in layers (``nn.Conv2d`` ...) the class lives inside torch, so the
  call site is usually what the user wants to jump to.
"""
from __future__ import annotations

import inspect
import os
import sys

import torch

_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))
_VISTORCH_DIR = os.path.dirname(os.path.abspath(__file__))
_MAX_SOURCE_LINES = 80


def _is_internal(filename: str) -> bool:
    fn = os.path.abspath(filename)
    return fn.startswith(_TORCH_DIR) or fn.startswith(_VISTORCH_DIR)


def is_torch_builtin(cls: type) -> bool:
    try:
        fn = os.path.abspath(inspect.getsourcefile(cls) or "")
    except TypeError:
        return False
    return fn.startswith(_TORCH_DIR)


def class_location(cls: type):
    """Return (file, line, source) for a module class, best effort."""
    try:
        file = inspect.getsourcefile(cls)
    except TypeError:
        file = None
    line = 1
    source = ""
    if file:
        try:
            lines, line = inspect.getsourcelines(cls)
            source = "".join(lines[:_MAX_SOURCE_LINES])
            if len(lines) > _MAX_SOURCE_LINES:
                source += f"\n# ... ({len(lines) - _MAX_SOURCE_LINES} more lines)"
        except (OSError, TypeError):
            pass
    return file, line, source


def find_call_site(skip: int = 0):
    """First frame outside torch/vistorch on the current stack, or None."""
    frame = sys._getframe(1)
    while frame is not None:
        fn = frame.f_code.co_filename
        if fn and fn[0] == "<":  # <stdin>, <string> ...
            frame = frame.f_back
            continue
        if not _is_internal(fn):
            return os.path.abspath(fn), frame.f_lineno
        frame = frame.f_back
    return None
