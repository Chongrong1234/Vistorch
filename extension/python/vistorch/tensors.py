"""Summarize tensors / arbitrary python values into JSON-safe descriptors."""
from __future__ import annotations

import os

import torch

_MAX_DEPTH = 3
_MAX_ITEMS = 16
_MAX_REPR = 120
_STATS_ENABLED = os.environ.get("VISTORCH_STATS", "1") != "0"


def _tensor_summary(t: torch.Tensor) -> dict:
    out = {
        "kind": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "requiresGrad": bool(t.requires_grad),
        "numel": t.numel(),
    }
    if _STATS_ENABLED and t.numel() > 0:
        try:
            with torch.no_grad():
                d = t.detach()
                if d.is_floating_point() or d.is_complex():
                    f = d.float()
                    out["mean"] = round(float(f.mean()), 6)
                    out["std"] = round(float(f.std()), 6)
                    out["min"] = round(float(f.min()), 6)
                    out["max"] = round(float(f.max()), 6)
                    out["norm"] = round(float(f.norm()), 6)
                else:
                    f = d.float()
                    out["mean"] = round(float(f.mean()), 6)
                    out["min"] = round(float(d.min()), 6)
                    out["max"] = round(float(d.max()), 6)
        except Exception:
            pass
    return out


def summarize(value, depth: int = 0):
    """Convert an arbitrary value into a JSON-safe summary."""
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if depth >= _MAX_DEPTH:
        return {"kind": type(value).__name__, "repr": _safe_repr(value)}
    if isinstance(value, (list, tuple)):
        items = [summarize(v, depth + 1) for v in value[:_MAX_ITEMS]]
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "len": len(value),
            "items": items,
            "truncated": len(value) > _MAX_ITEMS,
        }
    if isinstance(value, dict):
        items = [
            {"key": _safe_repr(k)[:40], "value": summarize(v, depth + 1)}
            for k, v in list(value.items())[:_MAX_ITEMS]
        ]
        return {
            "kind": "dict",
            "len": len(value),
            "items": items,
            "truncated": len(value) > _MAX_ITEMS,
        }
    if isinstance(value, (int, float, bool, str, type(None))):
        return {"kind": type(value).__name__, "repr": _safe_repr(value)}
    return {"kind": type(value).__name__, "repr": _safe_repr(value)}


def summarize_call(args, kwargs) -> list:
    """Summarize positional + keyword args of a module call."""
    parts = [summarize(a) for a in args]
    for k, v in kwargs.items():
        s = summarize(v)
        s["name"] = k
        parts.append(s)
    return parts


def _safe_repr(v) -> str:
    try:
        r = repr(v)
    except Exception:
        r = f"<{type(v).__name__}>"
    if len(r) > _MAX_REPR:
        r = r[: _MAX_REPR - 3] + "..."
    return r


def iter_tensors(value, depth: int = 0):
    """Yield every torch.Tensor nested inside *value* (depth-limited)."""
    if isinstance(value, torch.Tensor):
        yield value
    elif depth < _MAX_DEPTH:
        if isinstance(value, (list, tuple)):
            for v in value:
                yield from iter_tensors(v, depth + 1)
        elif isinstance(value, dict):
            for v in value.values():
                yield from iter_tensors(v, depth + 1)
