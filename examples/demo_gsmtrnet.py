"""Trace a real research model: GS-MTRNet (multi-spectral soybean traits).

Assumes the GS-MTRNet repo sits next to this repo (../../GS-MTRNet relative
to this file) or set GSMTRNET_ROOT. Random-init resnet18, CPU, 64x64 inputs.

Run under the tracer:
    PYTHONPATH=extension/python VISTORCH_PORT=8977 python -m vistorch examples/demo_gsmtrnet.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "GSMTRNET_ROOT",
    Path(__file__).resolve().parents[2].parent / "GS-MTRNet",
))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gsmtrnet import NUM_TRAITS  # noqa: E402
from gsmtrnet.models import build_model  # noqa: E402


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_model("GSMTRNet", num_targets=NUM_TRAITS, embed_dim=32,
                        rgb_encoder="resnet18", rgb_weights="none")
    B = 2
    rgb = torch.randn(B, 3, 64, 64)
    ndvi = torch.randn(B, 1, 64, 64)
    mask = (torch.rand(B, 1, 64, 64) > 0.3).float()
    days = torch.tensor([33.0, 92.0])
    stage = torch.tensor([0, 4])
    out = model(rgb, ndvi, mask, days, stage)
    loss = out.pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gsmtrnet] out={tuple(out.shape)} loss={float(loss):.4f} params={n_params}")
