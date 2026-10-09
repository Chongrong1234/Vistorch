"""Vistorch — live tensor-flow tracer for PyTorch.

Vistorch patches ``torch.nn.Module.__call__`` so every module invocation in
your script is captured (input/output tensor shapes, dtypes, statistics),
reported over a JSON-lines TCP protocol, and can be *paused* like a debugger
(step / step over / step out / continue / breakpoints).

Normally you do not import this package directly; the VSCode extension runs
your script as::

    PYTHONPATH=<extension>/python VISTORCH_PORT=<port> python -m vistorch your_script.py

Manual usage is identical.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
