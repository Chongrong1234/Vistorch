"""CLI entry:  python -m vistorch [--port N] [--no-entry-pause] script.py [args...]

Connection settings may also come from the environment:
VISTORCH_HOST (default 127.0.0.1), VISTORCH_PORT (default 8977),
VISTORCH_STATS=0 disables tensor statistics, VISTORCH_MAX_CALLS caps events.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
import traceback

import torch

from . import __version__
from .tracer import TraceStopped, Tracer


def main() -> int:
    parser = argparse.ArgumentParser(prog="vistorch")
    parser.add_argument("--host", default=os.environ.get("VISTORCH_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VISTORCH_PORT", "8977"))
    )
    parser.add_argument("--no-entry-pause", action="store_true")
    parser.add_argument("script", help="python script to run under the tracer")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="arguments passed to the script")
    ns = parser.parse_args()

    script = os.path.abspath(ns.script)
    if not os.path.isfile(script):
        print(f"vistorch: script not found: {script}", file=sys.stderr)
        return 2

    try:
        tracer = Tracer.connect(ns.host, ns.port,
                                pause_on_entry=not ns.no_entry_pause)
    except OSError as exc:
        print(f"vistorch: cannot connect to {ns.host}:{ns.port}: {exc}",
              file=sys.stderr)
        return 2

    tracer.install()
    tracer.emit({
        "type": "hello",
        "pid": tracer.pid,
        "script": script,
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "vistorch": __version__,
        "cuda": torch.cuda.is_available(),
        "time": time.time(),
    })

    sys.argv = [script] + list(ns.args)
    script_dir = os.path.dirname(script)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    exit_code = 0
    stopped = False
    try:
        runpy.run_path(script, run_name="__main__")
    except TraceStopped:
        stopped = True
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 0
    except KeyboardInterrupt:
        stopped = True
    except BaseException:
        tracer.emit({"type": "error", "message": "script raised an exception",
                     "traceback": traceback.format_exc()})
        exit_code = 1
    finally:
        tracer.emit({"type": "finished", "totalCalls": tracer.total_calls,
                     "stopped": stopped, "exitCode": exit_code})
        tracer.uninstall()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
