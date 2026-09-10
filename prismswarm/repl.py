"""The live control REPL: a plain terminal-embedded IPython shell (not a
notebook or kernel — no Jupyter client needed) running on a background
thread, with direct references to the running simulation.
"""

from __future__ import annotations

import sys
import threading
from typing import Any


def start(namespace: dict[str, Any], banner: str = "") -> threading.Thread | None:
    """Start the REPL on a background thread, or skip it if stdin isn't a
    real terminal. Without this guard, an embedded IPython shell reading
    from a non-tty stdin sees immediate EOF on every read and busy-loops
    re-prompting "Do you really want to exit?" at 100% CPU instead of
    exiting once — worth checking explicitly rather than letting it happen.
    """
    if not sys.stdin.isatty():
        print("prismswarm: stdin is not a terminal, skipping the REPL.", file=sys.stderr)
        return None

    def _run() -> None:
        from IPython.terminal.embed import InteractiveShellEmbed

        shell = InteractiveShellEmbed(user_ns=namespace, banner1=banner)
        shell()

    thread = threading.Thread(target=_run, name="prismswarm-repl", daemon=True)
    thread.start()
    return thread
