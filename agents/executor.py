"""Executor Agent - Runs code safely with timeouts."""

import io
import contextlib
import signal
from typing import Any, Tuple

class TimeoutException(Exception):
    pass

class ExecutorAgent:
    """Sandboxed code execution with timeout."""

    ALLOWED_BUILTINS = {'len', 'range', 'round', 'sum', 'min', 'max', 'abs', 'float', 'int', 'str', 'dict', 'list', 'sorted', 'enumerate', 'zip'}

    def __init__(self, df, timeout: int = 5):
        self.df = df.copy()
        self.timeout = timeout

    def _timeout_handler(self, signum, frame):
        raise TimeoutException("Code execution timed out")

    def execute(self, code: str) -> Tuple[bool, Any, str]:
        """Execute code with timeout. Returns (success, result, output)."""
        import pandas as pd
        import numpy as np
        from datetime import datetime

        builtins_dict = __builtins__.__dict__ if hasattr(__builtins__, '__dict__') else __builtins__
        safe_builtins = {n: builtins_dict[n] for n in self.ALLOWED_BUILTINS if n in builtins_dict}

        safe_globals = {
            "pd": pd, "np": np, "df": self.df, "datetime": datetime,
            "__builtins__": safe_builtins
        }
        safe_locals = {}
        stdout = io.StringIO()

        # Set timeout (Unix only; Windows uses different mechanism)
        try:
            old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)
        except (AttributeError, ValueError):
            old_handler = None

        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals, safe_locals)

            result = safe_locals.get("result")
            if result is None:
                return False, None, "'result' variable not set"
            return True, result, stdout.getvalue()

        except TimeoutException:
            return False, None, f"Execution timed out after {self.timeout} seconds"
        except Exception as e:
            return False, None, f"{type(e).__name__}: {str(e)}"
        finally:
            if old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
