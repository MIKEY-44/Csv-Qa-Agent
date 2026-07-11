"""
core/sandbox.py
Secure execution environment with Windows-compatible timeout (no SIGALRM).
"""
import ast
import io
import contextlib
import threading
from typing import Tuple, Any

class SecureSandbox:
    """Sandbox with cross-platform timeout - no SIGALRM (Unix-only)."""

    ALLOWED_BUILTINS = {
        'len', 'range', 'round', 'sum', 'min', 'max', 'abs', 'float', 'int', 'str',
        'dict', 'list', 'tuple', 'set', 'sorted', 'zip', 'enumerate', 'map', 'filter',
        'bool', 'type', 'isinstance', 'hasattr', 'getattr', 'print', 'ord', 'chr',
        'hex', 'bin', 'oct', 'pow', 'divmod', 'all', 'any', 'reversed', 'slice'
    }

    def __init__(self, df, timeout: float = 5.0):
        self.df = df.copy()
        self.timeout = timeout
        self._compile_builtins()

    def _compile_builtins(self):
        """Build restricted builtins dict."""
        import builtins
        safe = {}
        for name in self.ALLOWED_BUILTINS:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)
        for exc_name in ['Exception', 'ValueError', 'TypeError', 'KeyError', 
                          'IndexError', 'AttributeError', 'ZeroDivisionError',
                          'RuntimeError', 'StopIteration', 'NotImplementedError']:
            if hasattr(builtins, exc_name):
                safe[exc_name] = getattr(builtins, exc_name)
        self._safe_builtins = safe

    def _validate_ast(self, code: str) -> Tuple[bool, str]:
        """Static analysis to block dangerous operations."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Imports are not allowed in sandbox"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('open', 'exec', 'eval', 'compile', '__import__'):
                        return False, f"'{node.func.id}' is not allowed"
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('__subclasses__', '__bases__', '__globals__',
                                         'mro', 'func_globals', 'gi_frame', 'f_locals',
                                         'f_globals'):
                        return False, f"Attribute '{node.func.attr}' is blocked"
            if isinstance(node, ast.Delete):
                return False, "Delete statements are not allowed"
        return True, ""

    def _run_with_timeout(self, code: str, globals_dict: dict, locals_dict: dict) -> Tuple[bool, Any, str]:
        """Execute code with timeout using threading (Windows-compatible)."""
        result_container = [None]
        exception_container = [None]
        stdout_buffer = io.StringIO()

        def target():
            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(code, globals_dict, locals_dict)
                result_container[0] = locals_dict.get('result')
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return False, None, f"Execution timed out after {self.timeout}s"
        if exception_container[0] is not None:
            return False, None, f"{type(exception_container[0]).__name__}: {exception_container[0]}"
        if result_container[0] is None:
            return False, None, "'result' variable was not set"
        return True, result_container[0], stdout_buffer.getvalue()

    def run(self, code: str) -> Tuple[bool, Any, str]:
        valid, msg = self._validate_ast(code)
        if not valid:
            return False, None, msg

        import pandas as pd
        import numpy as np
        from datetime import datetime

        safe_globals = {
            "pd": pd, "np": np, "df": self.df, "datetime": datetime,
            "__builtins__": self._safe_builtins
        }
        safe_locals = {}
        return self._run_with_timeout(code, safe_globals, safe_locals)
