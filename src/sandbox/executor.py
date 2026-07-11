"""
src/sandbox/executor.py
Secure Sandbox: Restricted code execution environment.
"""
import ast
import io
import contextlib
import signal
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False  # Windows doesn't have resource module
import sys
from typing import Tuple, Any


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


class SecureSandbox:
    """
    Production-grade secure sandbox for executing generated Python code.

    Restrictions:
    - AST validation blocks dangerous imports/calls
    - 5-second execution timeout
    - 100MB memory limit
    - Restricted builtins (no file I/O, no network, no subprocess)
    """

    TIMEOUT_SECONDS = 5
    MAX_MEMORY_MB = 100

    ALLOWED_BUILTINS = {
        'len', 'range', 'round', 'sum', 'min', 'max', 'abs',
        'float', 'int', 'str', 'dict', 'list', 'tuple', 'set',
        'sorted', 'enumerate', 'zip', 'map', 'filter', 'any', 'all',
        'isinstance', 'hasattr', 'getattr', 'print', 'type',
        'vars', 'dir', 'repr', 'format', 'divmod', 'pow'
    }

    FORBIDDEN_CALLS = {
        'open', 'exec', 'eval', 'compile', '__import__',
        'subprocess', 'os.system', 'os.popen', 'os.remove',
        'os.rmdir', 'os.mkdir', 'os.makedirs', 'os.chdir',
        'input', 'exit', 'quit'
    }

    def __init__(self, df):
        self.df = df.copy()
        self._setup_restrictions()

    def _setup_restrictions(self):
        """Setup resource limits (Unix only)."""
        if HAS_RESOURCE:
            try:
                memory_limit = self.MAX_MEMORY_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            except (ValueError, OSError):
                pass

    def validate_ast(self, code: str) -> Tuple[bool, str]:
        """Static analysis to block dangerous code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Imports not allowed in sandbox"

            # Block forbidden calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                    return False, f"'{node.func.id}' is not allowed"

                if isinstance(node.func, ast.Attribute):
                    attr_chain = []
                    current = node.func
                    while isinstance(current, ast.Attribute):
                        attr_chain.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attr_chain.append(current.id)
                    full_name = '.'.join(reversed(attr_chain))

                    forbidden_prefixes = ['os.system', 'os.popen', 'subprocess', 'sys.exit']
                    if any(full_name.startswith(p) for p in forbidden_prefixes):
                        return False, f"Forbidden call: {full_name}"

            # Block dangerous attributes
            if isinstance(node, ast.Attribute):
                if node.attr in ['__subclasses__', '__bases__', '__globals__', '__code__', '__class__']:
                    return False, f"Forbidden attribute: {node.attr}"

        return True, "OK"

    def execute(self, code: str) -> Tuple[bool, Any, str]:
        """Execute code with full restrictions."""
        import pandas as pd
        import numpy as np
        from datetime import datetime

        valid, msg = self.validate_ast(code)
        if not valid:
            return False, None, msg

        builtins_dict = __builtins__.__dict__ if hasattr(__builtins__, '__dict__') else __builtins__
        safe_builtins = {name: builtins_dict[name] for name in self.ALLOWED_BUILTINS if name in builtins_dict}

        safe_globals = {
            "pd": pd,
            "np": np,
            "df": self.df,
            "datetime": datetime,
            "__builtins__": safe_builtins
        }
        safe_locals = {}
        stdout = io.StringIO()

        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.TIMEOUT_SECONDS)

        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals, safe_locals)

            signal.alarm(0)
            result = safe_locals.get("result")

            if result is None:
                return False, None, "'result' variable not set by the code"

            return True, result, stdout.getvalue()

        except TimeoutException:
            return False, None, f"Execution timed out after {self.TIMEOUT_SECONDS} seconds"
        except Exception as e:
            signal.alarm(0)
            return False, None, f"{type(e).__name__}: {str(e)}"
        finally:
            signal.signal(signal.SIGALRM, old_handler)
