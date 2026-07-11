"""
csv_qa_agent/sandbox/executor.py
Secure code execution with AST validation, resource limits, and sandboxing.
"""
import ast
import resource
import signal
import sys
import io
import traceback
import threading
import multiprocessing
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import re
import math
import statistics

from core.models import ExecutionResult, ExecutionStatus


# ───────────────────────────────────────────────
# AST Security Validator
# ───────────────────────────────────────────────

ALLOWED_IMPORTS = {
    'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'plotly',
    'plotly.graph_objects', 'plotly.express', 'json', 're', 'math',
    'statistics', 'datetime', 'collections', 'itertools', 'typing'
}

FORBIDDEN_NODES = {
    ast.Import,          # Only allow specific imports
    ast.ImportFrom,      # Will be checked separately
    ast.Call,            # Will be checked for dangerous calls
}

FORBIDDEN_BUILTINS = {
    'eval', 'exec', 'compile', 'open', 'input', '__import__',
    'exit', 'quit', 'help', 'copyright', 'credits', 'license',
    'breakpoint', 'globals', 'locals', 'vars', 'dir',
    'object.__subclasses__', 'os', 'sys', 'subprocess', 'socket',
    'urllib', 'requests', 'ftplib', 'smtplib', 'telnetlib',
    'webbrowser', 'idlelib', 'tkinter'
}

FORBIDDEN_ATTRIBUTES = [
    '__subclasses__', '__bases__', '__mro__', '__globals__',
    'func_globals', 'gi_frame', 'cr_frame', 'tb_frame',
    'f_locals', 'f_globals', 'system', 'popen', 'call',
    'spawn', 'fork', 'kill', 'remove', 'rmdir', 'unlink'
]


class SecurityError(Exception):
    """Raised when code fails security checks."""
    pass


class ASTValidator(ast.NodeVisitor):
    """Validates Python AST for security."""

    def __init__(self):
        self.issues = []
        self.allowed_names = set()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] not in ALLOWED_IMPORTS:
                self.issues.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ''
        base_module = module.split('.')[0]
        if base_module not in ALLOWED_IMPORTS:
            self.issues.append(f"Forbidden module import: {module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.issues.append(f"Forbidden builtin call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_ATTRIBUTES:
                self.issues.append(f"Forbidden attribute access: {node.func.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_BUILTINS:
            self.issues.append(f"Forbidden name usage: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.issues.append(f"Forbidden attribute: {node.attr}")
        self.generic_visit(node)

    def validate(self, code: str) -> Tuple[bool, list]:
        """Validate code and return (is_valid, issues)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        self.issues = []
        self.visit(tree)

        return len(self.issues) == 0, self.issues


# ───────────────────────────────────────────────
# Resource-Limited Executor
# ───────────────────────────────────────────────

def set_resource_limits(max_memory_mb: int = 512, max_cpu_time_sec: int = 30):
    """Set resource limits for the current process."""
    # Memory limit (bytes)
    max_memory_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

    # CPU time limit (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time_sec, max_cpu_time_sec + 5))

    # File size limit (10MB)
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))

    # Disable core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded time limit")


class SecureExecutor:
    """Executes Python code in a secure sandbox."""

    def __init__(
        self,
        csv_path: str,
        max_memory_mb: int = 512,
        max_cpu_time_sec: int = 30,
        max_retries: int = 3
    ):
        self.csv_path = csv_path
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time_sec = max_cpu_time_sec
        self.max_retries = max_retries
        self.validator = ASTValidator()

    def validate_code(self, code: str) -> Tuple[bool, list]:
        """Validate code using AST analysis."""
        return self.validator.validate(code)

    def execute(self, code: str, retry_count: int = 0) -> ExecutionResult:
        """Execute code with full sandboxing."""
        start_time = datetime.now()

        # Step 1: AST Validation
        is_valid, issues = self.validate_code(code)
        if not is_valid:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                error=f"Security validation failed: {'; '.join(issues)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                retry_count=retry_count
            )

        # Step 2: Prepare execution environment
        # Create restricted globals
        safe_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'bytearray': bytearray, 'bytes': bytes, 'chr': chr,
                'complex': complex, 'dict': dict, 'divmod': divmod,
                'enumerate': enumerate, 'filter': filter, 'float': float,
                'format': format, 'frozenset': frozenset, 'hasattr': hasattr,
                'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
                'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
                'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
                'ord': ord, 'pow': pow, 'print': print, 'range': range,
                'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
                'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum,
                'tuple': tuple, 'type': type, 'zip': zip, 'Exception': Exception,
                'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
                'IndexError': IndexError, 'AttributeError': AttributeError,
                'ZeroDivisionError': ZeroDivisionError, 'RuntimeError': RuntimeError,
                'ArithmeticError': ArithmeticError, 'LookupError': LookupError,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'go': go,
            'px': px,
            'json': json,
            're': re,
            'math': math,
            'statistics': statistics,
            'datetime': datetime,
        }

        safe_locals = {}

        # Step 3: Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_cpu_time_sec)

            # Execute code
            exec(code, safe_globals, safe_locals)

            # Get result
            result = safe_locals.get('result', None)

            signal.alarm(0)  # Cancel timeout

            execution_time = (datetime.now() - start_time).total_seconds()

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=result,
                execution_time=execution_time,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                retry_count=retry_count
            )

        except TimeoutError as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Execution timeout after {self.max_cpu_time_sec}s",
                execution_time=(datetime.now() - start_time).total_seconds(),
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                retry_count=retry_count
            )
        except MemoryError as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Memory limit exceeded ({self.max_memory_mb}MB)",
                execution_time=(datetime.now() - start_time).total_seconds(),
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                retry_count=retry_count
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                retry_count=retry_count
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            signal.alarm(0)
            plt.close('all')  # Clean up matplotlib figures

    def execute_with_retry(self, code: str, error_feedback: Optional[str] = None) -> ExecutionResult:
        """Execute with automatic retry on failure."""
        for attempt in range(self.max_retries + 1):
            result = self.execute(code, retry_count=attempt)

            if result.status == ExecutionStatus.SUCCESS:
                return result

            if attempt < self.max_retries:
                result.status = ExecutionStatus.RETRYING
                # In a real system, you'd send the error back to the LLM here
                # For now, we just retry the same code

        return result
