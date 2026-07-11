"""Validator Agent - Checks code safety before execution."""

import ast

class ValidatorAgent:
    """AST-based code validation with security checks."""

    FORBIDDEN_IMPORTS = {'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib'}
    FORBIDDEN_FUNCTIONS = {'open', 'exec', 'eval', 'compile', '__import__'}
    FORBIDDEN_METHODS = {'to_csv', 'to_excel', 'to_sql', 'save', 'to_json'}

    def validate(self, code: str) -> tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_IMPORTS:
                        return False, f"Import '{alias.name}' not allowed"

            if isinstance(node, ast.ImportFrom):
                if node.module in self.FORBIDDEN_IMPORTS:
                    return False, f"Import from '{node.module}' not allowed"

            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_FUNCTIONS:
                    return False, f"Function '{node.func.id}' not allowed"

            # Check method calls
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_METHODS:
                    return False, f"Method '{node.attr}' not allowed"

        return True, "OK"
