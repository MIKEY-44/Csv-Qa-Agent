"""
src/agents/critic.py
Critic Agent: Self-reflection for code review.
"""
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class CriticReview:
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    safety_score: float = 0.0
    correctness_score: float = 0.0
    simplicity_score: float = 0.0
    overall_score: float = 0.0


class CriticAgent:
    """
    Self-reflection agent that reviews generated code.
    Checks: safety, correctness, simplicity, relevance.
    """

    FORBIDDEN_PATTERNS = [
        ('open(', 'File I/O not allowed'),
        ('exec(', 'Dynamic execution not allowed'),
        ('eval(', 'Dynamic evaluation not allowed'),
        ('__import__', 'Dynamic imports not allowed'),
        ('subprocess', 'Subprocess not allowed'),
        ('os.system', 'System calls not allowed'),
        ('os.popen', 'System calls not allowed'),
        ('import os', 'OS module not allowed'),
        ('import sys', 'System module not allowed'),
        ('import socket', 'Network not allowed'),
        ('import urllib', 'Network not allowed'),
        ('import requests', 'Network not allowed'),
    ]

    def review(self, code: str, question: str, schema: Dict = None) -> CriticReview:
        """Comprehensive code review."""
        issues = []
        suggestions = []

        # 1. Safety Analysis
        safety_issues = self._check_safety(code)
        issues.extend(safety_issues)

        # 2. Correctness Analysis
        correctness_issues = self._check_correctness(code)
        issues.extend(correctness_issues)

        # 3. Simplicity Analysis
        simplicity_suggestions = self._check_simplicity(code)
        suggestions.extend(simplicity_suggestions)

        # 4. Relevance Analysis
        relevance_suggestions = self._check_relevance(code, question)
        suggestions.extend(relevance_suggestions)

        # Calculate scores
        safety_score = 100.0 if not safety_issues else max(0, 100 - len(safety_issues) * 25)
        correctness_score = 100.0 if not correctness_issues else max(0, 100 - len(correctness_issues) * 30)
        simplicity_score = max(0, 100 - len(simplicity_suggestions) * 10)

        overall = (safety_score * 0.4 + correctness_score * 0.35 + simplicity_score * 0.25)

        return CriticReview(
            passed=len(issues) == 0 and safety_score == 100.0,
            issues=issues,
            suggestions=suggestions,
            safety_score=round(safety_score, 1),
            correctness_score=round(correctness_score, 1),
            simplicity_score=round(simplicity_score, 1),
            overall_score=round(overall, 1)
        )

    def _check_safety(self, code: str) -> List[str]:
        """Check for unsafe code patterns."""
        issues = []
        for pattern, reason in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                issues.append(f"SAFETY: {reason} (found '{pattern}')")

        # Check for dangerous attributes
        dangerous_attrs = ['__subclasses__', '__bases__', '__globals__', '__code__', '__class__']
        for attr in dangerous_attrs:
            if attr in code:
                issues.append(f"SAFETY: Dangerous attribute access '{attr}'")

        return issues

    def _check_correctness(self, code: str) -> List[str]:
        """Check code correctness."""
        issues = []

        if 'result' not in code:
            issues.append("CORRECTNESS: Missing 'result' variable")

        if 'df' not in code:
            issues.append("CORRECTNESS: Code does not reference dataframe 'df'")

        # Check for balanced braces in dict literals
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            issues.append(f"CORRECTNESS: Unbalanced braces ({open_braces} open, {close_braces} close)")

        return issues

    def _check_simplicity(self, code: str) -> List[str]:
        """Check code simplicity."""
        suggestions = []
        lines = code.strip().split('\n')

        if len(lines) > 25:
            suggestions.append(f"SIMPLICITY: Code is {len(lines)} lines; consider simplifying")

        if code.count('for ') > 3:
            suggestions.append("SIMPLICITY: Multiple loops; consider vectorized operations")

        if code.count('if ') > 5:
            suggestions.append("SIMPLICITY: Many conditionals; consider simplifying logic")

        return suggestions

    def _check_relevance(self, code: str, question: str) -> List[str]:
        """Check if code addresses the question."""
        suggestions = []
        q_words = set(w.lower() for w in question.split() if len(w) > 3)
        code_lower = code.lower()

        # Check for key concepts from question
        key_concepts = ['revenue', 'sales', 'total', 'average', 'growth', 'compare']
        matched = sum(1 for c in key_concepts if c in q_words and c in code_lower)

        if matched == 0 and len(q_words) > 0:
            suggestions.append("RELEVANCE: Code may not address the question directly")

        return suggestions

    def generate_feedback(self, code: str, error: str, question: str) -> str:
        """Generate feedback prompt for retry."""
        return f"""The previous code failed with error:
{error}

Please fix the code to answer: "{question}"

Requirements:
1. Write ONLY Python code. No markdown.
2. df is pre-loaded. Set `result` = {{'answer': str, 'data': any, 'viz_type': str|None}}
3. No imports, no file I/O.
4. Compute actual values from data.

Failed code:
{code}"""
