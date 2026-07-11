"""
csv_qa_agent/agents/critic.py
Critic Agent: Self-reflection for code review.
"""
from typing import Dict, List
from core.models import CodeReview


class CriticAgent:
    """
    Self-reflection agent that reviews generated code.
    Checks: safety, correctness, simplicity.
    """

    FORBIDDEN_PATTERNS = ['open(', 'exec(', 'eval(', '__import__', 'subprocess', 'os.system']

    def review(self, code: str, question: str, schema: Dict = None) -> CodeReview:
        """Review code for correctness, safety, and simplicity."""
        issues = []
        suggestions = []

        # Safety checks
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                issues.append(f"Unsafe pattern: {pattern}")

        # Correctness checks
        if 'result' not in code:
            issues.append("Missing 'result' variable")

        if 'df[' not in code and 'df.' not in code:
            suggestions.append("Code may not reference dataframe 'df'")

        # Question relevance
        question_keywords = set(w for w in question.lower().split() if len(w) > 3)
        code_lower = code.lower()
        matches = sum(1 for kw in question_keywords if kw in code_lower)
        if matches < 1:
            suggestions.append("Code may not address the question directly")

        # Simplicity
        if code.count('\n') > 30:
            suggestions.append("Code is complex; consider simplifying")

        safety_score = 100.0 if not any(p in code for p in self.FORBIDDEN_PATTERNS) else 0.0
        correctness_score = 100.0 if 'result' in code else 50.0
        simplicity_score = max(0, 100 - len(code) / 10)

        return CodeReview(
            passed=len(issues) == 0 and safety_score == 100.0,
            issues=issues,
            suggestions=suggestions,
            safety_score=safety_score,
            correctness_score=correctness_score,
            simplicity_score=simplicity_score
        )

    def suggest_fix(self, code: str, error: str, question: str) -> str:
        """Generate retry prompt with error feedback."""
        return f"""The previous code failed with: {error}

Please fix the code to answer: "{question}"

Rules:
1. Write ONLY Python code. No markdown, no explanations.
2. df is pre-loaded. Store answer in `result` = {{'answer': str, 'data': any, 'viz_type': 'table'|'bar'|'line'|'number'|None}}
3. Compute actual values. No hardcoded numbers.
4. No imports, no file I/O.

Previous code that failed:
```python
{code}
```"""
