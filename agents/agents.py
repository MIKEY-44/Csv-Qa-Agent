"""
csv_qa_agent/agents/agents.py
Agent implementations: Planner, Code Generator, Critic, Validator.
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import pandas as pd

from core.models import (
    ExecutionPlan, GeneratedCode, CriticReview,
    ExecutionResult, ExecutionStatus, Message
)
from prompts.templates import (
    PLANNER_SYSTEM_PROMPT, PLANNER_FEW_SHOT_EXAMPLES,
    CODE_GENERATOR_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT,
    CRITIC_FEW_SHOT_EXAMPLES, RETRY_PROMPT, CONFIDENCE_PROMPT
)


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.history: List[Message] = []

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM API. In production, this calls OpenAI/Anthropic/etc.
        For demo, we simulate with a mock response.
        """
        return self._mock_llm_call(system_prompt, user_prompt)

    def _mock_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Mock LLM for demonstration purposes."""
        return '{"mock": true, "message": "Replace with actual LLM API call"}'

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return {}


class PlannerAgent(BaseAgent):
    """Plans execution steps based on user question and CSV schema."""

    def plan(self, question: str, csv_path: str) -> ExecutionPlan:
        """Create an execution plan for the given question."""
        df = pd.read_csv(csv_path)
        columns = list(df.columns)
        sample = df.head(3).to_dict('records')

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            columns=columns,
            sample=json.dumps(sample, indent=2)
        )

        few_shot = "\n\n## Examples\n"
        for ex in PLANNER_FEW_SHOT_EXAMPLES:
            few_shot += f"Question: {ex['question']}\n"
            few_shot += f"Schema: {ex['schema']}\n"
            few_shot += f"Output: {json.dumps(ex['output'])}\n\n"

        user_prompt = f"{few_shot}\n## Your Task\nQuestion: {question}\nSchema: columns: {columns}\n"

        response = self._call_llm(system_prompt, user_prompt)
        parsed = self._extract_json(response)

        return ExecutionPlan(
            steps=parsed.get('steps', []),
            reasoning=parsed.get('reasoning', ''),
            expected_output_type=parsed.get('expected_output_type', 'text'),
            confidence=parsed.get('confidence', 0.5)
        )


class CodeGeneratorAgent(BaseAgent):
    """Generates Python code to answer the question."""

    def generate(
        self,
        question: str,
        plan: ExecutionPlan,
        csv_path: str,
        previous_error: Optional[str] = None
    ) -> GeneratedCode:
        """Generate Python code based on the execution plan."""

        if previous_error:
            system_prompt = "You are fixing code that previously failed."
            user_prompt = RETRY_PROMPT.format(
                error=previous_error,
                previous_code="# previous code would be here",
                trace="# execution trace"
            )
        else:
            system_prompt = CODE_GENERATOR_SYSTEM_PROMPT.format(csv_path=csv_path)
            user_prompt = f"""
Question: {question}

Execution Plan:
{"\n".join(f"{i+1}. {step}" for i, step in enumerate(plan.steps))}

Reasoning: {plan.reasoning}
Expected Output Type: {plan.expected_output_type}

Generate Python code to answer this question.
"""

        response = self._call_llm(system_prompt, user_prompt)

        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response.strip()

        imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)', code, re.MULTILINE)
        import_list = [imp[0] or imp[1] for imp in imports]

        return GeneratedCode(
            code=code,
            imports=import_list,
            reasoning=f"Generated code for: {question}"
        )


class CriticAgent(BaseAgent):
    """Reviews generated code for correctness, safety, and quality."""

    def review(
        self,
        code: GeneratedCode,
        question: str,
        plan: ExecutionPlan
    ) -> CriticReview:
        """Review generated code and return approval decision."""

        system_prompt = CRITIC_SYSTEM_PROMPT

        few_shot = "\n\n## Examples\n"
        for ex in CRITIC_FEW_SHOT_EXAMPLES:
            few_shot += f"Code: {ex['code']}\n"
            few_shot += f"Question: {ex['question']}\n"
            few_shot += f"Review: {json.dumps(ex['output'])}\n\n"

        user_prompt = f"""
{few_shot}

## Code to Review
```python
{code.code}
```

## Context
Question: {question}
Expected Output Type: {plan.expected_output_type}
Execution Plan: {"\n".join(plan.steps)}

Please review this code.
"""

        response = self._call_llm(system_prompt, user_prompt)
        parsed = self._extract_json(response)

        return CriticReview(
            approved=parsed.get('approved', False),
            issues=parsed.get('issues', []),
            suggestions=parsed.get('suggestions', []),
            safety_concerns=parsed.get('safety_concerns', []),
            correctness_score=parsed.get('correctness_score', 0.0),
            simplification_score=parsed.get('simplification_score', 0.0),
            reasoning=parsed.get('reasoning', '')
        )


class ValidatorAgent:
    """Validates code using AST analysis before execution."""

    def __init__(self):
        from sandbox.executor import ASTValidator
        self.validator = ASTValidator()

    def validate(self, code: str) -> Dict[str, Any]:
        """Validate code and return detailed report."""
        is_valid, issues = self.validator.validate(code)

        return {
            'is_valid': is_valid,
            'issues': issues,
            'safety_score': 1.0 if is_valid else max(0.0, 1.0 - len(issues) * 0.2),
            'line_count': len(code.split('\n')),
            'import_count': len(re.findall(r'^import|^from', code, re.MULTILINE))
        }
