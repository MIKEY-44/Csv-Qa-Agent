"""
src/agents/retry.py
Retry Agent: Handles execution failures with iterative recovery.
"""
from typing import Tuple, Any, Dict
from dataclasses import dataclass


@dataclass
class RetryResult:
    success: bool
    result: Any
    output: str
    attempts: int
    trace: list


class RetryAgent:
    """Manages retry logic with error feedback."""

    def __init__(self, max_retries: int = 3, base_delay: float = 0.5):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, sandbox, code: str, critic, question: str) -> RetryResult:
        trace = []
        last_code = code

        for attempt in range(1, self.max_retries + 1):
            trace.append({"attempt": attempt, "status": "executing"})
            success, result, output = sandbox.execute(last_code)

            if success:
                trace.append({"attempt": attempt, "status": "success"})
                return RetryResult(success=True, result=result, output=output, attempts=attempt, trace=trace)

            trace.append({"attempt": attempt, "status": "failed", "error": str(result)[:200]})

            if attempt < self.max_retries:
                feedback = critic.generate_feedback(last_code, str(result), question)
                trace.append({"attempt": attempt, "status": "retrying"})
                last_code = feedback

        return RetryResult(success=False, result=result, output=output, attempts=self.max_retries, trace=trace)
