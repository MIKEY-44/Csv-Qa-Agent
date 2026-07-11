"""
core/models.py
Data models for CSV QA Agent.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class AgentResponse:
    answer: str
    confidence: float = 0.95
    viz_type: str = ""
    data: Any = None
    execution_trace: list = field(default_factory=list)
    code: str = ""
    mode: str = "rule-based"
    latency_ms: float = 0.0
