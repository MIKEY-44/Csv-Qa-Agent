"""
csv_qa_agent/core/logging_config.py
Structured logging configuration.
"""
import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Output logs as JSON for structured logging."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, "agent_id"):
            log_obj["agent_id"] = record.agent_id
        if hasattr(record, "question"):
            log_obj["question"] = record.question
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        if hasattr(record, "success"):
            log_obj["success"] = record.success
        return json.dumps(log_obj)


def setup_logging(level=logging.INFO, json_format=False):
    """Setup structured logging."""
    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )

    handler.setFormatter(formatter)

    logger = logging.getLogger("csv_qa_agent")
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
