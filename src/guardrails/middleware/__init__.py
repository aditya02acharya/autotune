"""
Guardrails Middleware - FastAPI Integration

Integrates all three tiers into a unified middleware with parallel execution,
signal aggregation, and latency budgeting (<200ms p95).
"""

from .api import app, GuardrailRequest, GuardrailResponse

__all__ = ["app", "GuardrailRequest", "GuardrailResponse"]
