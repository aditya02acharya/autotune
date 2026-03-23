"""
FastAPI Middleware for Guardrails

Provides REST API for guardrail checks with parallel execution of all tiers.
Target: p95 latency < 200ms
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_future
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import all tiers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails.tier1.engine import PolicyReasoningEngine, Signal, Action
from guardrails.tier2.deterministic import Tier2Deterministic
from guardrails.tier2.neural_pre_screen import NeuralPreScreen
from guardrails.reasoning.pgm import PGMReasoner


# Request/Response models
class GuardrailRequest(BaseModel):
    """Request model for guardrail check."""
    text: str
    check_pii: bool = False
    use_neural: bool = True
    metadata: dict = {}


class GuardrailResponse(BaseModel):
    """Response model for guardrail check."""
    action: str  # allow, block, warn, mask, escalate
    reason: str
    confidence: float
    latency_ms: float
    tier_results: dict
    triggered_rules: list[str] = []


# Create FastAPI app
app = FastAPI(
    title="Guardrails API",
    description="Three-tier adaptive guardrail middleware",
    version="0.1.0",
)

# Initialize components lazily
_engine: Optional[PolicyReasoningEngine] = None
_deterministic: Optional[Tier2Deterministic] = None
_neural_screen: Optional[NeuralPreScreen] = None
_pgm_reasoner: Optional[PGMReasoner] = None

# Thread pool for parallel execution
_executor = ThreadPoolExecutor(max_workers=4)


def get_engine() -> PolicyReasoningEngine:
    """Get or create policy engine."""
    global _engine
    if _engine is None:
        _engine = PolicyReasoningEngine()
    return _engine


def get_deterministic() -> Tier2Deterministic:
    """Get or create deterministic filter."""
    global _deterministic
    if _deterministic is None:
        _deterministic = Tier2Deterministic()
    return _deterministic


def get_neural_screen() -> NeuralPreScreen:
    """Get or create neural pre-screen."""
    global _neural_screen
    if _neural_screen is None:
        _neural_screen = NeuralPreScreen(threshold=0.5)
    return _neural_screen


def get_pgm_reasoner() -> PGMReasoner:
    """Get or create PGM reasoner."""
    global _pgm_reasoner
    if _pgm_reasoner is None:
        _pgm_reasoner = PGMReasoner()
    return _pgm_reasoner


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tiers": {
            "tier1": "policy_engine",
            "tier2": "deterministic + neural",
            "tier3": "lora_ensemble",
            "cross_tier": "pgm_reasoning",
        }
    }


@app.post("/check", response_model=GuardrailResponse)
async def check_text(request: GuardrailRequest):
    """
    Check text against all guardrail tiers.

    Runs Tier 2 (deterministic + neural) in parallel, then applies
    Tier 1 policy reasoning and cross-tier PGM logic.
    """
    start_time = time.perf_counter()

    # Run deterministic checks (fast, <5ms)
    det_result = get_deterministic().check(request.text, request.check_pii)

    # Prepare signals list
    signals = []

    # Add deterministic signal
    if det_result["injection"].blocked:
        signals.append(Signal(
            source="tier2_deterministic",
            category="prompt_injection",
            score=1.0,
            confidence=1.0,
            blocked=True,
            reason=det_result["injection"].reason or "Matched injection patterns",
            latency_ms=det_result["injection"].latency_ms,
        ))

    # Add neural pre-screen signal (run in parallel if needed)
    neural_result = None
    if request.use_neural and not det_result["injection"].blocked:
        neural_result = get_neural_screen().check(request.text)
        signals.append(Signal(
            source="tier2_neural",
            category="prompt_injection",
            score=neural_result.score,
            confidence=neural_result.confidence,
            blocked=neural_result.score >= 0.5,
            reason=f"Neural classifier score: {neural_result.score:.3f}",
            latency_ms=neural_result.latency_ms,
        ))

    # Add PGM reasoning
    classifier_outputs = {}
    for signal in signals:
        if signal.category not in classifier_outputs:
            classifier_outputs[signal.category] = signal.score
        else:
            # Take max score
            classifier_outputs[signal.category] = max(
                classifier_outputs[signal.category], signal.score
            )

    pgm_result = get_pgm_reasoner().reason(classifier_outputs)

    # Add PGM signal if compositional violation detected
    if pgm_result.violated and pgm_result.compositional_score > 0:
        signals.append(Signal(
            source="pgm_reasoning",
            category="compositional",
            score=pgm_result.compositional_score,
            confidence=pgm_result.confidence,
            blocked=pgm_result.violated,
            reason=f"Compositional violation: {', '.join(pgm_result.triggered_rules)}",
        ))

    # Apply Tier 1 policy reasoning
    decision = get_engine().evaluate(signals)

    total_latency = (time.perf_counter() - start_time) * 1000

    # Build response
    tier_results = {
        "deterministic": {
            "blocked": det_result["injection"].blocked,
            "latency_ms": det_result["injection"].latency_ms,
        },
        "summary": det_result["summary"],
    }

    if neural_result:
        tier_results["neural"] = {
            "is_safe": neural_result.is_safe,
            "score": neural_result.score,
            "latency_ms": neural_result.latency_ms,
        }

    tier_results["pgm"] = {
        "violated": pgm_result.violated,
        "compositional_score": pgm_result.compositional_score,
    }

    return GuardrailResponse(
        action=decision.action.value,
        reason=decision.reason,
        confidence=decision.confidence,
        latency_ms=total_latency,
        tier_results=tier_results,
        triggered_rules=decision.triggered_rules,
    )


@app.post("/check/batch")
async def check_batch(texts: list[str]):
    """Check multiple texts in batch."""
    results = []
    for text in texts:
        request = GuardrailRequest(text=text)
        result = await check_text(request)
        results.append(result)
    return results


@app.post("/policy/rule")
async def add_rule(
    rule_id: str,
    name: str,
    description: str,
    category: str,
    threshold: float = 0.5,
    action: str = "block",
):
    """Add a new policy rule."""
    from guardrails.tier1.engine import PolicyRule, Action as PolicyAction

    def make_condition(cat, thresh):
        def condition(signals):
            return signals.get(cat, 0) >= thresh
        return condition

    rule = PolicyRule(
        id=rule_id,
        name=name,
        description=description,
        category=category,
        condition=make_condition(category, threshold),
        action=PolicyAction[action.upper()],
    )

    get_engine().add_rule(rule)

    return {"status": "added", "rule_id": rule_id}


@app.get("/stats")
async def get_stats():
    """Get guardrail statistics."""
    return {
        "engine_rules": len(get_engine().rules),
        "pgm_rules": len(get_pgm_reasoner().rules),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
