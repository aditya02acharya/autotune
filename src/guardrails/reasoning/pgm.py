"""
PGM Reasoning Layer

Implements R²-Guard style probabilistic graphical model reasoning
for compositional violation detection.

Key insight: Individual classifiers detect single categories well,
but compliance violations are often compositional - they arise from
the combination of multiple categories.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import time


class LogicalOperator(Enum):
    """Logical operators for rule conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class ComplianceRule:
    """
    A compliance rule expressed as first-order logic.

    Example: "IF financial_advice AND NOT disclaimer AND retail_customer -> ESCALATE"
    """
    id: str
    name: str
    description: str
    conditions: list[tuple[str, str, float]]  # (category, operator, threshold)
    logic: LogicalOperator = LogicalOperator.AND
    action: str = "escalate"  # block, warn, escalate, log
    priority: int = 1  # Higher = more important
    enabled: bool = True

    def evaluate(self, classifier_outputs: dict[str, float]) -> bool:
        """
        Evaluate rule against classifier outputs.

        Args:
            classifier_outputs: Dict mapping category -> score (0.0 - 1.0)

        Returns:
            True if rule is triggered
        """
        if not self.enabled:
            return False

        results = []
        for category, operator, threshold in self.conditions:
            score = classifier_outputs.get(category, 0.0)

            if operator == ">":
                results.append(score > threshold)
            elif operator == ">=":
                results.append(score >= threshold)
            elif operator == "<":
                results.append(score < threshold)
            elif operator == "<=":
                results.append(score <= threshold)
            elif operator == "==":
                results.append(abs(score - threshold) < 0.01)
            elif operator == "not" or operator == "!":
                results.append(score < threshold)
            else:
                # Default: check if above threshold
                results.append(score >= threshold)

        if self.logic == LogicalOperator.AND:
            return all(results)
        elif self.logic == LogicalOperator.OR:
            return any(results)
        else:
            return results[0] if results else False


@dataclass
class ViolationResult:
    """Result of PGM reasoning."""
    violated: bool
    triggered_rules: list[str]
    action: str
    confidence: float
    compositional_score: float
    latency_ms: float
    details: dict = field(default_factory=dict)


class PGMReasoner:
    """
    Probabilistic Graphical Model Reasoner.

    Combines individual classifier outputs using logical rules
    to detect compositional violations.

    Example:
        # Individual classifiers might miss:
        # - financial_advice: 0.6 (below threshold)
        # - no_disclaimer: 0.7 (detected)
        # - retail_customer: 0.9 (detected)

        # But combined: "financial_advice AND no_disclaimer AND retail_customer"
        # indicates a compositional violation
    """

    def __init__(self):
        self.rules: list[ComplianceRule] = []
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default compliance rules."""
        # Prompt injection compositional rules
        self.rules.extend([
            ComplianceRule(
                id="pi_injection_high_confidence",
                name="High Confidence Injection",
                description="Single high-confidence injection detection",
                conditions=[("prompt_injection", ">=", 0.9)],
                logic=LogicalOperator.OR,
                action="block",
                priority=3,
            ),
            ComplianceRule(
                id="pi_deterministic_neural_agree",
                name="Deterministic + Neural Agreement",
                description="Both deterministic and neural agree on injection",
                conditions=[
                    ("deterministic_injection", ">=", 0.5),
                    ("neural_injection", ">=", 0.7),
                ],
                logic=LogicalOperator.AND,
                action="block",
                priority=3,
            ),
            ComplianceRule(
                id="pi_roleplay_bypass",
                name="Roleplay + Bypass Attempt",
                description="Role-playing attack combined with bypass attempt",
                conditions=[
                    ("roleplay_pattern", ">=", 0.5),
                    ("bypass_pattern", ">=", 0.5),
                ],
                logic=LogicalOperator.AND,
                action="block",
                priority=4,
            ),
            ComplianceRule(
                id="pi_medium_escalate",
                name="Medium Confidence - Escalate",
                description="Medium confidence injection - escalate for review",
                conditions=[("prompt_injection", ">=", 0.6)],
                logic=LogicalOperator.OR,
                action="escalate",
                priority=2,
            ),
            ComplianceRule(
                id="pi_low_warn",
                name="Low Confidence - Warn",
                description="Low confidence injection - warn user",
                conditions=[("prompt_injection", ">=", 0.4)],
                logic=LogicalOperator.OR,
                action="warn",
                priority=1,
            ),
        ])

    def add_rule(self, rule: ComplianceRule):
        """Add a custom compliance rule."""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def reason(
        self,
        classifier_outputs: dict[str, float],
        tier2_signals: dict = None,
    ) -> ViolationResult:
        """
        Apply PGM reasoning to classifier outputs.

        Args:
            classifier_outputs: Dict mapping category -> score
            tier2_signals: Additional signals from Tier 2

        Returns:
            ViolationResult with triggered rules and recommended action
        """
        start_time = time.perf_counter()

        # Merge tier2 signals if provided
        if tier2_signals:
            if tier2_signals.get("deterministic_blocked"):
                classifier_outputs["deterministic_injection"] = 1.0
            if "neural_score" in tier2_signals:
                classifier_outputs["neural_injection"] = tier2_signals["neural_score"]

        # Ensure we have prompt_injection aggregate
        if "prompt_injection" not in classifier_outputs:
            # Use max of individual injection scores
            injection_keys = ["lora_injection", "neural_injection", "deterministic_injection"]
            scores = [classifier_outputs.get(k, 0.0) for k in injection_keys]
            classifier_outputs["prompt_injection"] = max(scores) if scores else 0.0

        # Evaluate all rules
        triggered_rules = []
        actions = []

        for rule in self.rules:
            if rule.evaluate(classifier_outputs):
                triggered_rules.append(rule.id)
                actions.append((rule.action, rule.priority, rule.name))

        # Determine final action (highest priority)
        if not actions:
            final_action = "allow"
            violated = False
        else:
            actions.sort(key=lambda x: x[1], reverse=True)
            final_action = actions[0][0]
            violated = True

        # Calculate compositional score
        # This is the joint probability estimate based on rule satisfaction
        compositional_score = self._calculate_compositional_score(
            classifier_outputs, triggered_rules
        )

        latency = (time.perf_counter() - start_time) * 1000

        return ViolationResult(
            violated=violated,
            triggered_rules=triggered_rules,
            action=final_action,
            confidence=compositional_score,
            compositional_score=compositional_score,
            latency_ms=latency,
            details={
                "classifier_outputs": classifier_outputs,
                "triggered_actions": [a[2] for a in actions],
            }
        )

    def _calculate_compositional_score(
        self,
        outputs: dict[str, float],
        triggered_rules: list[str],
    ) -> float:
        """
        Calculate compositional violation score.

        Uses a simple heuristic: if more rules are triggered,
        the compositional score is higher.
        """
        if not triggered_rules:
            return 0.0

        # Base score from prompt_injection category
        base_score = outputs.get("prompt_injection", 0.0)

        # Boost for multiple triggered rules
        rule_boost = min(len(triggered_rules) * 0.1, 0.3)

        # Combine
        return min(base_score + rule_boost, 1.0)

    def get_rule_by_id(self, rule_id: str) -> Optional[ComplianceRule]:
        """Get a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def enable_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable a rule."""
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = enabled

    def remove_rule(self, rule_id: str):
        """Remove a rule."""
        self.rules = [r for r in self.rules if r.id != rule_id]


if __name__ == "__main__":
    # Test the PGM reasoner
    reasoner = PGMReasoner()

    # Test case 1: High confidence injection
    outputs1 = {
        "prompt_injection": 0.95,
    }
    result1 = reasoner.reason(outputs1)
    print(f"Test 1: High confidence injection")
    print(f"  Violated: {result1.violated}")
    print(f"  Action: {result1.action}")
    print(f"  Triggered: {result1.triggered_rules}")

    # Test case 2: Compositional - roleplay + bypass
    outputs2 = {
        "roleplay_pattern": 0.8,
        "bypass_pattern": 0.7,
        "prompt_injection": 0.5,
    }
    result2 = reasoner.reason(outputs2)
    print(f"\nTest 2: Compositional roleplay + bypass")
    print(f"  Violated: {result2.violated}")
    print(f"  Action: {result2.action}")
    print(f"  Triggered: {result2.triggered_rules}")
    print(f"  Compositional score: {result2.compositional_score:.3f}")

    # Test case 3: Deterministic + Neural agreement
    outputs3 = {
        "deterministic_injection": 1.0,
        "neural_injection": 0.85,
        "prompt_injection": 0.85,
    }
    result3 = reasoner.reason(outputs3)
    print(f"\nTest 3: Deterministic + Neural agreement")
    print(f"  Violated: {result3.violated}")
    print(f"  Action: {result3.action}")
    print(f"  Triggered: {result3.triggered_rules}")
