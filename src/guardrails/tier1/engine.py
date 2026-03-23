"""
Tier 1: Policy Reasoning Engine

Rule-based enforcement with configurable thresholds and action routing.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import time


class Action(Enum):
    """Possible actions for a detected violation."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    MASK = "mask"
    ESCALATE = "escalate"
    LOG = "log"


@dataclass
class PolicyRule:
    """A single policy rule."""
    id: str
    name: str
    description: str
    category: str
    condition: Callable[[dict], bool]  # Evaluated on signal dict
    action: Action
    threshold: float = 0.5
    severity: str = "medium"  # low, medium, high, critical
    enabled: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class Signal:
    """A signal from a detector (Tier 2 or Tier 3)."""
    source: str  # e.g., "tier2_deterministic", "tier3_lora"
    category: str  # e.g., "prompt_injection", "pii"
    score: float  # 0.0 - 1.0
    confidence: float
    blocked: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Final policy decision."""
    action: Action
    reason: str
    confidence: float
    triggered_rules: list[str]
    signals: list[Signal]
    total_latency_ms: float
    metadata: dict = field(default_factory=dict)


class ActionRouter:
    """
    Routes detected violations to appropriate actions.

    Maps (category, severity, score) -> Action
    """

    def __init__(self):
        # Default routing rules
        self.routing_rules = {
            # (category, severity): {score_range: action}
            ("prompt_injection", "critical"): {
                (0.7, 1.0): Action.BLOCK,
                (0.5, 0.7): Action.ESCALATE,
                (0.0, 0.5): Action.WARN,
            },
            ("prompt_injection", "high"): {
                (0.8, 1.0): Action.BLOCK,
                (0.6, 0.8): Action.WARN,
                (0.0, 0.6): Action.LOG,
            },
            ("prompt_injection", "medium"): {
                (0.9, 1.0): Action.BLOCK,
                (0.7, 0.9): Action.WARN,
                (0.0, 0.7): Action.LOG,
            },
            ("pii", "high"): {
                (0.5, 1.0): Action.MASK,
                (0.0, 0.5): Action.LOG,
            },
            ("pii", "medium"): {
                (0.7, 1.0): Action.MASK,
                (0.0, 0.7): Action.LOG,
            },
        }

        # Default actions for unknown combinations
        self.default_high = Action.WARN
        self.default_medium = Action.LOG
        self.default_low = Action.ALLOW

    def route(
        self,
        category: str,
        severity: str,
        score: float,
    ) -> Action:
        """Determine action for a given category, severity, and score."""
        key = (category, severity)

        if key in self.routing_rules:
            for (low, high), action in self.routing_rules[key].items():
                if low <= score < high:
                    return action

        # Default routing based on severity
        if severity == "critical":
            return Action.BLOCK if score > 0.5 else Action.WARN
        elif severity == "high":
            return Action.WARN if score > 0.5 else Action.LOG
        elif severity == "medium":
            return Action.LOG if score > 0.5 else Action.ALLOW
        else:
            return Action.ALLOW

    def add_rule(
        self,
        category: str,
        severity: str,
        score_ranges: dict[tuple[float, float], Action],
    ):
        """Add a custom routing rule."""
        self.routing_rules[(category, severity)] = score_ranges


class PolicyReasoningEngine:
    """
    Central policy reasoning engine.

    Aggregates signals from all tiers and applies policy rules to determine actions.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.rules: list[PolicyRule] = []
        self.router = ActionRouter()
        self.threshold_overrides: dict[str, float] = {}
        self.category_enabled: dict[str, bool] = {}

        if config_path:
            self.load_config(config_path)

        # Initialize default rules
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default policy rules."""
        # Prompt injection rules
        self.rules.extend([
            PolicyRule(
                id="pi_block_high",
                name="Block High-Confidence Injection",
                description="Block when injection confidence is high",
                category="prompt_injection",
                condition=lambda s: s.get("injection_score", 0) > 0.8,
                action=Action.BLOCK,
                severity="high",
            ),
            PolicyRule(
                id="pi_warn_medium",
                name="Warn Medium-Confidence Injection",
                description="Warn when injection confidence is medium",
                category="prompt_injection",
                condition=lambda s: 0.5 < s.get("injection_score", 0) <= 0.8,
                action=Action.WARN,
                severity="medium",
            ),
            PolicyRule(
                id="pi_deterministic_block",
                name="Block Deterministic Matches",
                description="Block when deterministic filter matches",
                category="prompt_injection",
                condition=lambda s: s.get("deterministic_blocked", False),
                action=Action.BLOCK,
                severity="critical",
            ),
        ])

    def load_config(self, config_path: Path):
        """Load configuration from file."""
        with open(config_path) as f:
            config = json.load(f)

        # Load threshold overrides
        if "thresholds" in config:
            self.threshold_overrides = config["thresholds"]

        # Load category toggles
        if "categories" in config:
            self.category_enabled = config["categories"]

    def set_threshold(self, category: str, threshold: float):
        """Set threshold override for a category."""
        self.threshold_overrides[category] = threshold

    def enable_category(self, category: str, enabled: bool = True):
        """Enable or disable a category."""
        self.category_enabled[category] = enabled

    def aggregate_signals(self, signals: list[Signal]) -> dict:
        """Aggregate signals from all detectors."""
        aggregated = {
            "injection_score": 0.0,
            "deterministic_blocked": False,
            "neural_score": 0.0,
            "lora_score": 0.0,
            "categories_detected": set(),
            "sources": [],
        }

        for signal in signals:
            aggregated["sources"].append(signal.source)

            if signal.category == "prompt_injection":
                if "deterministic" in signal.source:
                    aggregated["deterministic_blocked"] = signal.blocked
                    if signal.blocked:
                        aggregated["injection_score"] = max(
                            aggregated["injection_score"], 1.0
                        )
                elif "neural" in signal.source:
                    aggregated["neural_score"] = signal.score
                    aggregated["injection_score"] = max(
                        aggregated["injection_score"], signal.score
                    )
                elif "lora" in signal.source or "adapter" in signal.source:
                    aggregated["lora_score"] = signal.score
                    aggregated["injection_score"] = max(
                        aggregated["injection_score"], signal.score
                    )

            aggregated["categories_detected"].add(signal.category)

        aggregated["categories_detected"] = list(aggregated["categories_detected"])
        return aggregated

    def evaluate(self, signals: list[Signal]) -> PolicyDecision:
        """
        Evaluate signals and determine action.

        Args:
            signals: List of signals from all detectors

        Returns:
            PolicyDecision with action and reasoning
        """
        start_time = time.perf_counter()

        # Aggregate signals
        aggregated = self.aggregate_signals(signals)

        # Evaluate rules
        triggered_rules = []
        actions = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            if self.category_enabled.get(rule.category) == False:
                continue

            if rule.condition(aggregated):
                triggered_rules.append(rule.id)
                actions.append((rule.action, rule.severity, aggregated["injection_score"]))

        # Determine final action
        if not actions:
            final_action = Action.ALLOW
            reason = "No policy violations detected"
        else:
            # Priority: BLOCK > ESCALATE > MASK > WARN > LOG > ALLOW
            action_priority = {
                Action.BLOCK: 5,
                Action.ESCALATE: 4,
                Action.MASK: 3,
                Action.WARN: 2,
                Action.LOG: 1,
                Action.ALLOW: 0,
            }

            # Get highest priority action
            actions.sort(key=lambda x: action_priority[x[0]], reverse=True)
            final_action = actions[0][0]

            # Build reason
            severity = actions[0][1]
            score = actions[0][2]
            reason = f"Triggered rules: {triggered_rules}. Score: {score:.2f}, Severity: {severity}"

        total_latency = (time.perf_counter() - start_time) * 1000

        return PolicyDecision(
            action=final_action,
            reason=reason,
            confidence=aggregated["injection_score"],
            triggered_rules=triggered_rules,
            signals=signals,
            total_latency_ms=total_latency,
            metadata={
                "aggregated": aggregated,
            }
        )

    def add_rule(self, rule: PolicyRule):
        """Add a custom policy rule."""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str):
        """Remove a rule by ID."""
        self.rules = [r for r in self.rules if r.id != rule_id]


class QuickPolicyIngestion:
    """
    Quick policy ingestion for rapid threat response.

    Allows adding rules from natural language descriptions.
    """

    def __init__(self, engine: PolicyReasoningEngine):
        self.engine = engine
        self.ingested_policies = []

    def ingest_quick_description(
        self,
        description: str,
        action: Action = Action.BLOCK,
        severity: str = "high",
    ) -> PolicyRule:
        """
        Ingest a quick policy description.

        Example:
            "Block any request mentioning 'system prompt'"
        """
        # Extract keywords from description
        keywords = self._extract_keywords(description)

        # Create rule
        rule_id = f"quick_{len(self.ingested_policies)}"

        def condition(signals: dict, kw=keywords) -> bool:
            # Check if any keyword appears in the text
            text = signals.get("text", "")
            return any(kw.lower() in text.lower() for kw in kw)

        rule = PolicyRule(
            id=rule_id,
            name=f"Quick Policy: {description[:50]}",
            description=description,
            category="quick_policy",
            condition=condition,
            action=action,
            severity=severity,
        )

        self.engine.add_rule(rule)
        self.ingested_policies.append(rule)

        return rule

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from policy description."""
        # Simple extraction - look for quoted strings
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        keywords = [q[0] or q[1] for q in quoted]

        # Also extract important words
        important_words = [
            "system prompt", "jailbreak", "bypass", "ignore",
            "disregard", "override", "disable",
        ]

        for word in important_words:
            if word in text.lower():
                keywords.append(word)

        return keywords


if __name__ == "__main__":
    # Test the policy engine
    engine = PolicyReasoningEngine()

    # Simulate signals
    signals = [
        Signal(
            source="tier2_deterministic",
            category="prompt_injection",
            score=0.0,
            confidence=1.0,
            blocked=False,
            reason="No patterns matched",
        ),
        Signal(
            source="tier3_lora",
            category="prompt_injection",
            score=0.75,
            confidence=0.8,
            blocked=False,
            reason="LoRA classifier prediction",
        ),
    ]

    decision = engine.evaluate(signals)
    print(f"Action: {decision.action.value}")
    print(f"Reason: {decision.reason}")
    print(f"Triggered rules: {decision.triggered_rules}")

    # Test quick ingestion
    quick = QuickPolicyIngestion(engine)
    rule = quick.ingest_quick_description(
        "Block any request mentioning 'system prompt'",
        action=Action.BLOCK,
    )
    print(f"\nIngested rule: {rule.id}")
