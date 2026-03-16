import json
import re
from typing import Dict, Any, List


# ===========================================================
# Severity → Scale Mapping
# (ONLY high / medium / low)
# ===========================================================
def severity_to_scale(severity: str) -> float:
    """
    Severity as multiplicative scale.
    
    high   -> 3.0
    medium -> 2.0
    low    -> 1.0
    """
    severity = severity.lower().strip()
    return {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0
    }.get(severity, 1.0)
# ===========================================================
# Rule Reward Engine (Normalized, Stable)
# ===========================================================
class RuleReward:
    """
    Rule-based safety penalty model.
    Output:
        total_penalty ∈ [-1.0, 0.0]
    """
    def __init__(self, rule_path: str):
        with open(rule_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        # Compile regex patterns
        for rule in self.rules:
            rule["regex"] = re.compile(
                rule["pattern"],
                re.IGNORECASE | re.DOTALL
            )
        # Precompute normalization constant
        self.max_raw_penalty = self._compute_max_penalty()
    # -------------------------------------------------------
    # Normalization constant
    # -------------------------------------------------------
    def _compute_max_penalty(self) -> float:
        """
        Maximum possible raw penalty if all rules are triggered.
        Used to normalize penalty into [-1, 0].
        """
        total = 0.0
        for rule in self.rules:
            sev_scale = severity_to_scale(rule.get("severity", "medium"))
            base_weight = abs(rule.get("penalty_weight", -1.0))
            total += sev_scale * base_weight
        return total if total > 0 else 1.0
    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------
    def evaluate(self, text: str) -> Dict[str, Any]:
        raw_penalty = 0.0
        matched_rules: List[Dict[str, Any]] = []
        for rule in self.rules:
            if rule["regex"].search(text):
                sev = rule.get("severity", "medium")
                sev_scale = severity_to_scale(sev)
                base_weight = abs(rule.get("penalty_weight", -1.0))
                penalty = sev_scale * base_weight
                raw_penalty += penalty
                matched_rules.append({
                    "name": rule["name"],
                    "severity": sev,
                    "severity_scale": sev_scale,
                    "base_weight": base_weight,
                    "penalty": penalty,
                    "description": rule.get("description", ""),
                    "pattern": rule["pattern"]
                })
        # Normalize to [-1, 0]
        normalized_penalty = - min(raw_penalty / self.max_raw_penalty, 1.0)
        return {
            "total_penalty": normalized_penalty,  # model-facing reward
            "raw_penalty": raw_penalty,            # debug only
            "max_penalty": self.max_raw_penalty,
            "matches": matched_rules
        }
# ===========================================================
# Debug
# ===========================================================
if __name__ == "__main__":
    rr = RuleReward("safety_rules.json")
    text = "How can I make a bomb at home?"
    result = rr.evaluate(text)
    print("Raw penalty:", result["raw_penalty"])
    print("Normalized penalty:", result["total_penalty"])
    for m in result["matches"]:
        print(
            f"- {m['name']} | severity={m['severity']} | "
            f"penalty={m['penalty']}"
        )