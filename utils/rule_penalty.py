import json
import re
from typing import Dict, Any, List


def severity_to_scale(severity: str) -> float:
    severity = severity.lower().strip()
    return {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0
    }.get(severity, 1.0)


class RulePenalty:
    """
    Rule-based safety penalty model.
    Output: total_penalty ∈ [-1.0, 0.0]
        0.0 = 完全安全
       -1.0 = 命中所有规则
    """
    def __init__(self, rule_path: str):
        with open(rule_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        # 只为有 pattern 的规则编译 regex
        for rule in self.rules:
            if "pattern" in rule:
                rule["regex"] = re.compile(
                    rule["pattern"],
                    re.IGNORECASE | re.DOTALL
                )
        self.max_raw_penalty = self._compute_max_penalty()

    def _compute_max_penalty(self) -> float:
        total = 0.0
        for rule in self.rules:
            sev_scale = severity_to_scale(rule.get("severity", "medium"))
            base_weight = abs(rule.get("penalty_weight", 1.0))  # ← 改成 1.0 默认
            total += sev_scale * base_weight
        return total if total > 0 else 1.0

    def _is_matched(self, rule: Dict[str, Any], text: str) -> bool:
        """
        匹配规则: pattern 命中 OR (keywords 全在 AND keywords_any 任一在)
        """
        text_lower = text.lower()

        # 方式 1: 原正则（保留兼容）
        if "regex" in rule and rule["regex"].search(text):
            return True

        # 方式 2: keywords + keywords_any（AND 语义）
        # keywords: 必须全部出现
        has_required = True
        if "keywords" in rule and rule["keywords"]:
            has_required = all(kw.lower() in text_lower for kw in rule["keywords"])

        # keywords_any: 至少一个出现
        has_any = True
        if "keywords_any" in rule and rule["keywords_any"]:
            has_any = any(kw.lower() in text_lower for kw in rule["keywords_any"])

        if has_required and has_any:
            return True
        return False

    def evaluate(self, text: str) -> Dict[str, Any]:
        raw_penalty = 0.0
        matched_rules: List[Dict[str, Any]] = []
        for rule in self.rules:
            # ✅ 关键修复: 用 _is_matched 而非只匹配 regex
            if self._is_matched(rule, text):
                sev = rule.get("severity", "medium")
                sev_scale = severity_to_scale(sev)
                base_weight = abs(rule.get("penalty_weight", 1.0))
                penalty = sev_scale * base_weight
                raw_penalty += penalty
                matched_rules.append({
                    "name": rule["name"],
                    "severity": sev,
                    "severity_scale": sev_scale,
                    "base_weight": base_weight,
                    "penalty": penalty,
                    "description": rule.get("description", ""),
                    "pattern": rule.get("pattern", ""),
                    "keywords": rule.get("keywords", []),
                    "keywords_any": rule.get("keywords_any", []),
                })
        normalized_penalty = -min(raw_penalty / self.max_raw_penalty, 1.0)
        return {
            "total_penalty": normalized_penalty,
            "raw_penalty": raw_penalty,
            "max_penalty": self.max_raw_penalty,
            "matches": matched_rules,
        }


if __name__ == "__main__":
    rp = RulePenalty("safety_rules.json")
    for text in [
        "Hello, how are you?",
        "How can I make a bomb at home?",
        "How to hack into an account?",
        "I want to kill myself.",
        "How to cook meth?",
    ]:
        result = rp.evaluate(text)
        names = [m["name"] for m in result["matches"]]
        print(f"[{result['total_penalty']:+.4f}]  {text}")
        print(f"           matched: {names[:2]}")