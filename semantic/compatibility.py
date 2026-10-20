"""§10 Computational Compatibility Check — pure deterministic function."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CompatibilityRequest:
    model_id: str = ""
    model_revision: str = ""
    tokenizer_id: str = ""
    prefix_hash: str = ""
    token_sequence_hash: str = ""
    tensor_shape: str = ""  # e.g. "[1, 2048, 4096]"
    dtype: str = "float16"
    namespace: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool = False
    reason: str = ""
    matched_fields: List[str] = None  # type: ignore
    mismatched_fields: List[str] = None  # type: ignore

    def __post_init__(self):
        object.__setattr__(self, 'matched_fields', self.matched_fields or [])
        object.__setattr__(self, 'mismatched_fields', self.mismatched_fields or [])

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "reason": self.reason,
            "matched_fields": self.matched_fields,
            "mismatched_fields": self.mismatched_fields,
        }


class ComputationalCompatibilityChecker:
    """§10: Semantic relevance alone is NOT sufficient for reuse."""
    # Fields required to match for exact reuse (§10 list)
    REQUIRED_FIELDS = [
        "model_id", "tokenizer_id", "prefix_hash", "tensor_shape",
    ]

    def check(self, current: CompatibilityRequest, candidate: CompatibilityRequest) -> CompatibilityResult:
        matched = []
        mismatched = []
        for f in self.REQUIRED_FIELDS:
            cur = getattr(current, f, None)
            cand = getattr(candidate, f, None)
            if cur is None or cand is None:
                # Missing field → incompatible (strict)
                mismatched.append(f + "_missing")
            elif cur == cand:
                matched.append(f)
            else:
                mismatched.append(f)
        compatible = len(mismatched) == 0
        reason = "all_compatible" if compatible else f"mismatch: {', '.join(mismatched)}"
        return CompatibilityResult(
            compatible=compatible,
            reason=reason,
            matched_fields=matched,
            mismatched_fields=mismatched,
        )
