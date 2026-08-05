"""Annotation schema (USE-mode).

Attempt-mode fields (status, target_hypothesis) are intentionally omitted:
input text is assumed well-formed because grammatical error correction (GEC)
is applied upstream, before construction detection. When attempt-mode is added
later, extend Annotation with `status` and `target_hypothesis`.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Span:
    start_char: int
    end_char: int
    token_start: int
    token_end: int


@dataclass
class Annotation:
    text_id: str
    construct_id: str
    span: Span
    detector_type: str
    detector_version: str
    confidence: float = 1.0
    model: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    source: str = "system"  # system | human

    def to_dict(self) -> dict:
        return asdict(self)
