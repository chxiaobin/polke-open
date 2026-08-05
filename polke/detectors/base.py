"""Detector interface and a process-wide registry. No spaCy import here."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Detector(ABC):
    construct_ids: List[str] = []  # one detector may serve several sibling constructs
    detector_type: str = "unknown"
    version: str = "0.1"

    @abstractmethod
    def match(self, doc, text_id: str = "doc") -> list:
        """Return a list[Annotation] for the given spaCy doc."""
        raise NotImplementedError


_REGISTRY = {}  # construct_id -> Detector


def register(det: "Detector"):
    for cid in det.construct_ids:
        _REGISTRY[cid] = det
    return det


def registry() -> dict:
    return dict(_REGISTRY)


def implemented_ids() -> set:
    return set(_REGISTRY.keys())


def unique_detectors() -> list:
    seen, out = set(), []
    for det in _REGISTRY.values():
        if id(det) not in seen:
            seen.add(id(det))
            out.append(det)
    return out
