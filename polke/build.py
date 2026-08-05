"""Single entrypoint that constructs and registers EVERY detector.

The coding agent grows this function: right now only the three reference
detectors are wired. Add the remaining constructs here, tier by tier, following
polke/detectors/reference.py. Each construct in data/detectors.json
must end up registered (see detectors.base.register).
"""
from __future__ import annotations
from .detectors.reference import build_reference_detectors
from .detectors.base import register, unique_detectors
from .detectors.catalog import build_categories


def build_all(nlp, llm_client=None):
    # References first; category detectors are registered afterwards and may
    # intentionally override a reference id (e.g. the PAS-01 paradigm router).
    build_reference_detectors(nlp, llm_client=llm_client)
    for det in build_categories(nlp, client=llm_client):
        register(det)
    return unique_detectors()
