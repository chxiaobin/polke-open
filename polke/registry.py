"""Loads the construct registry, detector contract, and lexicons from data/."""
from __future__ import annotations
import json
import functools
from pathlib import Path

DATA = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=1)
def constructs() -> dict:
    d = json.loads((DATA / "constructs.json").read_text(encoding="utf-8"))
    return {c["id"]: c for c in d["constructs"]}


@functools.lru_cache(maxsize=1)
def detectors() -> dict:
    d = json.loads((DATA / "detectors.json").read_text(encoding="utf-8"))
    return {c["construct_id"]: c for c in d["detectors"]}


@functools.lru_cache(maxsize=1)
def lexicons() -> dict:
    return json.loads((DATA / "lexicons.json").read_text(encoding="utf-8"))


def construct(cid: str) -> dict:
    return constructs()[cid]
