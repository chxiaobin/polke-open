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


UNRATED = "unrated"
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


@functools.lru_cache(maxsize=1)
def cefr_levels() -> dict:
    """construct id -> {"level": "B1" | "unrated", "note": "..."} from
    data/cefr_levels.csv (A1–C2, or NR for the non-level-bearing vernacular
    and dysfluency strata). Constructs absent from the file are unrated."""
    import csv
    out: dict = {}
    path = DATA / "cefr_levels.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("id") or "").strip()
            lv = (row.get("level") or "").strip().upper()
            if cid:
                out[cid] = {"level": lv if lv in CEFR_LEVELS else UNRATED,
                            "note": (row.get("note") or "").strip()}
    return out


def cefr_level(cid: str) -> str:
    return cefr_levels().get(cid, {}).get("level", UNRATED)
