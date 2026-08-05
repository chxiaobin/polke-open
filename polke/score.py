"""Verification scoring (polke score): P/R/F1 per construct from verdicts.

Input: annotation records (with probe candidates) + the verdicts.json the
viewer's adjudication UI exports. Verdict keys are
``<kind>|<text_id>|<construct_id>|<start_char>|<end_char>`` with kind
``sys`` (a system annotation) or ``probe`` (a probe candidate); values are
{"verdict": ..., "corrected_id": ...}.

Counting rules
  sys  tp     -> TP
  sys  span   -> TP (right phenomenon, wrong extent) + boundary tally
  sys  fp     -> FP
  sys  wrong  -> FP for the marked id, FN for corrected_id
  probe fn    -> FN (a confirmed miss)
  probe reject-> nothing
  unadjudicated items are excluded from the metrics and reported as pending.

precision = TP/(TP+FP); recall = TP/(TP+FN); support = TP+FN (gold count).
Recall is relative to the pooled candidate set (system + probe): misses that
neither surfaced are invisible, so treat recall/F1 as upper bounds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .registry import constructs

SYS_VERDICTS = {"tp", "fp", "wrong", "span"}
PROBE_VERDICTS = {"fn", "reject"}


def key_of(kind: str, text_id: str, cid: str, start: int, end: int) -> str:
    return f"{kind}|{text_id}|{cid}|{start}|{end}"


def load_verdicts(path: Path) -> Dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    v = data.get("verdicts", data)
    if not isinstance(v, dict):
        raise ValueError(f"{path} is not a verdicts file")
    return {k: (x if isinstance(x, dict) else {"verdict": x})
            for k, x in v.items()}


def _iter_records(path: Path) -> List[dict]:
    from .probe import _files, _load
    out = []
    for f in _files(Path(path)):
        out.extend(r for r in _load(f)
                   if isinstance(r, dict) and "annotations" in r)
    if not out:
        raise ValueError(f"no annotation records under {path}")
    return out


def compute(records: List[dict], verdicts: Dict[str, dict]) -> dict:
    counts: Dict[str, dict] = {}

    def c(cid):
        return counts.setdefault(
            cid, {"tp": 0, "fp": 0, "fn": 0, "span": 0,
                  "pending_sys": 0, "pending_probe": 0})

    used = set()
    for rec in records:
        tid = rec.get("text_id", "")
        for a in rec.get("annotations") or []:
            cid = a["construct_id"]
            sp = a.get("span") or {}
            k = key_of("sys", tid, cid,
                       sp.get("start_char", 0), sp.get("end_char", 0))
            v = verdicts.get(k)
            used.add(k)
            verdict = (v or {}).get("verdict")
            if verdict in ("tp", "span"):
                c(cid)["tp"] += 1
                if verdict == "span":
                    c(cid)["span"] += 1
            elif verdict == "fp":
                c(cid)["fp"] += 1
            elif verdict == "wrong":
                c(cid)["fp"] += 1
                corr = (v or {}).get("corrected_id", "")
                if corr in constructs():
                    c(corr)["fn"] += 1
            else:
                c(cid)["pending_sys"] += 1
        for p in (rec.get("probe") or {}).get("candidates") or []:
            cid = p["construct_id"]
            sp = p.get("span") or {}
            k = key_of("probe", tid, cid,
                       sp.get("start_char", 0), sp.get("end_char", 0))
            v = verdicts.get(k)
            used.add(k)
            verdict = (v or {}).get("verdict")
            if verdict == "fn":
                c(cid)["fn"] += 1
            elif verdict == "reject":
                pass
            else:
                c(cid)["pending_probe"] += 1

    rows = []
    inv = constructs()
    for cid in sorted(counts):
        n = counts[cid]
        tp, fp, fn = n["tp"], n["fp"], n["fn"]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = (2 * p * r / (p + r)
              if p is not None and r is not None and (p + r) else None)
        meta = inv.get(cid, {})
        rows.append({
            "construct_id": cid,
            "category": meta.get("category", cid.split("-")[0]),
            "tier": meta.get("detector_type", ""),
            **n,
            "precision": p, "recall": r, "f1": f1,
            "support": tp + fn,
        })
    totals = {k: sum(r[k] for r in rows)
              for k in ("tp", "fp", "fn", "span",
                        "pending_sys", "pending_probe")}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    totals["precision"] = tp / (tp + fp) if tp + fp else None
    totals["recall"] = tp / (tp + fn) if tp + fn else None
    pr, rc = totals["precision"], totals["recall"]
    totals["f1"] = (2 * pr * rc / (pr + rc)
                    if pr is not None and rc is not None and (pr + rc)
                    else None)
    stale = sorted(k for k in verdicts if k not in used)
    return {"constructs": rows, "totals": totals, "stale_verdicts": stale}


def _fmt(x: Optional[float]) -> str:
    return "  —  " if x is None else f"{x:.3f}"


def report(result: dict, min_support: int = 0) -> str:
    lines = []
    rows = [r for r in result["constructs"]
            if r["support"] >= min_support or r["fp"] or r["pending_sys"]
            or r["pending_probe"]]
    if rows:
        lines.append(f"{'construct':<10} {'tier':<18} "
                     f"{'TP':>4} {'FP':>4} {'FN':>4} {'span':>4}  "
                     f"{'P':>5} {'R':>5} {'F1':>5}  {'supp':>4}  pending")
    cur = None
    for r in rows:
        if r["category"] != cur:
            cur = r["category"]
            lines.append(f"-- {cur}")
        pend = r["pending_sys"] + r["pending_probe"]
        lines.append(
            f"{r['construct_id']:<10} {r['tier']:<18} "
            f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['span']:>4}  "
            f"{_fmt(r['precision'])} {_fmt(r['recall'])} {_fmt(r['f1'])}  "
            f"{r['support']:>4}  {pend or ''}")
    t = result["totals"]
    lines.append(
        f"\n{'TOTAL':<10} {'':<18} {t['tp']:>4} {t['fp']:>4} {t['fn']:>4} "
        f"{t['span']:>4}  {_fmt(t['precision'])} {_fmt(t['recall'])} "
        f"{_fmt(t['f1'])}")
    pend = t["pending_sys"] + t["pending_probe"]
    if pend:
        lines.append(f"pending: {t['pending_sys']} system annotation(s) and "
                     f"{t['pending_probe']} probe candidate(s) not yet "
                     "adjudicated (excluded from the metrics)")
    if result["stale_verdicts"]:
        lines.append(f"note: {len(result['stale_verdicts'])} verdict(s) do "
                     "not match any current annotation (stale keys)")
    lines.append("\nRecall/F1 are relative to the pooled candidates "
                 "(system + probe): treat them as upper bounds.")
    return "\n".join(lines)
