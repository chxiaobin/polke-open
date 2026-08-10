#!/usr/bin/env python3
"""Build per-construct review batches from annotation records, and merge
judged batches back into verdicts.json.

Workflow (model- or human-adjudicated, the format is the same):

  build   sample up to N system annotations and N probe candidates per
          construct (seeded), grouped by category, with the sentence
          context — one JSON file per category under --out
  merge   fold a judged batch (each instance given "verdict", optionally
          "corrected_id") into verdicts.json next to the records; existing
          verdicts are NEVER overwritten (first judgment wins), so human
          verdicts always survive model ones

Verdict values: sys instances tp | fp | wrong (+corrected_id) | span;
probe instances fn | reject. A "judge" field records who judged.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _records(path: Path):
    for p in sorted(path.glob("*.annotations.json")):
        yield json.loads(p.read_text(encoding="utf-8"))


def _sentence_of(text: str, start: int, end: int) -> str:
    lo = text.rfind("\n", 0, start) + 1
    hi = text.find("\n", end)
    return text[lo:hi if hi >= 0 else len(text)].strip()


def cmd_build(args) -> int:
    src = Path(args.path)
    rng = random.Random(args.seed)
    by_construct = defaultdict(lambda: {"sys": [], "probe": []})
    for rec in _records(src):
        text, tid = rec.get("text", ""), rec.get("text_id", "")
        for a in rec.get("annotations", []):
            s, e = a["span"]["start_char"], a["span"]["end_char"]
            by_construct[a["construct_id"]]["sys"].append({
                "key": f"sys|{tid}|{a['construct_id']}|{s}|{e}",
                "sentence": _sentence_of(text, s, e),
                "matched": text[s:e].strip()[:80],
                "tier": a.get("detector_type", ""),
                "rationale": str((a.get("evidence") or {}).get(
                    "rationale", ""))[:200],
            })
        for c in (rec.get("probe") or {}).get("candidates", []):
            s, e = c["span"]["start_char"], c["span"]["end_char"]
            by_construct[c["construct_id"]]["probe"].append({
                "key": f"probe|{tid}|{c['construct_id']}|{s}|{e}",
                "sentence": _sentence_of(text, s, e),
                "evidence": str((c.get("evidence") or {}).get(
                    "quoted", ""))[:80],
                "agree": len(c.get("models", [])),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from polke.registry import constructs
    inv = constructs()
    by_cat = defaultdict(dict)
    for cid, groups in sorted(by_construct.items()):
        c = inv.get(cid, {})
        sys_sample = rng.sample(groups["sys"],
                                min(args.per_construct, len(groups["sys"])))
        probe_sorted = sorted(groups["probe"],
                              key=lambda x: -x["agree"])   # agreement first
        by_cat[cid.split("-")[0]][cid] = {
            "name": c.get("name", ""),
            "definition": (c.get("definition_raw") or "")[:200],
            "example": c.get("example", ""),
            "n_sys": len(groups["sys"]), "n_probe": len(groups["probe"]),
            "sys": sys_sample,
            "probe": probe_sorted[:args.per_construct],
        }
    for cat, constructs_ in sorted(by_cat.items()):
        (out / f"{cat}.json").write_text(
            json.dumps(constructs_, ensure_ascii=False, indent=1),
            encoding="utf-8")
    n = sum(len(v["sys"]) + len(v["probe"])
            for cats in by_cat.values() for v in cats.values())
    print(f"{len(by_cat)} category files, "
          f"{sum(len(c) for c in by_cat.values())} constructs, "
          f"{n} instances to judge -> {out}/")
    return 0


def cmd_merge(args) -> int:
    vf = Path(args.verdicts)
    store = {}
    if vf.exists():
        data = json.loads(vf.read_text(encoding="utf-8"))
        store = data.get("verdicts", data)
    added = skipped = 0
    for batch in args.batches:
        judged = json.loads(Path(batch).read_text(encoding="utf-8"))
        for key, v in judged.items():
            if not isinstance(v, dict) or "verdict" not in v:
                continue
            if key in store:
                skipped += 1               # existing (e.g. human) verdict wins
                continue
            store[key] = {"verdict": v["verdict"],
                          "corrected_id": v.get("corrected_id"),
                          "judge": v.get("judge", "model")}
            added += 1
    vf.write_text(json.dumps({"tool": "adjudication_batches",
                              "verdicts": store},
                             ensure_ascii=False, indent=1) + "\n",
                  encoding="utf-8")
    print(f"merged: +{added} verdicts, {skipped} already judged, "
          f"total {len(store)} -> {vf}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build")
    pb.add_argument("path", help="annotations folder")
    pb.add_argument("--out", required=True)
    pb.add_argument("--per-construct", type=int, default=5)
    pb.add_argument("--seed", type=int, default=42)
    pb.set_defaults(fn=cmd_build)
    pm = sub.add_parser("merge")
    pm.add_argument("verdicts", help="verdicts.json to update")
    pm.add_argument("batches", nargs="+",
                    help="judged batch files: {key: {verdict, ...}}")
    pm.set_defaults(fn=cmd_merge)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
