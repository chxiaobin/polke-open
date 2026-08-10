#!/usr/bin/env python3
"""Build a human-manageable verification package from annotation records.

Samples per construct: up to --per-construct system annotations (seeded
random) and up to --probe-per-construct probe candidates (two-model
agreement first), trimmed to a global --budget. Writes FILTERED copies of
the .annotations.json records (same text, only the sampled annotations /
probe candidates) into --out, so `polke adjudicate <out> --by-line` shows
the human exactly the sampled instances and nothing else. Verdicts land in
<out>/verdicts.json as usual.

The sampling manifest (which keys were chosen and why) is written to
<out>/package-manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _key(kind, tid, cid, span):
    return f"{kind}|{tid}|{cid}|{span['start_char']}|{span['end_char']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("path", help="annotations folder (from polke annotate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-construct", type=int, default=3)
    ap.add_argument("--probe-per-construct", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    src = Path(args.path)
    rng = random.Random(args.seed)
    records = {}
    sys_pool = defaultdict(list)    # cid -> [(key, rec_name, ann)]
    probe_pool = defaultdict(list)  # cid -> [(key, rec_name, cand, agree)]
    for p in sorted(src.glob("*.annotations.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        records[p.name] = rec
        tid = rec.get("text_id", "")
        for a in rec.get("annotations", []):
            cid = a["construct_id"]
            sys_pool[cid].append((_key("sys", tid, cid, a["span"]), p.name, a))
        for c in (rec.get("probe") or {}).get("candidates", []):
            cid = c["construct_id"]
            probe_pool[cid].append((_key("probe", tid, cid, c["span"]),
                                    p.name, c, len(c.get("models", []))))

    chosen_sys, chosen_probe = {}, {}
    for cid in sorted(set(sys_pool) | set(probe_pool)):
        pool = sys_pool.get(cid, [])
        chosen_sys[cid] = rng.sample(pool, min(args.per_construct, len(pool)))
        cand = sorted(probe_pool.get(cid, []),
                      key=lambda x: (-x[3], rng.random()))
        chosen_probe[cid] = cand[:args.probe_per_construct]

    def total():
        return sum(map(len, chosen_sys.values())) + \
            sum(map(len, chosen_probe.values()))

    while total() > args.budget:
        cid = max(chosen_sys,
                  key=lambda c: len(chosen_sys[c]) + len(chosen_probe.get(c, [])))
        if len(chosen_probe.get(cid, [])) >= len(chosen_sys[cid]) \
                and chosen_probe.get(cid):
            chosen_probe[cid].pop()
        elif chosen_sys[cid]:
            chosen_sys[cid].pop()
        else:
            break

    keep_ann = defaultdict(list)    # rec_name -> [ann]
    keep_probe = defaultdict(list)  # rec_name -> [cand]
    manifest = {"sys": {}, "probe": {}}
    for cid, items in chosen_sys.items():
        manifest["sys"][cid] = [k for k, _, _ in items]
        for _k, name, a in items:
            keep_ann[name].append(a)
    for cid, items in chosen_probe.items():
        manifest["probe"][cid] = [k for k, _, _, _ in items]
        for _k, name, c, _n in items:
            keep_probe[name].append(c)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for name, rec in records.items():
        anns, cands = keep_ann.get(name, []), keep_probe.get(name, [])
        if not anns and not cands:
            continue
        slim = dict(rec)
        slim["annotations"] = anns
        pr = dict(rec.get("probe") or {})
        pr["candidates"] = cands
        slim["probe"] = pr
        (out / name).write_text(json.dumps(slim, ensure_ascii=False),
                                encoding="utf-8")
        n_files += 1

    (out / "package-manifest.json").write_text(
        json.dumps({"source": str(src), "seed": args.seed,
                    "per_construct": args.per_construct,
                    "probe_per_construct": args.probe_per_construct,
                    "budget": args.budget, **manifest},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    n_sys = sum(map(len, chosen_sys.values()))
    n_probe = sum(map(len, chosen_probe.values()))
    print(f"{n_files} record files, {len(chosen_sys)} constructs, "
          f"{n_sys} system + {n_probe} probe = {n_sys + n_probe} judgments "
          f"-> {out}/")
    print(f"serve with: polke adjudicate {out} --by-line --port 8123")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
