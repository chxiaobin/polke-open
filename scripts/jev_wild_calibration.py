#!/usr/bin/env python3
"""Wild-calibration study for the Jev annotator.

The operating point validated on the human-judged test pool does not
necessarily transfer to the wild (all 670 constructs asked of every
utterance: far lower base rate, so worse precision at the same threshold).
This script measures precision-vs-probability in the wild:

  sample   draw N (utterance, construct) claims from a jev_annotate.py
           output, stratified by probability band (capped per construct
           within a band), and write them as probe-style candidates into a
           record that `polke adjudicate` can serve. Judging is binary:
           fn = the construction IS in the utterance, reject = it is not.

  analyze  join the human verdicts with the sampled probabilities: per-band
           precision (Wilson 95% CI), the lowest probability band reaching a
           target precision, and per-band correction weights for
           probability-thresholded frequency counting.

  python scripts/jev_wild_calibration.py sample PILOT.jev.json TEXTS_DIR \
      -o examples/jev-wild-calibration [--n 200] [--seed 7]
  python scripts/jev_wild_calibration.py analyze examples/jev-wild-calibration
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BANDS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.50),
         (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]


def band_of(p):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= p < hi:
            return i
    return None


def cmd_sample(args):
    data = json.loads(Path(args.pilot).read_text())
    text_id = data["text_id"]
    src = Path(args.texts) / f"{text_id}.txt"
    lines = [l for l in src.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    starts, pos = [], 0
    for l in lines:
        starts.append(pos)
        pos += len(l) + 1
    text = "\n".join(lines) + "\n"

    by_band = defaultdict(list)
    for u in data["utterances"]:
        for cid, p in u["probs"].items():
            b = band_of(p)
            if b is not None:
                by_band[b].append((u["line"], cid, p))

    rng = random.Random(args.seed)
    per_band = args.n // len(BANDS)
    chosen = []
    for b in range(len(BANDS)):
        pool = by_band.get(b, [])
        rng.shuffle(pool)
        picked, seen_cid = [], defaultdict(int)
        for line, cid, p in pool:
            if seen_cid[cid] >= args.cap_per_construct:
                continue
            picked.append((line, cid, p))
            seen_cid[cid] += 1
            if len(picked) >= per_band:
                break
        chosen += picked
        print(f"band {BANDS[b]}: {len(pool)} claims, sampled {len(picked)}")

    candidates, manifest = [], {}
    for line, cid, p in sorted(chosen):
        lo = starts[line - 1]
        hi = lo + len(lines[line - 1])
        candidates.append({
            "text_id": text_id, "construct_id": cid,
            "span": {"start_char": lo, "end_char": hi,
                     "token_start": 0, "token_end": 0},
            "detector_type": "llm_probe",
            "detector_version": f"jev-wild@{data['model']}",
            "confidence": p, "model": data["model"],
            "models": [data["model"]],
            "evidence": {"quoted": ""}, "source": "probe",
        })
        manifest[f"probe|{text_id}|{cid}|{lo}|{hi}"] = {
            "line": line, "construct_id": cid, "p": p, "band": band_of(p)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"file": str(src), "text_id": text_id,
              "spacy_model": "n/a",
              "llm": {"ready": False, "model": data["model"],
                      "reason": "jev wild-calibration sample"},
              "text": text, "annotations": [],
              "probe": {"model": data["model"], "candidates": candidates}}
    (out / f"{text_id}.annotations.json").write_text(
        json.dumps(record, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps(
        {"pilot": str(args.pilot), "seed": args.seed, "bands": BANDS,
         "claims": manifest}, indent=1))
    print(f"\n{len(candidates)} claims -> {out}/")
    print(f"judge with: polke adjudicate {out} --by-line --port 8124")
    print("fn = construction IS present in the utterance; reject = it is not")
    return 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cmd_analyze(args):
    out = Path(args.dir)
    manifest = json.loads((out / "manifest.json").read_text())
    verdicts = json.loads((out / "verdicts.json").read_text())["verdicts"]
    bands = [tuple(b) for b in manifest["bands"]]
    tally = defaultdict(lambda: [0, 0])          # band -> [positive, judged]
    pending = 0
    for key, meta in manifest["claims"].items():
        v = verdicts.get(key)
        if v is None:
            pending += 1
            continue
        tally[meta["band"]][1] += 1
        tally[meta["band"]][0] += v["verdict"] == "fn"

    print(f"judged {sum(t[1] for t in tally.values())} / "
          f"{len(manifest['claims'])} claims ({pending} pending)\n")
    print("| band | judged | wild precision | 95% CI |")
    print("|---|---|---|---|")
    rows = []
    for b, (lo, hi) in enumerate(bands):
        k, n = tally.get(b, [0, 0])
        p, ci_lo, ci_hi = wilson(k, n)
        rows.append((lo, p, ci_lo, n))
        print(f"| {lo:.2f}-{min(hi, 1.0):.2f} | {n} | {p:.3f} "
              f"| {ci_lo:.3f}-{ci_hi:.3f} |")

    print(f"\nlowest band whose precision reaches {args.target}:")
    hit = [r for r in rows if r[1] >= args.target and r[3] > 0]
    if hit:
        print(f"  threshold >= {min(r[0] for r in hit):.2f} "
              "(use the band lower edge as the wild operating threshold)")
    else:
        print("  none - even the top band is below the target; "
              "consider per-construct thresholds or a higher-precision "
              "question design")
    print("\nPer-band precision doubles as the correction weight for "
          "probability-thresholded frequency counting (count x precision).")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sample")
    ps.add_argument("pilot", help="a *.jev.json from jev_annotate.py")
    ps.add_argument("texts", help="dir with the corpus .txt files")
    ps.add_argument("-o", "--out", required=True)
    ps.add_argument("--n", type=int, default=200)
    ps.add_argument("--cap-per-construct", type=int, default=2)
    ps.add_argument("--seed", type=int, default=7)
    ps.set_defaults(fn=cmd_sample)
    pa = sub.add_parser("analyze")
    pa.add_argument("dir")
    pa.add_argument("--target", type=float, default=0.8)
    pa.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
