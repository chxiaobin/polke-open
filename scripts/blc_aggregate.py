#!/usr/bin/env python3
"""Aggregate BLC study annotations into per-construct analysis tables.

Combines whatever inputs exist (safe to run mid-annotation for interim
numbers):

* SUBSAMPLE (blc_sample.py + blc_run.py output): estimates for all 670
  constructs. Frequency and presence rate come from the uniform CORE
  component only (design-unbiased); speaker coverage uses core+topup, with
  rarefaction-standardised coverage at --std-n utterances per speaker.
* OFFLINE full-corpus run (--offline): exact presence/instance counts and
  exact speaker coverage for the offline-tier constructs.
* Speaker metadata TSV (--speakers, the BNC download's
  bnc2014spoken-speakerdata.tsv): education/social-grade strata coverage.
* Test-phase reliability (--scores): per-construct precision/recall joined
  as qualifiers.

Writes <out>/constructs.csv and <out>/summary.md.

  python scripts/blc_aggregate.py examples/blc-run \
      --offline examples/spokenBNC-corpus/texts \
      --speakers path/to/bnc2014spoken-speakerdata.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# rule-stage gaps (docs/detector_gaps.md): the pipeline cannot propose
# these on at least some realisations - frequencies are lower bounds
RULE_GAPS = {"ASC-06", "ASC-29", "CLS-16", "FOC-06", "FUT-13", "IMP-04",
             "IMP-07", "MOD-22", "MOD-28", "NEG-03", "NFV-15", "NFV-19",
             "NFV-31", "PAS-18", "VCP-07", "VCP-35", "VSP-09", "VTA-10"}


def line_index(text):
    starts, pos, lines = [], 0, text.splitlines()
    for l in lines:
        starts.append(pos)
        pos += len(l) + 1
    return lines, starts


def line_of(starts, ch):
    lo = 0
    for i, s in enumerate(starts):
        if s <= ch:
            lo = i
    return lo


def per_line_constructs(rec):
    """line index -> {construct_id: instance count}."""
    _, starts = line_index(rec["text"])
    out = defaultdict(lambda: defaultdict(int))
    for a in rec["annotations"]:
        out[line_of(starts, a["span"]["start_char"])][a["construct_id"]] += 1
    return out


def rarefied(n, k, m):
    """P(>=1 hit in m draws without replacement) given k hits in n."""
    if n < m or k == 0:
        return 1.0 if k > 0 and n < m else (0.0 if k == 0 else 1.0)
    if k >= n - m + 1:
        return 1.0
    # C(n-k, m) / C(n, m) in log space
    num = sum(math.log(n - k - i) for i in range(m))
    den = sum(math.log(n - i) for i in range(m))
    return 1.0 - math.exp(num - den)


def collect_subsample(sample: Path, std_n: int):
    stats = defaultdict(lambda: {"core_utts": 0, "core_inst": 0})
    speakers = defaultdict(lambda: {"n": 0, "hits": defaultdict(int)})
    core_total = 0
    batches = sorted((sample / "batches").glob("batch_*.map.json"))
    done = 0
    for map_path in batches:
        rec_path = (sample / "annotations"
                    / f"{map_path.stem.replace('.map', '')}.annotations.json")
        if not rec_path.exists():
            continue
        done += 1
        occ_map = json.loads(map_path.read_text(encoding="utf-8"))
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        by_line = per_line_constructs(rec)
        for line0, occurrences in enumerate(occ_map):
            found = by_line.get(line0, {})
            for occ in occurrences:
                if occ["component"] == "core":
                    core_total += 1
                sp = speakers[occ["who"]]
                sp["n"] += 1
                for cid, k in found.items():
                    if occ["component"] == "core":
                        stats[cid]["core_utts"] += 1
                        stats[cid]["core_inst"] += k
                    sp["hits"][cid] += 1
    return stats, speakers, core_total, done, len(batches)


def collect_offline(texts_dir: Path):
    stats = defaultdict(lambda: {"utts": 0, "inst": 0})
    speakers = defaultdict(lambda: {"n": 0, "hits": defaultdict(int)})
    total = 0
    recs = sorted((texts_dir / "annotations").glob("*.annotations.json"))
    for rec_path in recs:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        side = texts_dir / f"{rec['text_id']}.utt.json"
        if not side.exists():
            continue
        who = [u["who"] for u in
               json.loads(side.read_text(encoding="utf-8"))["utterances"]]
        by_line = per_line_constructs(rec)
        total += len(who)
        for line0, w in enumerate(who):
            speakers[w]["n"] += 1
            for cid, k in by_line.get(line0, {}).items():
                stats[cid]["utts"] += 1
                stats[cid]["inst"] += k
                speakers[w]["hits"][cid] += 1
    return stats, speakers, total, len(recs)


def coverage_of(speakers, cid, min_n, std_n):
    elig = [s for s in speakers.values() if s["n"] >= min_n]
    if not elig:
        return None, None, 0
    raw = sum(1 for s in elig if s["hits"].get(cid, 0) > 0) / len(elig)
    std = (sum(rarefied(s["n"], s["hits"].get(cid, 0), min(std_n, s["n"]))
               for s in elig) / len(elig))
    return raw, std, len(elig)


def load_strata(tsv: Path):
    strata = {}
    with tsv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    for r in rows:
        if len(r) > 18 and r[0].startswith("S"):
            strata[r[0]] = r[18]          # edqual column
    return strata


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("sample")
    ap.add_argument("--offline", default=None)
    ap.add_argument("--speakers", default=None)
    ap.add_argument("--scores",
                    default="examples/test-corpus-bnc/score-final-2026-09-23.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-speaker-n", type=int, default=40)
    ap.add_argument("--std-n", type=int, default=60)
    args = ap.parse_args(argv)

    from polke.annotate import catalog
    inv = {c["id"]: c for c in catalog()}

    sample = Path(args.sample)
    sub, sub_speakers, core_total, done, n_batches = collect_subsample(
        sample, args.std_n)
    print(f"subsample: {done}/{n_batches} batches aggregated, "
          f"{core_total} core utterance-occurrences")

    off = off_speakers = None
    off_total = off_recs = 0
    if args.offline:
        off, off_speakers, off_total, off_recs = collect_offline(
            Path(args.offline))
        print(f"offline: {off_recs} recordings, {off_total} utterances")

    strata = load_strata(Path(args.speakers)) if args.speakers else {}
    edu_groups = sorted(set(strata.values())) if strata else []

    reliability = {}
    scores_path = Path(args.scores)
    if scores_path.exists():
        for r in json.loads(scores_path.read_text())["constructs"]:
            reliability[r["construct_id"]] = r

    out = Path(args.out) if args.out else sample / "tables"
    out.mkdir(parents=True, exist_ok=True)
    cols = (["id", "name", "category", "tier",
             "est_presence_pct", "est_per_1k_utts",
             "cov_raw", "cov_std", "elig_speakers",
             "exact_presence_pct", "exact_per_1k", "exact_cov",
             "precision", "recall", "support", "flags"]
            + [f"cov_edu_{g}" for g in edu_groups])
    rows_out = []
    for cid, c in sorted(inv.items()):
        s = sub.get(cid, {"core_utts": 0, "core_inst": 0})
        raw, std, elig = coverage_of(sub_speakers, cid,
                                     args.min_speaker_n, args.std_n)
        r = reliability.get(cid, {})
        flags = []
        if cid in RULE_GAPS:
            flags.append("rule-gap")
        if r and (r.get("support") or 0) < 4:
            flags.append("low-validation-support")
        row = {
            "id": cid, "name": c["name"], "category": c["category"],
            "tier": c["detector_type"],
            "est_presence_pct": round(100 * s["core_utts"] / core_total, 3)
            if core_total else "",
            "est_per_1k_utts": round(1000 * s["core_inst"] / core_total, 2)
            if core_total else "",
            "cov_raw": round(raw, 3) if raw is not None else "",
            "cov_std": round(std, 3) if std is not None else "",
            "elig_speakers": elig,
            "precision": r.get("precision", ""), "recall": r.get("recall", ""),
            "support": r.get("support", ""),
            "flags": ";".join(flags),
            "exact_presence_pct": "", "exact_per_1k": "", "exact_cov": "",
        }
        if off is not None and c["detector_type"] in ("rule", "lexicon") \
                and off_total:
            o = off.get(cid, {"utts": 0, "inst": 0})
            oraw, _, _ = coverage_of(off_speakers, cid, 50, args.std_n)
            row["exact_presence_pct"] = round(100 * o["utts"] / off_total, 3)
            row["exact_per_1k"] = round(1000 * o["inst"] / off_total, 2)
            row["exact_cov"] = round(oraw, 3) if oraw is not None else ""
        for g in edu_groups:
            grp = {w: s2 for w, s2 in sub_speakers.items()
                   if strata.get(w) == g}
            graw, _, gn = coverage_of(grp, cid, args.min_speaker_n,
                                      args.std_n)
            row[f"cov_edu_{g}"] = round(graw, 3) if graw is not None else ""
        rows_out.append(row)

    with (out / "constructs.csv").open("w", newline="",
                                       encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)

    ranked = sorted((r for r in rows_out if r["cov_std"] != ""),
                    key=lambda r: -float(r["cov_std"]))
    lines = ["# BLC candidate ranking (interim)" if done < n_batches
             else "# BLC candidate ranking", "",
             f"Subsample: {done}/{n_batches} batches, {core_total} core "
             f"utterances. Offline: {off_recs} recordings, {off_total} "
             f"utterances. Strata: "
             f"{'yes' if strata else 'NO SPEAKER TSV - pending'}.", "",
             "Top 30 by standardised speaker coverage "
             f"(rarefied to {args.std_n} utterances/speaker):", "",
             "| construct | name | tier | cov_std | per-1k | P | R | flags |",
             "|---|---|---|---|---|---|---|---|"]
    for r in ranked[:30]:
        lines.append(f"| {r['id']} | {r['name'][:38]} | {r['tier']} "
                     f"| {r['cov_std']} | {r['est_per_1k_utts']} "
                     f"| {r['precision']} | {r['recall']} | {r['flags']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/constructs.csv and summary.md "
          f"({len(rows_out)} constructs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
