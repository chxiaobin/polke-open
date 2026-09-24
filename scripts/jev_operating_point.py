#!/usr/bin/env python3
"""Validate the Jev operating point on the test-package human gold.

Uses the stored probabilities of jev_presence_experiment.py (no new API
calls). Three analyses:

1. Held-out threshold selection: split the (utterance, construct) pairs by
   RECORDING (odd/even over the sorted record list), pick the F1-best
   threshold on one half, score it on the other, and swap.
2. Calibration: reliability table (predicted probability bins vs empirical
   positive rate) + expected-calibration-error, and the probability-mass
   check Sum(p) vs actual gold positives (matters for probability-weighted
   frequency estimation).
3. Per-construct reliability: constructs (>= 4 judged pairs) where Jev is
   weak at the chosen threshold - the flag list for the paper.

  python scripts/jev_operating_point.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_presence_experiment import OUT, load_pairs, build_requests, prf  # noqa: E402

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def flat(pairs, requests, answers):
    """[(pair_key, prob, gold, polke)] in a stable order."""
    rows = []
    for cid, (_, _, pks) in requests.items():
        for pk, p in zip(pks, answers[cid]):
            rows.append((pk, p, pairs[pk]["gold"], pairs[pk]["polke"]))
    return rows


def f1_at(rows, th):
    tp = sum(1 for _, p, g, _ in rows if p >= th and g)
    pred = sum(1 for _, p, _, _ in rows if p >= th)
    gold = sum(1 for _, _, g, _ in rows if g)
    return prf(tp, pred, gold)


def main():
    pairs, _ = load_pairs()
    requests = build_requests(pairs)
    answers = {r["construct"]: r["answers"] for r in
               json.loads((OUT / "answers.json").read_text())["requests"]}
    rows = flat(pairs, requests, answers)

    records = sorted({pk[0] for pk, _, _, _ in rows})
    half = {tid: i % 2 for i, tid in enumerate(records)}
    lines = ["# Jev operating point — held-out validation & calibration", ""]

    lines += ["## Held-out threshold selection (split by recording)", "",
              "| tune half | best th (tune F1) | held-out P | R | F1 |",
              "|---|---|---|---|---|"]
    chosen = []
    for tune_side in (0, 1):
        tune = [r for r in rows if half[r[0][0]] == tune_side]
        evl = [r for r in rows if half[r[0][0]] != tune_side]
        best = max(THRESHOLDS, key=lambda th: f1_at(tune, th)[2])
        tf = f1_at(tune, best)[2]
        p, r, f1 = f1_at(evl, best)
        chosen.append(best)
        lines.append(f"| {'even' if tune_side == 0 else 'odd'} records | "
                     f"{best} ({tf:.3f}) | {p:.3f} | {r:.3f} | {f1:.3f} |")
    p, r, f1 = f1_at(rows, 0.3)
    lines += ["", f"All pairs at threshold 0.3: P={p:.3f} R={r:.3f} "
              f"F1={f1:.3f}; POLKE pipeline on the same pairs: F1=0.743.", ""]

    lines += ["## Calibration (all pairs)", "",
              "| bin | n | mean p | empirical positive rate |",
              "|---|---|---|---|"]
    ece_num = 0.0
    for lo in [i / 10 for i in range(10)]:
        binned = [(p, g) for _, p, g, _ in rows if lo <= p < lo + 0.1] \
            if lo < 0.9 else [(p, g) for _, p, g, _ in rows if p >= 0.9]
        if not binned:
            continue
        mp = sum(p for p, _ in binned) / len(binned)
        emp = sum(g for _, g in binned) / len(binned)
        ece_num += len(binned) * abs(mp - emp)
        lines.append(f"| {lo:.1f}-{lo + 0.1:.1f} | {len(binned)} | {mp:.3f} "
                     f"| {emp:.3f} |")
    n = len(rows)
    mass = sum(p for _, p, _, _ in rows)
    gold_n = sum(1 for _, _, g, _ in rows if g)
    lines += ["", f"ECE = {ece_num / n:.3f}. Probability mass Sum(p) = "
              f"{mass:.0f} vs {gold_n} actual gold positives "
              f"({mass / gold_n:.2f}x) over {n} pairs.", ""]

    th = round(sum(chosen) / len(chosen), 2)
    lines += [f"## Per-construct reliability at threshold {th}", ""]
    by_cid = defaultdict(list)
    for pk, p, g, pol in rows:
        by_cid[pk[2]].append((pk, p, g, pol))
    weak, strong, small = [], 0, 0
    for cid, rs in sorted(by_cid.items()):
        if len(rs) < 4:
            small += 1
            continue
        f1 = f1_at(rs, th)[2]
        gold_pos = sum(1 for _, _, g, _ in rs if g)
        (weak.append((f1, cid, len(rs), gold_pos)) if f1 < 0.5 else None)
        strong += f1 >= 0.5
    lines += [f"{strong} constructs OK (F1 >= 0.5), {len(weak)} weak, "
              f"{small} with < 4 judged pairs (not assessable).", "",
              "Weak constructs (flag list):", ""]
    for f1, cid, n_pairs, gold_pos in sorted(weak):
        lines.append(f"- {cid}: F1 {f1:.2f} ({n_pairs} pairs, "
                     f"{gold_pos} gold positive)")
    out = ROOT / "docs" / "jev-operating-point.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
