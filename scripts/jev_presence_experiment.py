#!/usr/bin/env python3
"""Jev (TypeSafe System One) vs the POLKE pipeline on the human-judged
test-set items: utterance-level construct presence.

Follows the protocol of the earlier written-corpus study
(~/work/project/polke/projects/polke-evaluation/typesafe_presence.py),
adapted to the Spoken BNC2014 test package:

* Gold: the 1,800 human verdicts over the test package. Each judged item
  (system annotation or probe candidate) maps to an (utterance, construct)
  pair; the pair is GOLD-POSITIVE if any of its items was judged tp/span/fn
  and GOLD-NEGATIVE if all were judged fp/reject.
* POLKE's prediction for a pair: positive iff the system annotated that
  construct in that utterance (probe-only pairs are system misses).
* Jev's prediction: one request per construct — state carries the
  construct's inventory entry (name, area, definition, catalog example;
  nothing from the test corpus), one Noul question per judged utterance.
  Positive from a probability threshold (default 0.5; a sweep is reported).

Both systems are scored on the identical pairs, so the comparison is fair;
like all pool-based recall this measures against judged items only.

Timing mirrors the earlier study: a sequential pass and a parallel pass
(--concurrency), with input tokens and cost. Saved outputs contain
probabilities and metrics only — no corpus text (licensed material).

  python scripts/jev_presence_experiment.py            # run + evaluate
  python scripts/jev_presence_experiment.py --evaluate-only --threshold 0.4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PKG = ROOT / "examples" / "test-corpus-bnc" / "annotations-human"
OUT = ROOT / "examples" / "test-corpus-bnc" / "jev-experiment"
EVAL_ENV = Path.home() / "work/project/polke/projects/polke-evaluation/.env"
USD_PER_MTOK = 0.042          # jev-1.13 input price; output tokens are free

POSITIVE = {"tp", "span", "fn"}
NEGATIVE = {"fp", "reject"}


def load_env_key() -> None:
    if "TYPESAFE_API_KEY" in os.environ:
        return
    for path in (ROOT / ".env", EVAL_ENV):
        if path.exists():
            for line in path.read_text().splitlines():
                k, _, v = line.strip().partition("=")
                if k == "TYPESAFE_API_KEY" and v:
                    os.environ[k] = v.strip("'\"")
                    return


def load_pairs():
    """(text_id, line, construct) -> {gold, polke, sentence} from the
    records + human verdicts."""
    verdicts = json.loads((PKG / "verdicts.json").read_text())["verdicts"]
    records = {}
    for p in sorted(PKG.glob("*.annotations.json")):
        rec = json.loads(p.read_text())
        lines = rec["text"].splitlines()
        starts, pos = [], 0
        for l in lines:
            starts.append(pos)
            pos += len(l) + 1
        records[rec["text_id"]] = (rec, lines, starts)

    def line_of(tid, start_char):
        starts = records[tid][2]
        lo = 0
        for i, s in enumerate(starts):
            if s <= start_char:
                lo = i
        return lo

    pairs = {}
    n_items = 0
    for key, v in verdicts.items():
        kind, tid, cid, start, end = key.split("|")
        if tid not in records:
            continue
        line = line_of(tid, int(start))
        pk = (tid, line, cid)
        e = pairs.setdefault(pk, {"gold": False, "polke": False,
                                  "sentence": records[tid][1][line]})
        verdict = v["verdict"]
        if verdict in POSITIVE:
            e["gold"] = True
        if kind == "sys":
            e["polke"] = True
        n_items += 1
    return pairs, n_items


def build_requests(pairs):
    """construct -> (state, questions, ordered pair keys)."""
    from polke.registry import constructs
    inv = constructs()
    by_cid = defaultdict(list)
    for pk in sorted(pairs):
        by_cid[pk[2]].append(pk)
    requests = {}
    for cid, pks in by_cid.items():
        c = inv.get(cid)
        if c is None:
            continue
        state = {"construct": {
            "grammar_area": c.get("category_name", c["category"]),
            "name": c["name"],
            "description": (c.get("definition_raw") or c["name"]).strip(),
            "example": c.get("example", ""),
        }}
        questions = {}
        for i, pk in enumerate(pks):
            questions[f"s{i}"] = {
                "type": "noul",
                "instructions": {
                    "sentence": pairs[pk]["sentence"],
                    "question": "Does `sentence` contain at least one "
                                "instance of the grammatical construction "
                                "described in `construct`? `sentence` is one "
                                "utterance of transcribed casual conversation "
                                "and may lack punctuation.",
                },
                "criteria": {
                    "true": "Some word sequence of `sentence` is an instance "
                            "of `construct`.",
                    "false": "No part of `sentence` is an instance of "
                             "`construct`.",
                },
            }
        requests[cid] = (state, questions, pks)
    return requests


async def ask(client, model, cid, state, questions):
    t0 = time.perf_counter()
    r = await client.system_one(state, questions, model=model)
    return {"construct": cid,
            "latency_s": round(time.perf_counter() - t0, 4),
            "input_tokens": r.usage.input_tokens, "model": r.model,
            "answers": [r.nouls[f"s{i}"].noul for i in range(len(questions))]}


async def run(requests, args):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    retry = RetryPolicy(max_retries=6, backoff_max=30.0, timeout=300.0)
    async with AsyncTypeSafeClient(retry=retry, timeout=300.0) as client:
        await ask(client, args.model, "warmup", "warm up",
                  {"s0": {"type": "noul", "instructions": "Is this a test?"}})
        t0 = time.perf_counter()
        sequential = [await ask(client, args.model, cid, st, qs)
                      for cid, (st, qs, _) in requests.items()]
        seq_s = time.perf_counter() - t0

        sem = asyncio.Semaphore(args.concurrency)

        async def limited(cid, st, qs):
            async with sem:
                return await ask(client, args.model, cid, st, qs)

        t0 = time.perf_counter()
        parallel = await asyncio.gather(*(limited(cid, st, qs)
                                          for cid, (st, qs, _) in requests.items()))
        par_s = time.perf_counter() - t0
    return sequential, seq_s, parallel, par_s


def timing_of(records, wall, n_q):
    lat = sorted(r["latency_s"] for r in records)
    return {"requests": len(records), "questions": n_q,
            "wall_seconds": round(wall, 2),
            "questions_per_second": round(n_q / wall, 1),
            "latency_mean_s": round(statistics.mean(lat), 3),
            "latency_median_s": round(statistics.median(lat), 3),
            "latency_max_s": lat[-1]}


def prf(tp, pred, gold):
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(pairs, requests, answers, threshold):
    """per-construct [tp, predicted, gold] for both systems."""
    per_cid = {}
    for cid, (_, _, pks) in requests.items():
        e = per_cid[cid] = {"jev": [0, 0, 0], "polke": [0, 0, 0],
                            "pairs": len(pks)}
        for pk, prob in zip(pks, answers[cid]):
            gold = pairs[pk]["gold"]
            for system, pred in (("jev", prob >= threshold),
                                 ("polke", pairs[pk]["polke"])):
                e[system][0] += int(pred and gold)
                e[system][1] += int(pred)
                e[system][2] += int(gold)
    return per_cid


def report(pairs, per_cid, timing, threshold, answers, requests):
    from polke.registry import constructs
    inv = constructs()

    def micro(system, subset=None):
        rows = [e for cid, e in per_cid.items()
                if subset is None or cid in subset]
        return prf(*(sum(e[system][i] for e in rows) for i in range(3)))

    lines = ["# Jev vs POLKE pipeline — utterance-level presence on the "
             "human-judged test items", "",
             f"{len(per_cid)} constructs, {sum(e['pairs'] for e in per_cid.values())} "
             f"(utterance, construct) pairs from the 1,800-verdict human gold. "
             f"Jev positive from probability {threshold}.", "",
             "| system | P | R | F1 |", "|---|---|---|---|"]
    for system in ("polke", "jev"):
        p, r, f1 = micro(system)
        lines.append(f"| {system.upper()} | {p:.3f} | {r:.3f} | {f1:.3f} |")

    lines += ["", "## By detector tier of the construct", "",
              "| tier | n pairs | POLKE P/R/F1 | Jev P/R/F1 |", "|---|---|---|---|"]
    tiers = defaultdict(set)
    for cid in per_cid:
        tiers[inv.get(cid, {}).get("detector_type", "?")].add(cid)
    for tier in ("rule", "lexicon", "hybrid_rule_llm",
                 "hybrid_lexicon_llm", "llm"):
        sub = tiers.get(tier)
        if not sub:
            continue
        n = sum(per_cid[c]["pairs"] for c in sub)
        pp = "{:.3f}/{:.3f}/{:.3f}".format(*micro("polke", sub))
        jj = "{:.3f}/{:.3f}/{:.3f}".format(*micro("jev", sub))
        lines.append(f"| {tier} | {n} | {pp} | {jj} |")

    lines += ["", "## Jev threshold sweep (micro)", "",
              "| threshold | P | R | F1 |", "|---|---|---|---|"]
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        e = evaluate(pairs, requests, answers, th)
        p, r, f1 = prf(*(sum(v["jev"][i] for v in e.values()) for i in range(3)))
        lines.append(f"| {th} | {p:.3f} | {r:.3f} | {f1:.3f} |")

    t = timing
    lines += ["", "## Speed and cost (Jev)", "",
              f"Model {t['model']}, {t['input_tokens']/1e6:.3f} Mtok input per "
              f"pass ≈ ${t['input_tokens']/1e6*USD_PER_MTOK:.3f} "
              f"(at jev-1.13 pricing).", "",
              "| pass | requests | questions | wall s | questions/s | "
              "latency mean/median/max s |", "|---|---|---|---|---|---|"]
    for name in ("sequential", "parallel"):
        x = t[name]
        lines.append(f"| {name} | {x['requests']} | {x['questions']} | "
                     f"{x['wall_seconds']} | {x['questions_per_second']} | "
                     f"{x['latency_mean_s']}/{x['latency_median_s']}/"
                     f"{x['latency_max_s']} |")
    lines += ["", f"Parallel = {t['concurrency']} concurrent requests. Max "
              f"answer difference between passes: {t['max_answer_difference']}.",
              ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args(argv)

    pairs, n_items = load_pairs()
    requests = build_requests(pairs)
    n_q = sum(len(q) for _, q, _ in requests.values())
    print(f"{n_items} judged items -> {len(pairs)} (utterance, construct) "
          f"pairs -> {len(requests)} requests / {n_q} questions")
    OUT.mkdir(parents=True, exist_ok=True)
    answers_path, timing_path = OUT / "answers.json", OUT / "timing.json"

    if not args.evaluate_only:
        load_env_key()
        if "TYPESAFE_API_KEY" not in os.environ:
            ap.error("TYPESAFE_API_KEY not set (env, .env, or the "
                     "polke-evaluation .env)")
        seq, seq_s, par, par_s = asyncio.run(run(requests, args))
        diff = max(abs(a - b) for s, p in zip(seq, par)
                   for a, b in zip(s["answers"], p["answers"]))
        timing = {"model": seq[0]["model"], "concurrency": args.concurrency,
                  "input_tokens": sum(r["input_tokens"] for r in seq),
                  "sequential": timing_of(seq, seq_s, n_q),
                  "parallel": timing_of(par, par_s, n_q),
                  "max_answer_difference": round(diff, 4)}
        timing_path.write_text(json.dumps(timing, indent=1))
        answers_path.write_text(json.dumps(
            {"requests": [{k: r[k] for k in
                           ("construct", "latency_s", "input_tokens", "answers")}
                          for r in seq]}))

    answers = {r["construct"]: r["answers"]
               for r in json.loads(answers_path.read_text())["requests"]}
    per_cid = evaluate(pairs, requests, answers, args.threshold)
    md = report(pairs, per_cid, json.loads(timing_path.read_text()),
                args.threshold, answers, requests)
    (OUT / "jev-presence-report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
