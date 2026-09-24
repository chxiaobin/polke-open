#!/usr/bin/env python3
"""Production Jev annotator: utterance-level construct presence for a corpus
of utterance-per-line .txt files (the spoken_bnc_prepare.py output format).

Design (cost-optimised vs the experiment script):
* one request per (construct, batch of up to --batch utterances): the
  construct description and the task/convention text sit in the request
  STATE once, so the marginal cost per question is the utterance text plus
  a one-line question;
* answers below --floor are not stored (sparse output);
* resumable: a text whose output file already exists is skipped;
* tokens, wall time and derived cost are written next to the results.

Output per text: <out>/<text_id>.jev.json
  {"text_id", "model", "threshold_floor",
   "utterances": [{"line": 1-based, "probs": {"PAS-01": 0.93, ...}}, ...]}

  python scripts/jev_annotate.py CORPUS_DIR -o OUT_DIR [--limit-texts N]
      [--constructs PAS,REL-01] [--batch 200] [--concurrency 24]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_presence_experiment import USD_PER_MTOK, load_env_key  # noqa: E402

TASK = ("Each question presents one utterance of transcribed casual "
        "conversation (spoken English; punctuation may be missing and "
        "false starts are possible). Judge whether the utterance contains "
        "at least one instance of the grammatical construction described "
        "in `construct`: true if some word sequence of the utterance is an "
        "instance of it, false otherwise.")


def construct_states(selection=None):
    from polke.annotate import resolve_ids
    from polke.registry import constructs
    inv = constructs()
    wanted = resolve_ids(selection)
    states = {}
    for cid in sorted(wanted):
        c = inv[cid]
        states[cid] = {
            "task": TASK,
            "construct": {
                "grammar_area": c.get("category_name", c["category"]),
                "name": c["name"],
                "description": (c.get("definition_raw") or c["name"]).strip(),
                "example": c.get("example", ""),
            },
        }
    return states


def batches(lines, cids, states, batch_size):
    for cid in cids:
        for lo in range(0, len(lines), batch_size):
            chunk = lines[lo:lo + batch_size]
            questions = {
                f"u{lo + i}": {
                    "type": "noul",
                    "instructions": {
                        "utterance": text,
                        "q": "Does `utterance` contain the construction "
                             "in `construct` (see `task`)?",
                    },
                } for i, text in enumerate(chunk)}
            yield cid, lo, states[cid], questions


async def annotate_text(client, model, path, out_path, cids, states, args,
                        stats):
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    probs = [dict() for _ in lines]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(cid, lo, state, questions):
        async with sem:
            r = await client.system_one(state, questions, model=model)
        stats["input_tokens"] += r.usage.input_tokens
        stats["questions"] += len(questions)
        stats["requests"] += 1
        for key, ans in r.nouls.items():
            p = ans.noul
            if p >= args.floor:
                probs[int(key[1:])][cid] = round(p, 4)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(*b) for b in
                           batches(lines, cids, states, args.batch)))
    dt = time.perf_counter() - t0
    out_path.write_text(json.dumps({
        "text_id": path.stem, "model": model, "threshold_floor": args.floor,
        "wall_seconds": round(dt, 2),
        "utterances": [{"line": i + 1, "probs": p}
                       for i, p in enumerate(probs)],
    }, ensure_ascii=False))
    return len(lines), dt


async def run(args):
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    sel = ([s for part in args.constructs for s in part.split(",") if s]
           if args.constructs else None)
    states = construct_states(sel)
    cids = sorted(states)
    src = Path(args.corpus)
    texts = sorted(src.glob("*.txt"))[:args.limit_texts or None]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"input_tokens": 0, "questions": 0, "requests": 0}
    retry = RetryPolicy(max_retries=6, backoff_max=30.0, timeout=600.0)
    t0 = time.perf_counter()
    n_utts = 0
    async with AsyncTypeSafeClient(retry=retry, timeout=600.0) as client:
        for path in texts:
            out_path = out_dir / f"{path.stem}.jev.json"
            if out_path.exists():
                print(f"  {path.stem}: exists, skipped")
                continue
            n, dt = await annotate_text(client, args.model, path, out_path,
                                        cids, states, args, stats)
            n_utts += n
            print(f"  {path.stem}: {n} utterances, {len(cids)} constructs, "
                  f"{dt:.1f}s")
    wall = time.perf_counter() - t0
    if stats["questions"]:
        mtok = stats["input_tokens"] / 1e6
        summary = {
            "texts": len(texts), "utterances": n_utts,
            "constructs": len(cids), **stats,
            "wall_seconds": round(wall, 1),
            "tokens_per_question": round(
                stats["input_tokens"] / stats["questions"], 1),
            "questions_per_second": round(stats["questions"] / wall, 1),
            "usd": round(mtok * USD_PER_MTOK, 4),
            "usd_per_utterance": round(
                mtok * USD_PER_MTOK / max(1, n_utts), 6),
            "seconds_per_utterance": round(wall / max(1, n_utts), 3),
        }
        (out_dir / "run-stats.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("corpus", help="dir of utterance-per-line .txt files")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-c", "--constructs", action="append", default=None,
                    help="construct ids / category prefixes (default: all)")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--batch", type=int, default=200,
                    help="utterances per request (default 200)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--floor", type=float, default=0.1,
                    help="store probabilities >= floor (default 0.1)")
    ap.add_argument("--limit-texts", type=int, default=None)
    args = ap.parse_args(argv)
    load_env_key()
    import os
    if "TYPESAFE_API_KEY" not in os.environ:
        ap.error("TYPESAFE_API_KEY not set")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
