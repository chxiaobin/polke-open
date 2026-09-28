#!/usr/bin/env python3
"""Resumable annotation runner for the BLC subsample (blc_sample.py output).

Loads the Annotator once (spaCy + detectors + LLM client) and processes the
sample's batch files in order, one output record per batch:

  <sample>/annotations/batch_0001.annotations.json

A batch whose output exists is skipped, so the run resumes for free after
any interruption - run it under tmux/nohup, NOT as a child of an
interactive session. Progress and ETA go to stdout and <sample>/status.json
(watchable while the run is live).

  POLKE_LLM_CONCURRENCY=32 python scripts/blc_run.py examples/blc-run
  python scripts/blc_run.py examples/blc-run --limit 2      # smoke test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("sample", help="blc_sample.py output dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N pending batches (smoke test)")
    ap.add_argument("--no-llm", action="store_true",
                    help="offline tiers only")
    args = ap.parse_args(argv)

    sample = Path(args.sample)
    batches = sorted((sample / "batches").glob("batch_*.txt"))
    out_dir = sample / "annotations"
    out_dir.mkdir(exist_ok=True)
    pending = [b for b in batches
               if not (out_dir / f"{b.stem}.annotations.json").exists()]
    print(f"{len(batches)} batches, {len(pending)} pending "
          f"(concurrency {os.getenv('POLKE_LLM_CONCURRENCY', '8')}, "
          f"{'offline only' if args.no_llm else 'full pipeline'})")
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("nothing to do")
        return 0

    from polke.annotate import Annotator
    from polke.env import spacy_model
    ann = Annotator(no_llm=args.no_llm, segment="line")

    t0 = time.time()
    done_utts = 0
    for k, batch in enumerate(pending, 1):
        text = batch.read_text(encoding="utf-8")
        n_lines = sum(1 for l in text.splitlines() if l.strip())
        bt0 = time.time()
        annotations = ann.annotate(text, text_id=batch.stem)
        record = {
            "file": str(batch), "text_id": batch.stem,
            "spacy_model": spacy_model(), "llm": ann.llm_status,
            "text": text, "annotations": annotations,
        }
        out = out_dir / f"{batch.stem}.annotations.json"
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(out)
        done_utts += n_lines
        dt = time.time() - bt0
        rate = done_utts / (time.time() - t0)
        remaining = sum(1 for b in pending[k:])
        eta_s = remaining * (time.time() - t0) / k
        eta = (datetime.datetime.now()
               + datetime.timedelta(seconds=eta_s)).strftime("%m-%d %H:%M")
        print(f"[{k}/{len(pending)}] {batch.stem}: {n_lines} utts, "
              f"{len(annotations)} anns, {dt:.0f}s "
              f"({rate:.2f} utts/s overall, ETA {eta})", flush=True)
        (sample / "status.json").write_text(json.dumps({
            "done_batches": k, "pending_batches": len(pending) - k,
            "utts_per_second": round(rate, 3),
            "eta": eta, "updated": time.strftime("%F %T")}))
    print(f"finished {len(pending)} batches in "
          f"{(time.time() - t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
