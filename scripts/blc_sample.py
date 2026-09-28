#!/usr/bin/env python3
"""Draw the BLC study's LLM-tier subsample from the converted Spoken BNC.

Two-component, speaker-aware design:

* CORE — a uniform random sample of --core utterances over the whole corpus
  (all speakers, including UNK*): design-unbiased frequency estimates.
* TOPUP — for every identifiable speaker (S....) with at least
  --min-speaker-utts utterances in the corpus, additional draws until the
  speaker has at least --per-speaker sampled utterances: floor support for
  per-speaker coverage estimation. Top-up rows are tagged so frequency
  statistics can exclude or reweight them.

Identical utterance texts are merged: each unique text is annotated once
and an occurrence map (text_id, line, speaker, component) fans the
annotations back out. Output layout (KEEP GITIGNORED - corpus text):

  <out>/batches/batch_0001.txt         500 unique utterances per line
  <out>/batches/batch_0001.map.json    per line: [occurrences...]
  <out>/sample-manifest.json           parameters, seed, totals

  python scripts/blc_sample.py examples/spokenBNC-corpus -o examples/blc-run \
      [--core 70000] [--per-speaker 60] [--min-speaker-utts 50] [--seed 2026]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_universe(corpus: Path):
    """[(text_id, line, speaker, text)] for every non-empty utterance."""
    rows = []
    texts_dir = corpus / "texts" if (corpus / "texts").is_dir() else corpus
    for side in sorted(texts_dir.glob("*.utt.json")):
        meta = json.loads(side.read_text(encoding="utf-8"))
        lines = (texts_dir / f"{meta['text_id']}.txt").read_text(
            encoding="utf-8").splitlines()
        for u in meta["utterances"]:
            text = lines[u["line"] - 1].strip()
            if text:
                rows.append((meta["text_id"], u["line"], u["who"], text))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("corpus")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--core", type=int, default=70000)
    ap.add_argument("--per-speaker", type=int, default=60)
    ap.add_argument("--min-speaker-utts", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    universe = load_universe(Path(args.corpus))
    print(f"universe: {len(universe)} utterances")
    rng = random.Random(args.seed)

    core_idx = set(rng.sample(range(len(universe)), args.core))
    by_speaker = defaultdict(list)
    for i, (tid, line, who, _) in enumerate(universe):
        if who.startswith("S"):
            by_speaker[who].append(i)

    sampled_count = defaultdict(int)
    for i in core_idx:
        sampled_count[universe[i][2]] += 1

    topup_idx = set()
    eligible = 0
    for who, idxs in by_speaker.items():
        if len(idxs) < args.min_speaker_utts:
            continue
        eligible += 1
        deficit = args.per_speaker - sampled_count[who]
        if deficit <= 0:
            continue
        pool = [i for i in idxs if i not in core_idx]
        rng.shuffle(pool)
        topup_idx.update(pool[:deficit])

    chosen = sorted(core_idx | topup_idx)
    print(f"core {len(core_idx)} + topup {len(topup_idx)} = {len(chosen)} "
          f"utterances ({eligible} speakers with >= "
          f"{args.min_speaker_utts} utts get >= {args.per_speaker} sampled)")

    # dedup identical texts; keep every occurrence with its component tag
    unique: dict = {}
    for i in chosen:
        tid, line, who, text = universe[i]
        unique.setdefault(text, []).append(
            {"text_id": tid, "line": line, "who": who,
             "component": "core" if i in core_idx else "topup"})
    texts = sorted(unique)          # deterministic order
    rng.shuffle(texts)              # mix long/short across batches
    n_occ = sum(len(v) for v in unique.values())
    print(f"unique texts: {len(texts)} ({n_occ} occurrences, "
          f"dedup saves {1 - len(texts)/n_occ:.1%})")

    out = Path(args.out)
    bdir = out / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    n_batches = 0
    for lo in range(0, len(texts), args.batch_size):
        n_batches += 1
        chunk = texts[lo:lo + args.batch_size]
        stem = f"batch_{n_batches:04d}"
        (bdir / f"{stem}.txt").write_text(
            "\n".join(chunk) + "\n", encoding="utf-8")
        (bdir / f"{stem}.map.json").write_text(json.dumps(
            [unique[t] for t in chunk], ensure_ascii=False),
            encoding="utf-8")
    (out / "sample-manifest.json").write_text(json.dumps({
        "corpus": str(args.corpus), "seed": args.seed,
        "core": args.core, "per_speaker": args.per_speaker,
        "min_speaker_utts": args.min_speaker_utts,
        "batch_size": args.batch_size,
        "universe": len(universe), "sampled_occurrences": n_occ,
        "topup": len(topup_idx), "unique_texts": len(texts),
        "eligible_speakers": eligible, "batches": n_batches,
    }, indent=1))
    print(f"{n_batches} batches -> {bdir}/")
    print(f"annotate with: python scripts/blc_run.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
