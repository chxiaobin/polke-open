#!/usr/bin/env python3
"""Prepare the Spoken BNC2014 XML download for annotation with POLKE.

Two steps:

  convert   untagged corpus XML -> plain text, one utterance per line, plus a
            .utt.json sidecar per text mapping each line to its original
            utterance number and speaker id (for speaker-level dispersion
            analysis; speaker demographics live in the download's
            spoken/metadata/*.tsv files)

  sample    draw N validation chunks (windows of consecutive utterances) from
            the converted corpus, seeded for reproducibility — the input for
            the annotate -> probe -> adjudicate -> score verification loop

Annotate the output with line segmentation, so the utterance is the
annotation unit:  polke annotate <dir> --by-line

Cleaning rules (tag inventory per bnc2014spoken.dtd):
  vocal/event/pause/shift   removed (non-lexical)
  unclear                   transcriber's best guess kept
  trunc                     removed (truncated words, false starts)
  foreign                   removed (non-English material)
  anon                      replaced by a fixed same-category placeholder
                            (the licence forbids undoing anonymisation; a
                            parseable stand-in keeps the syntax intact)
Utterances that are empty after cleaning (e.g. laughter only) are dropped.

The corpus is licensed (copyright Cambridge University Press) and must not
be committed or redistributed — keep all input/output folders gitignored.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DROP = {"vocal", "event", "pause", "shift"}     # empty, non-lexical
DROP_CONTENT = {"trunc", "foreign"}             # content removed too

_NAME = {"m": "James", "f": "Sarah", "n": "Sam"}
ANON = {
    "name": None,                               # gender-aware, see _NAME
    "place": "Milltown",
    "telephoneNumber": "0123 456 7890",
    "address": "10 Mill Lane",
    "email": "sam@example.com",
    "financialDetails": "1234",
    "socialMediaName": "sam123",
    "dateOfBirth": "1990",
    "miscPersonalInfo": "such-and-such",
}


def anon_placeholder(el: ET.Element) -> str:
    t = el.get("type", "miscPersonalInfo")
    if t == "name":
        return _NAME.get(el.get("nameType", "n"), "Sam")
    return ANON.get(t) or "such-and-such"


def _collect(el: ET.Element, parts: list) -> None:
    tag = el.tag
    if tag == "anon":
        parts.append(anon_placeholder(el))
        return
    if tag in DROP or tag in DROP_CONTENT:
        return
    if el.text:
        parts.append(el.text)
    for child in el:
        _collect(child, parts)
        if child.tail:
            parts.append(child.tail)


def utterance_text(u: ET.Element) -> str:
    """Clean one <u> element to plain text ('' if nothing lexical remains)."""
    parts: list = []
    _collect(u, parts)
    return " ".join("".join(parts).split())


def convert_text(xml_path: Path) -> tuple[str, list]:
    """One corpus file -> (text_id, [{'n', 'who', 'text'}, ...])."""
    root = ET.parse(xml_path).getroot()
    text_id = root.get("id", xml_path.stem)
    body = root.find("body")
    utts = []
    if body is not None:
        for u in body.findall("u"):
            cleaned = utterance_text(u)
            if cleaned:
                utts.append({"n": u.get("n", ""),
                             "who": u.get("who", "UNK"),
                             "text": cleaned})
    return text_id, utts


def cmd_convert(args) -> int:
    src = Path(args.src) / "spoken" / "untagged"
    if not src.is_dir():
        print(f"error: {src} not found — point --src at the BNC2014 "
              "download root", file=sys.stderr)
        return 2
    out = Path(args.out) / "texts"
    out.mkdir(parents=True, exist_ok=True)
    n_texts = n_utts = n_words = 0
    for xml_path in sorted(src.glob("*.xml")):
        text_id, utts = convert_text(xml_path)
        if not utts:
            continue
        (out / f"{text_id}.txt").write_text(
            "\n".join(u["text"] for u in utts) + "\n", encoding="utf-8")
        sidecar = {"text_id": text_id, "source": xml_path.name,
                   "utterances": [{"line": i, "n": u["n"], "who": u["who"]}
                                  for i, u in enumerate(utts, 1)]}
        (out / f"{text_id}.utt.json").write_text(
            json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
        n_texts += 1
        n_utts += len(utts)
        n_words += sum(len(u["text"].split()) for u in utts)
        if n_texts % 100 == 0:
            print(f"  {n_texts} texts…", file=sys.stderr)
    print(f"converted {n_texts} texts, {n_utts} utterances, "
          f"~{n_words} words -> {out}/")
    print(f"annotate with: polke annotate {out} --by-line")
    return 0


def _excluded_sources(dirs) -> set:
    """Source text ids used by earlier samples (chunk files are named
    '<TEXTID>__uN-uM.*'); a new sample must draw from DISJOINT recordings."""
    ids = set()
    for d in dirs or []:
        for f in Path(d).glob("*.txt"):
            ids.add(f.stem.split("__")[0])
    return ids


def cmd_sample(args) -> int:
    corpus = Path(args.corpus)
    texts_dir = corpus / "texts" if (corpus / "texts").is_dir() else corpus
    files = sorted(texts_dir.glob("*.txt"))
    banned = _excluded_sources(getattr(args, "exclude", None))
    if banned:
        before = len(files)
        files = [f for f in files if f.stem not in banned]
        print(f"excluding {len(banned)} source recordings from earlier "
              f"samples ({before} -> {len(files)} candidates)")
    if len(files) < args.chunks:
        print(f"error: only {len(files)} converted texts under {texts_dir}, "
              f"cannot sample {args.chunks}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n_words = 0
    for f in rng.sample(files, args.chunks):
        lines = f.read_text(encoding="utf-8").splitlines()
        sidecar = json.loads(
            f.with_suffix("").with_suffix(".utt.json").read_text("utf-8"))
        utts = sidecar["utterances"]
        k = min(args.utterances, len(lines))
        start = rng.randrange(len(lines) - k + 1)
        window, meta = lines[start:start + k], utts[start:start + k]
        first, last = meta[0]["n"] or start + 1, meta[-1]["n"] or start + k
        stem = f"{sidecar['text_id']}__u{first}-u{last}"
        (out / f"{stem}.txt").write_text("\n".join(window) + "\n",
                                         encoding="utf-8")
        chunk_meta = {"text_id": stem, "source_text": sidecar["text_id"],
                      "window": {"first_line": start + 1,
                                 "last_line": start + k},
                      "utterances": [{"line": i, "n": m["n"], "who": m["who"]}
                                     for i, m in enumerate(meta, 1)]}
        (out / f"{stem}.utt.json").write_text(
            json.dumps(chunk_meta, ensure_ascii=False), encoding="utf-8")
        n_words += sum(len(l.split()) for l in window)
    print(f"sampled {args.chunks} chunks x <= {args.utterances} utterances "
          f"(~{n_words} words, seed {args.seed}) -> {out}/")
    print(f"annotate with: polke annotate {out} --by-line")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("convert", help="corpus XML -> utterance-per-line "
                                        "text + speaker sidecars")
    pc.add_argument("--src", required=True,
                    help="BNC2014 download root (contains spoken/untagged/)")
    pc.add_argument("--out", required=True,
                    help="output corpus folder (a texts/ subfolder is "
                         "created; MUST be gitignored)")
    pc.set_defaults(fn=cmd_convert)

    ps = sub.add_parser("sample", help="draw seeded validation chunks from "
                                       "the converted corpus")
    ps.add_argument("--corpus", required=True,
                    help="converted corpus folder (from `convert`)")
    ps.add_argument("--out", required=True,
                    help="output folder for the chunks (MUST be gitignored)")
    ps.add_argument("--chunks", type=int, default=50,
                    help="number of chunks, one per sampled text "
                         "(default 50)")
    ps.add_argument("--utterances", type=int, default=40,
                    help="consecutive utterances per chunk (default 40)")
    ps.add_argument("--seed", type=int, default=42,
                    help="random seed (default 42)")
    ps.add_argument("--exclude", action="append", default=[],
                    help="folder of earlier sampled chunks whose source "
                         "recordings must NOT be drawn again (repeatable)")
    ps.set_defaults(fn=cmd_sample)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
