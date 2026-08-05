"""POLKE command line.

    polke check                         environment + model API diagnostics
    polke catalog [--json]              list the construct inventory
    polke annotate PATH [options]       annotate a text file or corpus folder
    polke serve [--host H] [--port P]   run the HTTP API

`annotate` reads every *.txt file under PATH (or the single file PATH), runs
the detectors, and writes one `<name>.annotations.json` per input into
`--output` (default: an `annotations/` folder next to the corpus). The model
API is probed once at startup; if it is not ready a warning is printed and the
LLM tiers are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .env import load_env, spacy_model


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_check(args) -> int:
    from . import llm as llm_mod
    load_env()
    print(f"spaCy model      : {spacy_model()}", end=" ")
    try:
        from .annotate import load_nlp
        load_nlp()
        print("(loads OK)")
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
    status = llm_mod.check_llm()
    n_llm = len(llm_mod.llm_construct_ids())
    from .registry import constructs
    print(f"constructs       : {len(constructs())} total, {n_llm} need an LLM")
    print(f"LLM model        : {status['model']}")
    print(f"LLM API ready    : {status['ready']} ({status['reason']})")
    if not status["ready"]:
        _warn(llm_mod.warning_text(status))
    return 0


def cmd_catalog(args) -> int:
    from .annotate import catalog
    rows = catalog()
    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    cur = None
    for r in rows:
        if r["category"] != cur:
            cur = r["category"]
            print(f"\n{cur} — {r['category_name']}  [{r['part']}]")
        llm = " (LLM)" if r["needs_llm"] else ""
        print(f"  {r['id']:<8} {r['name']}{llm}")
    return 0


def _iter_inputs(path: Path, ext: str):
    if path.is_file():
        yield path
        return
    yield from sorted(p for p in path.rglob(f"*{ext}") if p.is_file())


def cmd_annotate(args) -> int:
    from . import llm as llm_mod
    from .annotate import Annotator

    src = Path(args.path)
    if not src.exists():
        _warn(f"error: no such file or directory: {src}")
        return 2
    inputs = list(_iter_inputs(src, args.ext))
    if not inputs:
        _warn(f"error: no *{args.ext} files under {src}")
        return 2

    selection = ([s for part in args.constructions for s in part.split(",") if s]
                 if args.constructions else None)

    # Startup probe — warn once, before any file is processed.
    status = None
    if not args.no_llm:
        status = llm_mod.check_llm()
        if not status["ready"]:
            _warn(llm_mod.warning_text(status))
    try:
        ann = Annotator(no_llm=args.no_llm, llm_status=status)
    except RuntimeError as exc:
        _warn(f"error: {exc}")
        return 2

    out_dir = Path(args.output) if args.output else (
        src.parent / "annotations" if src.is_file() else src / "annotations")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None

    n_files = n_anns = 0
    for f in inputs:
        text = f.read_text(encoding="utf-8")
        text_id = f.stem
        annotations = ann.annotate(text, selection=selection, text_id=text_id)
        record = {
            "file": str(f),
            "text_id": text_id,
            "spacy_model": spacy_model(),
            "llm": ann.llm_status,
            "annotations": annotations,
        }
        if jsonl is not None:
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            out = out_dir / f"{f.stem}.annotations.json"
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        n_files += 1
        n_anns += len(annotations)
        print(f"  {f.name}: {len(annotations)} annotations")
    if jsonl is not None:
        jsonl.close()
        print(f"done: {n_files} file(s), {n_anns} annotations -> {args.jsonl}")
    else:
        print(f"done: {n_files} file(s), {n_anns} annotations -> {out_dir}/")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("polke.server:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="polke",
                                description="POLKE grammatical-construction "
                                            "annotator (670 constructs).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="environment + model API diagnostics")

    pc = sub.add_parser("catalog", help="list the construct inventory")
    pc.add_argument("--json", action="store_true", help="emit JSON")

    pa = sub.add_parser("annotate", help="annotate a text file or corpus folder")
    pa.add_argument("path", help="a .txt file, or a folder scanned recursively")
    pa.add_argument("-c", "--constructions", action="append", default=None,
                    metavar="IDS",
                    help="comma-separated construct ids and/or category "
                         "prefixes (e.g. PAS,REL-01); default: all")
    pa.add_argument("-o", "--output", default=None,
                    help="output folder for per-file JSON "
                         "(default: <corpus>/annotations)")
    pa.add_argument("--jsonl", default=None, metavar="FILE",
                    help="write one combined JSONL file instead")
    pa.add_argument("--ext", default=".txt",
                    help="input extension when PATH is a folder (default .txt)")
    pa.add_argument("--no-llm", action="store_true",
                    help="skip the LLM tiers even if a key is configured")

    ps = sub.add_parser("serve", help="run the HTTP API")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8100)
    ps.add_argument("--reload", action="store_true")

    args = p.parse_args(argv)
    return {"check": cmd_check, "catalog": cmd_catalog,
            "annotate": cmd_annotate, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
