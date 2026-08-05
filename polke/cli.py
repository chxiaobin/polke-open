"""POLKE command line.

    polke check                         environment + model API diagnostics
    polke catalog [--full] [-c IDS]     list the construct inventory
    polke explain IDS                   how constructs are detected (pipeline)
    polke annotate PATH [options]       annotate a text file or corpus folder
    polke view PATH [-o FILE]           build an HTML viewer for annotations
    polke probe PATH [-c IDS]           LLM sweep for false-negative candidates
    polke score PATH VERDICTS           P/R/F1 per construct from adjudication
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


def _progress_printer(fname: str):
    """Live LLM-call progress on stderr: in-place on a tty, sparse lines
    otherwise. Cleared once the file completes."""
    tty = sys.stderr.isatty()

    def cb(done: int, total: int) -> None:
        if tty:
            sys.stderr.write(f"\r  {fname}: LLM calls {done}/{total}")
            if done == total:
                sys.stderr.write("\r\x1b[2K")
            sys.stderr.flush()
        elif done == total or done % 50 == 0:
            print(f"  {fname}: LLM calls {done}/{total}", file=sys.stderr)

    return cb


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
    from .annotate import catalog, resolve_ids
    from .reference import TIER_LABELS
    rows = catalog()
    if args.constructions:
        sel = [s for part in args.constructions for s in part.split(",") if s]
        wanted = resolve_ids(sel)
        rows = [r for r in rows if r["id"] in wanted]
        if not rows:
            _warn(f"error: no constructs match {','.join(sel)}")
            return 2
    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    cur = None
    for r in rows:
        if r["category"] != cur:
            cur = r["category"]
            print(f"\n{cur} — {r['category_name']}  [{r['part']}]")
        if not args.full:
            llm = " (LLM)" if r["needs_llm"] else ""
            print(f"  {r['id']:<8} {r['name']}{llm}")
            continue
        tier = TIER_LABELS.get(r["detector_type"], r["detector_type"])
        print(f"  {r['id']:<8} {r['name']}  [{tier}]")
        if r["family"]:
            print(f"{'':<11}family : {r['family']}")
        if r["example"]:
            print(f"{'':<11}e.g.   : {r['example']}")
        if r["use_notes"]:
            print(f"{'':<11}note   : {r['use_notes']}")
    return 0


def cmd_explain(args) -> int:
    from .annotate import Annotator, resolve_ids
    from .explain import detector_info
    from .registry import constructs, detectors as contract

    sel = [s for part in args.constructions for s in part.split(",") if s]
    wanted = resolve_ids(sel)
    if not wanted:
        _warn(f"error: no constructs match {','.join(sel)}")
        return 2
    try:
        Annotator(no_llm=True)   # builds spaCy + the full detector registry
    except RuntimeError as exc:
        _warn(f"error: {exc}")
        return 2
    from .detectors.base import registry
    reg, inv, con = registry(), constructs(), contract()

    if args.json:
        out = {cid: (detector_info(reg[cid]) if cid in reg else None)
               for cid in sorted(wanted)}
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2, default=str)
        print()
        return 0

    def field(label, value):
        print(f"  {label:<9}: {value}")

    for cid in sorted(wanted):
        c = inv.get(cid, {})
        print(f"\n{cid} — {c.get('name', '?')}")
        det = reg.get(cid)
        if det is None:
            field("detector", "(none registered)")
            continue
        info = detector_info(det)
        field("tier", info["tier"])
        field("detector", f"{info['class']} @ {info['version']}")
        field("mechanism", info["mechanism"])
        if len(info["serves"]) > 1:
            field("serves", f"{info['serves'][0]}…{info['serves'][-1]} "
                            f"({len(info['serves'])} siblings, one detector)")
        if "patterns" in info:
            field("pattern", json.dumps(info["patterns"]))
        for key, label in (("phrases", "phrases"), ("lexicon", "lexicon")):
            if key in info:
                vals = info[key]
                sample = ", ".join(vals[:12]) + (", …" if len(vals) > 12 else "")
                field(label, f"{len(vals)} entries: {sample}")
        if "scan_rule" in info:
            sr = info["scan_rule"]
            field("rule fn", f"{sr['name']} — {sr['doc']}" if sr["doc"]
                             else sr["name"])
        if "gate" in info:
            field("gate", info["gate"])
        if info.get("wants_context"):
            field("context", "uses preceding dialogue utterances")
        if "llm_prompt" in info:
            print("  prompt   :")
            for line in info["llm_prompt"].strip().splitlines():
                print(f"    | {line}")
        for sub in info.get("sub_detectors", []):
            field("sub", f"{sub['class']} @ {sub['version']} — "
                         f"{sub['mechanism']}")
        cases = con.get(cid, {}).get("test_cases", {})
        for sign, key in (("+", "positive"), ("-", "negative")):
            for s in cases.get(key, []):
                print(f"    {sign} {s}")
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
        annotations = ann.annotate(text, selection=selection, text_id=text_id,
                                   progress=_progress_printer(f.name))
        record = {
            "file": str(f),
            "text_id": text_id,
            "spacy_model": spacy_model(),
            "llm": ann.llm_status,
            "text": text,   # embedded so `polke view` needs no source files
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
    if args.view or args.open:
        from .viewer import write_viewer
        target = Path(args.jsonl) if args.jsonl else out_dir
        try:
            # Reuse the annotator's pipeline for sentence splitting.
            view = write_viewer(target, nlp=ann.nlp)
        except (ValueError, OSError) as exc:
            _warn(f"error: could not build the viewer for {target}: {exc}")
            return 2
        print(f"viewer written: {view}")
        if args.open:
            import webbrowser
            webbrowser.open(view.resolve().as_uri())
    return 0


def cmd_view(args) -> int:
    from .viewer import write_viewer

    src = Path(args.path)
    if not src.exists():
        _warn(f"error: no such file or directory: {src}")
        return 2
    nlp = None
    try:
        from .annotate import load_nlp
        nlp = load_nlp()
        # Register the detectors so the viewer can describe, per construct,
        # how detection works (mechanism tooltips).
        from .build import build_all
        build_all(nlp)
    except RuntimeError as exc:
        _warn(f"note: {exc}; using approximate sentence splitting")
    try:
        out = write_viewer(src, out=args.output, nlp=nlp)
    except ValueError as exc:
        _warn(f"error: {exc}")
        return 2
    print(f"viewer written: {out}")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_probe(args) -> int:
    from .annotate import load_nlp, resolve_ids
    from .probe import make_prober, run

    src = Path(args.path)
    if not src.exists():
        _warn(f"error: no such file or directory: {src}")
        return 2
    sel = ([s for part in args.constructions for s in part.split(",") if s]
           if args.constructions else None)
    wanted = resolve_ids(sel)
    if not wanted:
        _warn(f"error: no constructs match {','.join(sel or [])}")
        return 2
    try:
        prober = make_prober(model=args.model)
    except RuntimeError as exc:
        _warn(f"error: {exc}")
        return 2
    reason = prober.check()
    if reason is not None:
        _warn(f"error: probe model {prober.model!r} not usable ({reason})")
        return 2
    try:
        nlp = load_nlp()
    except RuntimeError as exc:
        _warn(f"error: {exc}")
        return 2
    print(f"probing with {prober.model} "
          f"({len(wanted)} constructs, one call per sentence x category)…")
    try:
        stats = run(src, nlp, prober, wanted,
                    progress=_progress_printer(src.name))
    except ValueError as exc:
        _warn(f"error: {exc}")
        return 2
    print(f"done: {stats['candidates']} candidate(s) from {stats['calls']} "
          f"probe call(s) over {stats['sentences']} sentence(s) in "
          f"{stats['files']} file(s); records updated in place")
    print("next: `polke view` the same path — probe candidates appear as an "
          "extra layer with confirm/reject buttons")
    return 0


def cmd_score(args) -> int:
    from .score import compute, load_verdicts, report, _iter_records

    try:
        records = _iter_records(Path(args.path))
        verdicts = load_verdicts(Path(args.verdicts))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _warn(f"error: {exc}")
        return 2
    result = compute(records, verdicts)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print(report(result, min_support=args.min_support))
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
    pc.add_argument("--full", action="store_true",
                    help="show details per construct: family, example, "
                         "use notes, detector tier")
    pc.add_argument("-c", "--constructions", action="append", default=None,
                    metavar="IDS",
                    help="limit to comma-separated construct ids and/or "
                         "category prefixes (e.g. PAS,REL-01)")

    pe = sub.add_parser("explain",
                        help="show HOW constructs are detected: mechanism, "
                             "patterns, lexicons, LLM prompts, test contract")
    pe.add_argument("constructions", nargs="+", metavar="IDS",
                    help="construct ids and/or category prefixes "
                         "(e.g. PAS or PAS-01,VTA-29)")
    pe.add_argument("--json", action="store_true", help="emit JSON")

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
    pa.add_argument("--view", action="store_true",
                    help="also build the HTML annotation viewer for the "
                         "output (same as running `polke view` on it)")
    pa.add_argument("--open", action="store_true",
                    help="build the viewer (implies --view) and open it in "
                         "a browser")

    pv = sub.add_parser("view",
                        help="build a self-contained HTML page showing "
                             "annotations aligned to their sentences")
    pv.add_argument("path", help="an .annotations.json / .jsonl file, or a "
                                 "folder containing them (scanned recursively)")
    pv.add_argument("-o", "--output", default=None, metavar="FILE",
                    help="output HTML path (default: view.html next to the "
                         "input)")
    pv.add_argument("--open", action="store_true",
                    help="open the page in a browser when done")

    pp = sub.add_parser("probe",
                        help="LLM recall probe: surface false-negative "
                             "CANDIDATES in annotated records (adds a probe "
                             "layer for adjudication in the viewer)")
    pp.add_argument("path", help="annotation output: an .annotations.json / "
                                 ".jsonl file or a folder of them")
    pp.add_argument("-c", "--constructions", action="append", default=None,
                    metavar="IDS",
                    help="construct ids and/or category prefixes to probe "
                         "for (default: all 670 — one LLM call per sentence "
                         "per category)")
    pp.add_argument("--model", default=None,
                    help="probe model (default: POLKE_PROBE_MODEL or the "
                         "annotator model; use a DIFFERENT model for the "
                         "LLM-tier constructs). claude-* models run on the "
                         "Anthropic API (pip install anthropic + "
                         "ANTHROPIC_API_KEY), others on the OpenAI API")

    pr = sub.add_parser("score",
                        help="per-construct precision/recall/F1 from "
                             "adjudicated verdicts")
    pr.add_argument("path", help="the annotation output that was adjudicated")
    pr.add_argument("verdicts", help="verdicts.json exported from the viewer")
    pr.add_argument("--json", action="store_true", help="emit JSON")
    pr.add_argument("--min-support", type=int, default=0, metavar="N",
                    help="hide constructs with support below N (they still "
                         "count in totals)")

    ps = sub.add_parser("serve", help="run the HTTP API")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8100)
    ps.add_argument("--reload", action="store_true")

    args = p.parse_args(argv)
    return {"check": cmd_check, "catalog": cmd_catalog,
            "explain": cmd_explain, "annotate": cmd_annotate,
            "view": cmd_view, "probe": cmd_probe, "score": cmd_score,
            "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
