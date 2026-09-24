"""HTTP API.

    polke serve                # or: uvicorn polke.server:app

The spaCy pipeline and all detectors are built once at startup; the model API
is probed at the same time and a warning is logged if the LLM tiers are
unavailable (the service still runs — rule/lexicon tiers are unaffected).

Endpoints:
    GET  /health            service + LLM readiness (+ word-list sizes)
    GET  /catalog           the construct inventory (with CEFR levels)
    GET  /reference         human-readable, filterable inventory reference (HTML)
    POST /annotate          {text, constructions?, text_id?, context?}
    GET  /verify            CEFR level verifier UI (GET / redirects here)
    POST /verify/analyze    {text, use_llm?, constructions?} -> NDJSON stream of
                            per-sentence vocabulary + grammar levels
    POST /verify/analyze/json   same, one JSON document
    GET  /verify/vocab?q=w  word-list entries for one word
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from . import llm as llm_mod
from .annotate import Annotator, catalog
from .cefr import LevelAnalyzer
from .env import load_env, spacy_model
from .reference import reference_html

log = logging.getLogger("polke")
STATIC = Path(__file__).resolve().parent / "static"

_annotator: Optional[Annotator] = None
_verifier: Optional[LevelAnalyzer] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _annotator, _verifier
    load_env()
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    status = llm_mod.check_llm()
    if not status["ready"]:
        log.warning(llm_mod.warning_text(status))
    _annotator = Annotator(llm_status=status, segment="line")
    _verifier = LevelAnalyzer(annotator=_annotator)
    log.info("ready: spaCy=%s llm_ready=%s word_lists=%s", spacy_model(),
             status["ready"], _verifier.vocab.counts)
    yield


app = FastAPI(title="POLKE grammatical-construction annotator",
              version=__version__, lifespan=_lifespan)


class AnnotateRequest(BaseModel):
    text: str = Field(..., description="The text to annotate (English).")
    constructions: Optional[List[str]] = Field(
        None, description="Construct ids and/or category prefixes "
                          "(e.g. ['PAS', 'REL-01']). Omit for all constructs.")
    text_id: str = "doc"
    context: Optional[List[str]] = Field(
        None, description="Preceding dialogue utterances (oldest first) for "
                          "discourse-level detectors.")


class VerifyRequest(BaseModel):
    text: str = Field(..., description="Text to level; line breaks force "
                                       "sentence boundaries.")
    use_llm: bool = Field(True, description="Run the LLM-backed construction "
                                            "tiers (when configured).")
    constructions: Optional[List[str]] = Field(
        None, description="Construct ids / category prefixes to restrict "
                          "the grammar side to (default: all).")


@app.get("/health")
def health() -> dict:
    ready = _annotator is not None
    return {
        "status": "ok" if ready else "starting",
        "version": __version__,
        "spacy_model": spacy_model(),
        "llm": _annotator.llm_status if ready else None,
        "vocab_lists": _verifier.vocab.counts if _verifier else None,
    }


@app.get("/catalog")
def get_catalog() -> dict:
    return {"constructions": catalog()}


@app.get("/reference", response_class=HTMLResponse)
def get_reference() -> str:
    """Browsable inventory: search + part/category/LLM filters, one stable
    anchor per construction (e.g. /reference#PAS-01)."""
    return reference_html()


@app.post("/annotate")
def post_annotate(req: AnnotateRequest) -> dict:
    if _annotator is None:
        raise HTTPException(status_code=503, detail="annotator still loading")
    annotations = _annotator.annotate(
        req.text, selection=req.constructions, text_id=req.text_id,
        context=req.context)
    return {"text_id": req.text_id, "llm": _annotator.llm_status,
            "annotations": annotations}


# ---- CEFR level verifier -------------------------------------------------

def _verifier_ready() -> LevelAnalyzer:
    if _verifier is None:
        raise HTTPException(status_code=503, detail="annotator still loading")
    return _verifier


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/verify")


_FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            '<rect width="32" height="32" rx="6" fill="#2563eb"/>'
            '<text x="16" y="22" font-size="16" font-family="sans-serif" '
            'font-weight="700" fill="#fff" text-anchor="middle">P</text></svg>')


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(_FAVICON, media_type="image/svg+xml")


@app.get("/verify", response_class=HTMLResponse)
def verify_ui():
    """The level verifier UI: paste text, get per-sentence vocabulary and
    grammar levels with level/tier/category filters."""
    return FileResponse(STATIC / "verify" / "index.html")


@app.get("/verify/vocab")
def verify_vocab(q: str = Query(..., min_length=1)) -> dict:
    an = _verifier_ready()
    return {"query": q,
            "entries": [e.__dict__ for e in an.vocab.lookup_all(q.strip())]}


def _ndjson(gen):
    for rec in gen:
        yield json.dumps(rec, ensure_ascii=False) + "\n"


@app.post("/verify/analyze")
def verify_analyze(req: VerifyRequest):
    """Streams NDJSON: one `meta` record, one `sentence` record per sentence
    (in order, as soon as each is done), one `done` record with totals."""
    an = _verifier_ready()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    gen = an.analyze_iter(req.text, selection=req.constructions,
                          use_llm=req.use_llm)
    return StreamingResponse(_ndjson(gen), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/verify/analyze/json")
def verify_analyze_json(req: VerifyRequest) -> dict:
    an = _verifier_ready()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    meta, sentences, done = None, [], None
    for rec in an.analyze_iter(req.text, selection=req.constructions,
                               use_llm=req.use_llm):
        if rec["type"] == "meta":
            meta = rec
        elif rec["type"] == "sentence":
            sentences.append(rec)
        else:
            done = rec
    return {"meta": meta, "sentences": sentences,
            "summary": (done or {}).get("summary")}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
