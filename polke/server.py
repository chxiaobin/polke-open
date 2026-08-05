"""HTTP API.

    polke serve                # or: uvicorn polke.server:app

The spaCy pipeline and all detectors are built once at startup; the model API
is probed at the same time and a warning is logged if the LLM tiers are
unavailable (the service still runs — rule/lexicon tiers are unaffected).

Endpoints:
    GET  /health     service + LLM readiness
    GET  /catalog    the construct inventory
    GET  /reference  human-readable, filterable inventory reference (HTML)
    POST /annotate   {text, constructions?, text_id?, context?}
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from . import llm as llm_mod
from .annotate import Annotator, catalog
from .env import load_env, spacy_model
from .reference import reference_html

log = logging.getLogger("polke")

_annotator: Optional[Annotator] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _annotator
    load_env()
    status = llm_mod.check_llm()
    if not status["ready"]:
        log.warning(llm_mod.warning_text(status))
    _annotator = Annotator(llm_status=status)
    log.info("ready: spaCy=%s llm_ready=%s", spacy_model(), status["ready"])
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


@app.get("/health")
def health() -> dict:
    ready = _annotator is not None
    return {
        "status": "ok" if ready else "starting",
        "spacy_model": spacy_model(),
        "llm": _annotator.llm_status if ready else None,
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
