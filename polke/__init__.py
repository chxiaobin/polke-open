from . import schema, registry, pipeline  # light imports; no spaCy at import time
from .schema import Annotation, Span
__all__ = ["schema", "registry", "pipeline", "Annotation", "Span"]
__version__ = "0.1.0"

# Heavy convenience import (spaCy etc.) — import polke.annotate explicitly:
#   from polke.annotate import Annotator
