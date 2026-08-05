"""Per-category detector builders, auto-discovered.

Each sibling module in this package exposes ``build(nlp, client=None) ->
list[Detector]``. They are imported and collected automatically, so adding a new
category is just dropping in ``<cat>.py`` — no shared registration file to edit
(keeps categories independent and conflict-free).

``_ORDER`` pins the few cases where registration order matters (a later detector
intentionally overrides a reference id, e.g. PAS-01). Everything else is
appended in alphabetical module order.
"""
from __future__ import annotations
import importlib
import os
import pkgutil
import sys

# Categories that must register in a specific relative order (override refs).
_ORDER = ["pas"]

# During parallel category development a half-written sibling module must not
# break collection for the others. Default: skip modules that fail to import
# (warn on stderr). Set POLKE_STRICT_CATALOG=1 for the final integrity check
# that asserts every module imports.
_STRICT = os.getenv("POLKE_STRICT_CATALOG") == "1"


def _module_names():
    found = [m.name for m in pkgutil.iter_modules(__path__)
             if not m.name.startswith("_") and m.name != "common"]
    ordered = [n for n in _ORDER if n in found]
    rest = sorted(n for n in found if n not in _ORDER)
    return ordered + rest


def category_builders():
    builders = []
    for name in _module_names():
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # noqa: BLE001
            if _STRICT:
                raise
            print(f"[catalog] skipping {name!r}: {exc}", file=sys.stderr)
            continue
        if hasattr(mod, "build"):
            builders.append(mod.build)
    return builders


def build_categories(nlp, client=None):
    dets = []
    for build in category_builders():
        dets.extend(build(nlp, client=client))
    return dets
