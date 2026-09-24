#!/usr/bin/env python3
"""Regenerate docs/constructions.md from polke/data/constructs.json.

Usage: python scripts/generate_constructions_doc.py
"""

import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "polke" / "data" / "constructs.json"
LEVELS = REPO / "polke" / "data" / "cefr_levels.csv"
TARGET = REPO / "docs" / "constructions.md"

TIER_LABELS = {
    "rule": "rule",
    "lexicon": "lexicon",
    "hybrid_rule_llm": "rule+LLM",
    "hybrid_lexicon_llm": "lexicon+LLM",
    "llm": "LLM",
}


def anchor(text: str) -> str:
    """GitHub-style heading anchor."""
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def cell(text: str) -> str:
    """Make arbitrary text safe inside a markdown table cell."""
    return re.sub(r"\s+", " ", text.strip()).replace("|", "\\|")


def cefr_levels() -> dict:
    """id -> CEFR level from cefr_levels.csv ("—" when unrated)."""
    import csv
    out = {}
    with LEVELS.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lv = row["level"].strip().upper()
            out[row["id"].strip()] = lv if lv[:1] in "ABC" else "—"
    return out


def main() -> None:
    data = json.loads(SOURCE.read_text())
    constructs = data["constructs"]
    levels = cefr_levels()

    parts: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for c in constructs:
        cats = parts.setdefault(c["part"], OrderedDict())
        cats.setdefault((c["category"], c["category_name"]), []).append(c)

    tiers = Counter(c["detector_type"] for c in constructs)

    out = []
    out.append("# Construction inventory\n")
    out.append(
        f"POLKE annotates **{data['construct_count']} grammatical constructions** "
        f"(inventory schema {data['schema_version']}). This reference is generated "
        "from `polke/data/constructs.json` by `scripts/generate_constructions_doc.py` "
        "— do not edit it by hand.\n"
    )
    out.append(
        "The **CEFR** column is the level assigned to each construction in "
        "`polke/data/cefr_levels.csv` (EGP / CEFR-J-informed estimates used by "
        "the level verifier, `polke level` and `/verify`; edit the CSV to "
        "recalibrate). The vernacular and dysfluency strata are not "
        "level-bearing (—).\n"
    )
    out.append(
        "Each construction has a stable ID (`CATEGORY-NN`) usable in the CLI "
        "(`polke annotate -c PAS,REL-01`), the HTTP API (`constructions` field), "
        "and the Python API (`selection=`). Passing a bare category code selects "
        "every construction in that category.\n"
    )

    out.append("## Detector tiers\n")
    out.append("| tier | mechanism | needs LLM | constructions |")
    out.append("|---|---|---|---|")
    out.append(f"| `rule` | spaCy `DependencyMatcher` patterns | no | {tiers['rule']} |")
    out.append(f"| `lexicon` | dependency pattern gated by curated lexicons | no | {tiers['lexicon']} |")
    out.append(f"| `rule+LLM` | rule proposes a span, LLM picks the reading | yes | {tiers['hybrid_rule_llm']} |")
    out.append(f"| `lexicon+LLM` | lexicon proposes, LLM picks the reading | yes | {tiers['hybrid_lexicon_llm']} |")
    out.append(f"| `LLM` | LLM judges presence over sentence spans | yes | {tiers['llm']} |")
    out.append("")

    out.append("## Contents\n")
    for part, cats in parts.items():
        out.append(f"- **{part}**")
        for (cat, cat_name), items in cats.items():
            heading = f"{cat} — {cat_name}"
            out.append(f"  - [{heading}](#{anchor(heading)}) ({len(items)})")
    out.append("")

    for part, cats in parts.items():
        out.append(f"## {part}\n")
        for (cat, cat_name), items in cats.items():
            out.append(f"### {cat} — {cat_name}\n")
            out.append("| ID | CEFR | Family | Construction | Example | Detector |")
            out.append("|---|---|---|---|---|---|")
            for c in items:
                name = cell(c["name"])
                if c.get("use_notes", "").strip():
                    name += f" — {cell(c['use_notes'])}"
                out.append(
                    f"| {c['id']} | {levels.get(c['id'], '—')} | {cell(c['family'])} | {name} "
                    f"| {cell(c['example'])} | {TIER_LABELS[c['detector_type']]} |"
                )
            out.append("")

    TARGET.write_text("\n".join(out) + "\n")
    print(f"wrote {TARGET.relative_to(REPO)} ({data['construct_count']} constructions)")


if __name__ == "__main__":
    main()
