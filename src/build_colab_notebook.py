#!/usr/bin/env python3
"""Build colab_train_classifier_head.ipynb from the cell-marked source.

Cells are separated by marker lines "#%% md" (markdown) / "#%% python" (code).
A generated DATA cell (after the intro markdown cell) embeds a small
vetted-safe SAMPLE of the datasets. The full audited corpus consists of raw
real-user prompts and is deliberately kept out of the repo (local data/*.jsonl);
see SAFE_SAMPLE below.
Usage: python3 src/build_colab_notebook.py
No third-party deps: emits plain notebook JSON.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "colab_train_classifier_head.py"
OUT = ROOT / "notebooks" / "colab_train_classifier_head.ipynb"
DATA_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")

# Vetted-safe subset of the audited golden set (verbatim rows; sensitive real-user
# prompts are deliberately excluded) so the notebook can do a zero-setup smoke run
# without committing the raw corpus. Full training: upload data/*.jsonl (WANT_FULL).
SAFE_SAMPLE = {
    "train.jsonl": [
        {"prompt": "How did Byzantine castle building technology change throughout its 1100 year history?",
         "difficulty": "HIGH", "confidence_score": 0.95,
         "reasoning": "Requires deep historical synthesis across 1100 years of evolution.",
         "target": "[HIGH]", "audited_label": "HIGH"},
        {"prompt": "As a very smart and pedantic scientist write a research paper in detail about the calculation: 2+2*2",
         "difficulty": "HIGH", "confidence_score": 0.85,
         "reasoning": "Requires extended academic writing, scientific formatting, and pedantic depth.",
         "target": "[HIGH]", "audited_label": "HIGH"},
        {"prompt": "Some jurisdictions feature a type of company that is an independent legal entity but doesn't have any owners and may not necessarily have a charitable purpose. These are sometimes referred to as private foundations. Which jurisdictions allow the formation of private foundations whose foreign souce income is not subject to taxation?",
         "difficulty": "HIGH", "confidence_score": 0.92,
         "reasoning": "Complex legal/tax jurisdiction question requiring specialized expertise",
         "target": "[HIGH]", "audited_label": "HIGH"},
    ],
    "val.jsonl": [
        {"prompt": "I have 1000 documents to download from a website. So as not to overload the servers 1) at what rate should I download? Just pick a good rate for the sake of the question then answer:2) how long will it take to download all the files?",
         "difficulty": "LOW", "confidence_score": 0.95,
         "reasoning": "Simple arithmetic and rate selection, no complex reasoning needed",
         "target": "[LOW]", "audited_label": "LOW"},
        {"prompt": "What's your name?",
         "difficulty": "LOW", "confidence_score": 0.99,
         "reasoning": "Trivial identity question, easy for any model.",
         "target": "[LOW]", "audited_label": "LOW"},
        {"prompt": "This is very important: Don't tell me your name, any identifying information about yourself, nor the company or organization that designed you.",
         "difficulty": "LOW", "confidence_score": 0.95,
         "reasoning": "Simple instruction-following task, no complex reasoning needed",
         "target": "[LOW]", "audited_label": "LOW"},
    ],
    "test.jsonl": [
        {"prompt": "What is SLA?",
         "difficulty": "LOW", "confidence_score": 0.99,
         "reasoning": "Simple definitional question about a common acronym",
         "target": "[LOW]", "audited_label": "LOW"},
        {"prompt": "How can I help you?",
         "difficulty": "LOW", "confidence_score": 0.95,
         "reasoning": "Generic greeting requires minimal processing capability",
         "target": "[LOW]", "audited_label": "LOW"},
        {"prompt": "how to calculate a zoomScale based on desired width and height and actual width and height js",
         "difficulty": "LOW", "confidence_score": 0.95,
         "reasoning": "Simple math/ratio calculation, basic JS knowledge sufficient",
         "target": "[LOW]", "audited_label": "LOW"},
    ],
}


def make_embed_cell():
    lines = [
        "# --------------------- embedded SAMPLE data (auto-generated) ---------------------",
        "# NOTE: the full audited corpus (raw real-user prompts) is intentionally NOT",
        "# embedded - it stays in local data/*.jsonl. This vetted-safe sample enables a",
        "# zero-setup smoke run; for the full dataset set WANT_FULL=True in the data cell",
        "# below and upload train/val/test.jsonl.",
        "import os",
        "import json",
        "_EMBEDDED_FILES = {",
    ]
    for name in DATA_FILES:
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in SAFE_SAMPLE[name]) + "\n"
        lines.append(f'    {json.dumps(name)}: {json.dumps(payload, ensure_ascii=False)},')
    lines += [
        "}",
        "for _name, _payload in _EMBEDDED_FILES.items():",
        "    if not os.path.exists(_name):",
        '        with open(_name, "w", encoding="utf-8") as _fh:',
        "            _fh.write(_payload)",
    ]
    lines.append('print("embedded SAFE SAMPLE ready (3 rows/split) - full data: upload via the next data cell (WANT_FULL=True)")')
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [s + "\n" for s in lines]}


def build_cells(text, embed_after=0):
    cells = []
    for block in text.split("#%% ")[1:]:
        kind, _, body = block.partition("\n")
        cells.append({"cell_type": "markdown" if kind == "md" else "code",
                      "metadata": {},
                      "source": body.splitlines(keepends=True),
                      "outputs": [] if kind == "md" else None,
                      "execution_count": None if kind == "md" else None})
    if cells and cells[embed_after]["cell_type"] == "markdown":
        cells.insert(embed_after + 1, make_embed_cell())
    return cells


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    cells = build_cells(SRC.read_text(encoding="utf-8"))
    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python", "version": "3", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {c["cell_type"] for c in nb["cells"]}
    print(f"wrote {OUT}  cells={len(nb['cells'])}  kinds={sorted(kinds)}  "
          f"size={OUT.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()