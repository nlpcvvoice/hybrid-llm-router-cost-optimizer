#!/usr/bin/env python3
"""Build colab_train_classifier_head.ipynb from the cell-marked source.

Cells are separated by marker lines "#%% md" (markdown) / "#%% python" (code).
The train/val/test data is embedded into the notebook as a generated DATA cell
(after the intro markdown cell), so the notebook is fully self-contained.
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


def make_embed_cell():
    lines = [
        "# --------------------- embedded data (auto-generated) ---------------------",
        "import os",
        "import json",
        "_EMBEDDED_FILES = {",
    ]
    for name in DATA_FILES:
        payload = (ROOT / "data" / name).read_text(encoding="utf-8")
        lines.append(f'    {json.dumps(name)}: {json.dumps(payload, ensure_ascii=False)},')
    lines += [
        "}",
        "for _name, _payload in _EMBEDDED_FILES.items():",
        "    if not os.path.exists(_name):",
        '        with open(_name, "w", encoding="utf-8") as _fh:',
        "            _fh.write(_payload)",
    ]
    size_kb = {n: round(len((ROOT / "data" / n).read_text(encoding="utf-8")) / 1024, 1)
               for n in DATA_FILES}
    lines.append(f'print("embedded data ready:", {json.dumps(size_kb)})')
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