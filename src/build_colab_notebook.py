#!/usr/bin/env python3
"""Build colab_train_classifier_head.ipynb from the cell-marked source.

Cells are separated by marker lines "#%% md" (markdown) / "#%% python" (code).
Usage: python3 src/build_colab_notebook.py
No third-party deps: emits plain notebook JSON.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "colab_train_classifier_head.py"
OUT = ROOT / "notebooks" / "colab_train_classifier_head.ipynb"


def build_cells(text):
    cells = []
    for block in text.split("#%% ")[1:]:
        kind, _, body = block.partition("\n")
        cells.append({"cell_type": "markdown" if kind == "md" else "code",
                      "metadata": {},
                      "source": body.splitlines(keepends=True),
                      "outputs": [] if kind == "md" else None,
                      "execution_count": None if kind == "md" else None})
    return cells


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    nb = {"cells": build_cells(SRC.read_text(encoding="utf-8")),
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python", "version": "3", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {c["cell_type"] for c in nb["cells"]}
    print(f"wrote {OUT}  cells={len(nb['cells'])}  kinds={sorted(kinds)}")


if __name__ == "__main__":
    main()