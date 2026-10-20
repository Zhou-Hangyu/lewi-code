"""Figures: the neuron-embedding strips of the workloads stacked vertically (decision 2026-09-22): one strip per workload
(embedding_<workload>: the top-ranked LeWi network at n = 2^15, design/teaser-paper) in the order of the learning-curve
figures, no panel letters - the target graph of each strip names it.

  embeddings_main   Figure 3's order without the depth-5 tree (no checkpoints at d = 8): tree, reuse DAG, sampled depth-4 DAG,
                    flat modules, high-arity node
  embeddings_all    the appendix order of learning_curves_all: tree, reuse DAG, sampled depth-4 DAG, flat modules, high-arity node,
                    Fourier tree, Fourier DAG

The strips are composed as vector pages (pypdf): every strip keeps its own rendering; the page is as wide as the widest strip,
the strips stand flush left with a small gap between them. The SVG and PNG copies come from the composed PDF (pdftocairo,
pdftoppm at figstyle.PNG_DPI). Output: <stem>.pdf, svg/<stem>.svg, png/<stem>.png in the output folder.
"""
from __future__ import annotations

import subprocess

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from figstyle import FOLDER, OUT, PNG_DPI

GAP_PT = 4.0
STACKS = {"embeddings_main": ["tree", "dag", "rand", "flat", "dense"],
          "embeddings_all": ["tree", "dag", "rand", "flat", "dense", "ftree", "fdag"]}


def stack(stem: str, workloads: list[str]) -> None:
    pages = []
    for w in workloads:
        f = OUT / f"embedding_{w}.pdf"
        if not f.exists():
            raise SystemExit(f"[{stem}] {f.name} missing: build the strips first (design/teaser-paper)")
        pages.append(PdfReader(str(f)).pages[0])
    width = max(float(p.mediabox.width) for p in pages)
    heights = [float(p.mediabox.height) for p in pages]
    total = sum(heights) + GAP_PT * (len(pages) - 1)
    out = PageObject.create_blank_page(width=width, height=total)
    y = total
    for p, h in zip(pages, heights):
        y -= h
        out.merge_transformed_page(p, Transformation().translate(-float(p.mediabox.left), y - float(p.mediabox.bottom)))
        y -= GAP_PT
    writer = PdfWriter()
    writer.add_page(out)
    pdf = FOLDER["pdf"] / f"{stem}.pdf"
    with open(pdf, "wb") as fh:
        writer.write(fh)
    for folder in ("svg", "png"):
        FOLDER[folder].mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(FOLDER["svg"] / f"{stem}.svg")], check=True)
    subprocess.run(["pdftoppm", "-r", str(PNG_DPI), "-png", "-singlefile", str(pdf), str(FOLDER["png"] / stem)], check=True)
    print(f"[fig] {stem}  {len(pages)} strips, {width / 72:.2f} x {total / 72:.2f} in")


def main() -> None:
    for stem, workloads in STACKS.items():
        stack(stem, workloads)


if __name__ == "__main__":
    main()
