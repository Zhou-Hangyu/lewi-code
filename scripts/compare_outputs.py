#!/usr/bin/env python3
"""Compare rebuilt figure files with the shipped ones (their checksums are in CHECKSUMS.sha256).

    python scripts/compare_outputs.py [PATH ...]

Without arguments every file under studies/communication/paper/figures/out-paper listed in CHECKSUMS.sha256 is
compared. A table (csv / md / tex / caption.txt) must be byte-identical. A PDF rendered by another matplotlib build
can differ in bytes while showing the same figure, so a PDF that differs is compared by its extracted text
(pdftotext, when installed) and reported as "same text" or "text differs"; the plotted numbers of most figures are
also written beside them as CSV files, which are compared byte for byte.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies/communication/paper/figures/out-paper"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_text(p: Path) -> str | None:
    if not shutil.which("pdftotext"):
        return None
    return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout


def main(argv: list[str]) -> int:
    shipped = {}
    for line in (ROOT / "CHECKSUMS.sha256").read_text().splitlines():
        digest, _, rel = line.partition("  ")
        shipped[rel] = digest
    targets = [Path(a).resolve() for a in argv] if argv else [ROOT / rel for rel in shipped if rel.startswith("studies/communication/paper/figures/out-paper/")]
    same = text_same = differ = missing = 0
    for p in sorted(targets):
        rel = p.relative_to(ROOT).as_posix()
        if rel not in shipped:
            print(f"?  {rel}: not a shipped file")
            continue
        if not p.exists():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        if sha256(p) == shipped[rel]:
            print(f"same     {rel}")
            same += 1
            continue
        if p.suffix == ".pdf":
            # the shipped copy is gone once the file is rebuilt in place; compare against a saved copy when present
            saved = p.with_suffix(".pdf.shipped")
            if saved.exists():
                a, b = pdf_text(p), pdf_text(saved)
                if a is not None and a == b:
                    print(f"same text  {rel} (bytes differ: another PDF producer)")
                    text_same += 1
                    continue
            print(f"DIFFERS  {rel} (PDF bytes differ; keep a .pdf.shipped copy before rebuilding to compare the text)")
            differ += 1
        else:
            print(f"DIFFERS  {rel}")
            differ += 1
    print(f"\ncompared {same + text_same + differ + missing}: identical {same}, same text {text_same}, differ {differ}, missing {missing}")
    return 1 if differ or missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
