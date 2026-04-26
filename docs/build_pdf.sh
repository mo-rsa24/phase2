#!/usr/bin/env bash
# Build PROJECT_GUIDE.pdf
# Run from project root:  bash docs/build_pdf.sh
set -euo pipefail

SRC="PROJECT_GUIDE.md"
OUT="PROJECT_GUIDE.pdf"
HEADER="docs/pdf_style.tex"

pandoc "$SRC" \
  --output="$OUT" \
  --pdf-engine=xelatex \
  \
  --variable mainfont="Inter" \
  --variable mainfontoptions="Ligatures=TeX" \
  --variable monofont="DejaVu Sans Mono" \
  --variable monofontoptions="Scale=0.82" \
  \
  --variable geometry="margin=1.25in,top=1in,bottom=1.1in" \
  --variable fontsize="11pt" \
  --variable linestretch="1.25" \
  --variable colorlinks="true" \
  --variable linkcolor="headblue" \
  --variable urlcolor="subblue" \
  \
  --include-in-header="$HEADER" \
  --highlight-style="tango" \
  \
  --table-of-contents \
  --toc-depth=2 \
  2>&1

echo "Built $OUT"
