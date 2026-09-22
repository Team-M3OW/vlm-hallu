#!/bin/bash
# The conda TeX Live install is broken: no .fmt files and TeXLive::TLUtils.pm is missing, so
# pdflatex/lualatex/latexmk all fail with "I can't find the format file". tectonic is
# self-contained and builds this document cleanly, so it is the supported route.
set -e
cd "$(dirname "$0")"
OUT=${1:-build}
mkdir -p "$OUT"
tectonic -X compile main.tex --outdir "$OUT" --keep-logs
# The committed PDF is ghostscript /ebook compressed: GitHub's inline viewer refuses PDFs above
# ~10 MB and the raw build is ~15 MB. Full quality stays in the gitignored $OUT/main.pdf.
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH \
   -dDetectDuplicateImages=true -sOutputFile=iclr2027_submission.pdf "$OUT/main.pdf"
echo "submission pdf: $(du -h iclr2027_submission.pdf | cut -f1)"
echo "pages: $(pdfinfo "$OUT/main.pdf" | awk '/Pages/{print $2}')"
grep -icE "undefined (reference|citation)" "$OUT/main.log" | xargs -I{} echo "undefined refs/cites: {}"
