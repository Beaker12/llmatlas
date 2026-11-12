#!/usr/bin/env bash
# Usage: check_pdf.sh <pdf> [CJK font] [Arabic font] [LaTeX log]
#
# Passes when
#   * the CJK and Arabic fonts are embedded (pdffonts; spaces and case are ignored);
#   * the LaTeX log, if given, has no "Missing character" line (xelatex -halt-on-error does not
#     stop on one, and the glyph is silently dropped from the PDF);
#   * unless CHECK_SAMPLES=0, every sample line of samples.txt is present in the text pdftotext
#     returns. Text extraction depends on the font's mapping and returns Arabic in visual order
#     and presentation forms, so lines are compared as sets of characters after NFKC
#     normalisation: every distinct non-space character of a sample must appear in the extracted
#     text. For Arabic samples only the Arabic letters are required. Whether letters are joined
#     and read right to left is judged by looking at the page.
#
# CHECK_PDF_ALLOW_PUA=1 reports a failing sample line as a warning when the extracted text holds
# private-use or unassigned code points (a font whose ToUnicode map is wrong). For local runs with substitute
# fonts only; CI must not set it.
#
# Font names: keep the spellings identical to latex_elements.preamble in _config.yml and to
# tools/latex_smoke/fonts.tex; all three must change together.
set -euo pipefail

pdf=${1:?usage: check_pdf.sh <pdf> [CJK font] [Arabic font] [LaTeX log]}
cjk=${2:-Noto Sans CJK SC}
arabic=${3:-Noto Naskh Arabic}
log=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
status=0

for tool in pdffonts pdftotext python3; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool"; exit 2; }
done

embedded=$(pdffonts "$pdf" | tail -n +3 | awk '$(NF-4) == "yes" {print $1}' | tr -d ' ')
for want in "$cjk" "$arabic"; do
  if grep -qiF -- "${want// /}" <<<"$embedded"; then
    echo "ok: font $want is embedded"
  else
    echo "FAIL: font $want is not embedded"
    status=1
  fi
done

if [ -n "$log" ]; then
  if [ ! -r "$log" ]; then
    echo "FAIL: cannot read the LaTeX log $log"
    status=1
  elif grep -q 'Missing character' "$log"; then
    echo "FAIL: $log reports missing glyphs:"
    grep -m5 'Missing character' "$log"
    status=1
  else
    echo "ok: no missing characters in $log"
  fi
fi

if [ "${CHECK_SAMPLES:-1}" != 0 ]; then
  text=$(mktemp)
  trap 'rm -f "$text"' EXIT
  pdftotext "$pdf" "$text"
  python3 - "$text" "$here/samples.txt" <<'PY' || status=1
import os
import sys
import unicodedata

raw = open(sys.argv[1], encoding="utf-8").read()
text = set(unicodedata.normalize("NFKC", raw)) - set(" \t\n\r\f\x0c")
# private-use or unassigned code points: a font whose ToUnicode map is wrong
has_pua = any(unicodedata.category(c) in ("Co", "Cn") for c in raw)
allow_pua = os.environ.get("CHECK_PDF_ALLOW_PUA") == "1"


def arabic_letter(c):
    return "؀" <= c <= "ۿ" and unicodedata.category(c).startswith("L")


bad = 0
for line in open(sys.argv[2], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line:
        continue
    kind, _, sample = line.partition(":")
    chars = set(unicodedata.normalize("NFKC", sample)) - set(" \t\n\r\f\x0c")
    if kind == "ar":
        chars = {c for c in chars if arabic_letter(c)}
    missing = sorted(chars - text)
    if not missing:
        print(f"ok: {kind}: {sample.strip()}")
    elif allow_pua and has_pua:
        print(f"WARN: {kind}: {sample.strip()} (missing {missing}; private-use code points seen)")
    else:
        print(f"FAIL: {kind}: {sample.strip()} (missing {missing})")
        bad += 1
sys.exit(1 if bad else 0)
PY
fi
exit $status
