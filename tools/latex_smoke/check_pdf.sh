#!/usr/bin/env bash
# Usage: check_pdf.sh <pdf> [CJK font] [Arabic font]
#
# Passes when the expected CJK and Arabic fonts are embedded (pdffonts) and, unless
# CHECK_SAMPLES=0, every string in samples.txt comes back from pdftotext. The font arguments
# default to the CI fonts (Noto Sans CJK, Noto Naskh Arabic); spaces are ignored when matching.
#
# pdftotext returns Arabic as presentation forms and may list the words of an RTL run in
# visual order, so Arabic samples are matched word by word after NFKC normalisation. Whether the
# letters are joined and the run reads right to left is judged by looking at the page.
set -euo pipefail

pdf=${1:?usage: check_pdf.sh <pdf> [CJK font] [Arabic font]}
cjk=${2:-NotoSansCJK}
arabic=${3:-NotoNaskhArabic}
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

if [ "${CHECK_SAMPLES:-1}" != 0 ]; then
  text=$(mktemp)
  trap 'rm -f "$text"' EXIT
  pdftotext "$pdf" "$text"
  python3 - "$text" "$here/samples.txt" <<'PY' || status=1
import re
import sys
import unicodedata


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


text = norm(open(sys.argv[1], encoding="utf-8").read())
bad = 0
for line in open(sys.argv[2], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line:
        continue
    kind, _, sample = line.partition(":")
    parts = sample.split() if kind == "ar" else [sample]
    missing = [p for p in parts if norm(p) not in text]
    print(("FAIL" if missing else "ok") + f": {kind}: {sample}")
    bad += bool(missing)
sys.exit(1 if bad else 0)
PY
fi
exit $status
