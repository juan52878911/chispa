#!/usr/bin/env bash
# R9-m: el resto de massive-es por logprobs (lo que queda del fondo de R1; luego massive-es de fondo2), con el ud de R7b sin cambios y el
# cliente del sha de R5. Se saltan los textos ya reetiquetados (LP_HECHAS = R7 + R7b + R8). No sube nada.
# Uso: bash model/alt/aws/empaquetar_llm_r9m.sh SHA SHA_R5 FONDO_R1 FONDO2 LP_HECHAS DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r5=$2; f1=$3; f2=$4; lph=$5; sal=$6; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$r5" model/alt/llm_logprobs.py model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py 2>/dev/null \
  || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py
python3 - "$f1" "$f2" "$lph" "$sal/reetiqueta.jsonl" <<'PY'
import json, sys
f1, f2, lph, out = sys.argv[1:]
leer = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
hechos = {r["id"] for p in lph.split(",") for r in leer(p)}
filas = [r for p in (f1, f2) for r in leer(p) if r["conjunto"] == "massive-es" and r["id"] not in hechos]
open(out, "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))
print(len(filas), "de R1:", sum(1 for r in filas if r["id"].startswith("fondo:")), "de fondo2:", sum(1 for r in filas if r["id"].startswith("fondo2:")))
PY
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
git show "$sha:model/alt/aws/llm_r7b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/llm-r9m-$sha#" -e "s#__N_REET__#$(wc -l < "$sal/reetiqueta.jsonl" | tr -d ' ')#" \
  > "model/alt/aws/lanzados/ud-llm-r9m-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-llm-r9m-$sha.sh"; (cd "$sal" && shasum -a 256 llm.tgz reetiqueta.jsonl)
