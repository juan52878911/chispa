#!/usr/bin/env bash
# R7: llm.tgz (llm_logprobs.py y llm_jevbench.py en el sha de R5), filas-confirma.jsonl (`llm_bateria resto`) y reetiqueta.jsonl, más el ud.
# reetiqueta.jsonl, por prioridad: filas-nuevas (test de lexboe y fakenews-es), fondo de R1 con las tareas baratas primero
# (hatecheck-es, inyecciones, jailbreak, tuits-es intercalados; luego massive-esc; luego massive-es) y fondo2 en su orden. No sube nada.
# Uso: bash model/alt/aws/empaquetar_llm_r7.sh SHA SHA_R5 FILAS_CONFIRMA FILAS_NUEVAS FONDO_R1 FONDO2 DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r5=$2; conf=$3; nu=$4; f1=$5; f2=$6; sal=$7; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$r5" model/alt/llm_logprobs.py model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py
cp "$conf" "$sal/filas-confirma.jsonl"
python3 - "$nu" "$f1" "$f2" "$sal/reetiqueta.jsonl" <<'PY'
import json, sys
nu, f1, f2, sal = sys.argv[1:]
leer = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
fo = leer(f1)
baratas = [r for r in fo if r["conjunto"] not in ("massive-es", "massive-esc")]     # ya vienen intercaladas
filas = leer(nu) + baratas + [r for r in fo if r["conjunto"] == "massive-esc"] + [r for r in fo if r["conjunto"] == "massive-es"] + leer(f2)
ids = [r["id"] for r in filas]
assert len(ids) == len(set(ids)), "ids repetidos"
open(sal, "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))
PY
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/llm_r7.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/llm-r7-$sha#" \
  -e "s#__N_CONF__#$(n "$sal/filas-confirma.jsonl")#" -e "s#__N_REET__#$(n "$sal/reetiqueta.jsonl")#" > "model/alt/aws/lanzados/ud-llm-r7-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-llm-r7-$sha.sh"; (cd "$sal" && shasum -a 256 llm.tgz filas-confirma.jsonl reetiqueta.jsonl)
