#!/usr/bin/env bash
# R8: dos colas de reetiquetado por logprobs con el ud de R7b (`llm_r7b.sh`, sin cambios) y el cliente del sha de R5. No sube nada.
#   A: fondo2 de lexboe con etiqueta JSON de R4 (2 719); luego la 2.ª mitad de massive-es del fondo de R1.
#   B: la 1.ª mitad de massive-es del fondo de R1; luego massive-es de fondo2 (sin etiqueta JSON: soporte nuevo).
#   Se saltan los textos que ya tienen etiqueta por logprobs (LP_HECHAS: R7 + R7b).
# Uso: bash model/alt/aws/empaquetar_llm_r8.sh SHA SHA_R5 FONDO_R1 FONDO2 L3_FONDO2_R4 LP_HECHAS DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r5=$2; f1=$3; f2=$4; l3f2=$5; lph=$6; sal=$7; mkdir -p "$sal/a" "$sal/b"
t=$(mktemp -d)
git archive "$r5" model/alt/llm_logprobs.py model/alt/llm_jevbench.py | tar -x -C "$t"
for x in a b; do COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/$x/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py 2>/dev/null \
  || COPYFILE_DISABLE=1 tar -czf "$sal/$x/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py; done
python3 - "$f1" "$f2" "$l3f2" "$lph" "$sal" <<'PY'
import json, sys
f1, f2, l3f2, lph, sal = sys.argv[1:]
leer = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
hechos = {r["id"] for r in leer(lph)}
con_r4 = {r["id"] for r in leer(l3f2)}
fo, fo2 = leer(f1), leer(f2)
me1 = [r for r in fo if r["conjunto"] == "massive-es" and r["id"] not in hechos]
mitad = len(me1) // 2
a = [r for r in fo2 if r["conjunto"] == "lexboe" and r["id"] in con_r4 and r["id"] not in hechos] + me1[mitad:]
b = me1[:mitad] + [r for r in fo2 if r["conjunto"] == "massive-es" and r["id"] not in hechos]
for x, filas in (("a", a), ("b", b)):
    ids = [r["id"] for r in filas]
    assert len(ids) == len(set(ids))
    open("%s/%s/reetiqueta.jsonl" % (sal, x), "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))
    print(x, len(filas), {c: sum(1 for r in filas if r["conjunto"] == c) for c in sorted({r["conjunto"] for r in filas})})
assert not ({r["id"] for r in a} & {r["id"] for r in b})
PY
n() { wc -l < "$1" | tr -d ' '; }
for x in a b; do
  tar -tzf "$sal/$x/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
  git show "$sha:model/alt/aws/llm_r7b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/llm-r8-$x-$sha#" -e "s#__N_REET__#$(n "$sal/$x/reetiqueta.jsonl")#" \
    > "model/alt/aws/lanzados/ud-llm-r8-$x-$sha.sh"
  wc -c "model/alt/aws/lanzados/ud-llm-r8-$x-$sha.sh"; (cd "$sal/$x" && shasum -a 256 llm.tgz reetiqueta.jsonl)
done
