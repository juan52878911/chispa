#!/usr/bin/env bash
# R7b: dos colas de reetiquetado por logprobs (A y B), cada una con su prefijo y su ud; llm.tgz = el cliente del sha de R5. No sube nada.
#   A: resto del test de lexboe (lo que R7 no hizo); fondo de R1, tareas baratas; fondo de R1, massive-esc; fondo2, lexboe con etiqueta JSON de R4.
#   B: fondo2, massive-esc con etiqueta JSON de R4; fondo de R1, massive-es. (Solo textos que ya tienen etiqueta JSON: R6b compara etiquetas,
#      no tamaño de soporte; fondo2 sin etiquetar de R4 no entra.)
# Uso: bash model/alt/aws/empaquetar_llm_r7b.sh SHA SHA_R5 FILAS_NUEVAS LP_R7 FONDO_R1 FONDO2 L3_FONDO2_R4 DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r5=$2; nu=$3; lp7=$4; f1=$5; f2=$6; l3f2=$7; sal=$8; mkdir -p "$sal/a" "$sal/b"
t=$(mktemp -d)
git archive "$r5" model/alt/llm_logprobs.py model/alt/llm_jevbench.py | tar -x -C "$t"
for x in a b; do COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/$x/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py 2>/dev/null \
  || COPYFILE_DISABLE=1 tar -czf "$sal/$x/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py; done
python3 - "$nu" "$lp7" "$f1" "$f2" "$l3f2" "$sal" <<'PY'
import json, sys
nu, lp7, f1, f2, l3f2, sal = sys.argv[1:]
leer = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]  # noqa: E731
hechos = {r["id"] for r in leer(lp7)}
con_r4 = {r["id"] for r in leer(l3f2)}
fo, fo2 = leer(f1), leer(f2)
caras = ("massive-es", "massive-esc")
a = ([r for r in leer(nu) if r["conjunto"] == "lexboe" and r["id"] not in hechos] + [r for r in fo if r["conjunto"] not in caras]
     + [r for r in fo if r["conjunto"] == "massive-esc"] + [r for r in fo2 if r["conjunto"] == "lexboe" and r["id"] in con_r4])
b = [r for r in fo2 if r["conjunto"] == "massive-esc" and r["id"] in con_r4] + [r for r in fo if r["conjunto"] == "massive-es"]
for x, filas in (("a", a), ("b", b)):
    ids = [r["id"] for r in filas]
    assert len(ids) == len(set(ids))
    open("%s/%s/reetiqueta.jsonl" % (sal, x), "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))
    print(x, len(filas), {c: sum(1 for r in filas if r["conjunto"] == c) for c in sorted({r["conjunto"] for r in filas})})
PY
n() { wc -l < "$1" | tr -d ' '; }
for x in a b; do
  tar -tzf "$sal/$x/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
  git show "$sha:model/alt/aws/llm_r7b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/llm-r7b-$x-$sha#" -e "s#__N_REET__#$(n "$sal/$x/reetiqueta.jsonl")#" \
    > "model/alt/aws/lanzados/ud-llm-r7b-$x-$sha.sh"
  wc -c "model/alt/aws/lanzados/ud-llm-r7b-$x-$sha.sh"; (cd "$sal/$x" && shasum -a 256 llm.tgz reetiqueta.jsonl)
done
