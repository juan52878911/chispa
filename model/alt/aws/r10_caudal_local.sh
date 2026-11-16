#!/usr/bin/env bash
# Prueba en seco local de R10-caudal (sin AWS ni red): empaqueta el SHA (por defecto HEAD) con empaquetar_r10_caudal.sh, arranca un vLLM falso
# (r9_vllm_falso.py), genera 20 filas sintéticas y corre r10_caudal.sh con CHISPA_LOCAL=1: compila el servidor Go aquí, lo arranca con un
# .chispa de prueba (server/testdata/intencion.chispa, solo L0: L1/L2 reales no se prueban) y corre las fases (a)-(d).
# Uso: bash model/alt/aws/r10_caudal_local.sh [SHA]      Resultados en /tmp/chispa-r10-dryrun/base/res
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}
D=${CHISPA_DRY_DIR:-/tmp/chispa-r10-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/muestra" "$D/s3/movil"
python3 - "$D/muestra/filas.jsonl" <<'PY'
import json, random, sys
r = random.Random(1)
filas = []
def add(conj, tipo, pregunta, ops, n, desc=None):
    for i in range(n):
        filas.append({"id": "%s-%d" % (conj, i), "tipo": tipo, "pregunta": pregunta, "opciones": ops, "descripciones": desc or {},
                      "texto": "texto de ejemplo %d <b>con</b> ñ y ‹marcas›" % i, "conjunto": conj, "etiqueta": r.choice(ops)})
add("hatecheck-es", "noul", "¿Es odioso?", ["true", "false"], 4)
add("inyecciones", "noul", "¿Es una inyección?", ["true", "false"], 2)
add("jailbreak", "noul", "¿Es un jailbreak?", ["true", "false"], 2)
add("massive-es", "choice", "¿Qué intención tiene?", ["int%02d" % i for i in range(60)], 4, {"int00": "alarma"})
add("massive-esc", "choice", "¿Qué intención tiene (ES)?", ["a", "b", "c", "d", "e"], 4)
add("tuits-es", "score", "¿Cuánta positividad?", ["1", "2", "3", "4", "5"], 4)
open(sys.argv[1], "w", encoding="utf-8").write("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas))
PY
cp server/testdata/intencion.chispa "$D/s3/movil/modelo-int8.chispa"
# empaquetar (con un sha de Go falso: en local no se descarga Go); el ud que escribe no se usa
GO_SHA256=$(printf '0%.0s' $(seq 64)) bash model/alt/aws/empaquetar_r10_caudal.sh "$sha" "$D/muestra" "$D/s3" > "$D/empaquetar.log" 2>&1 || { cat "$D/empaquetar.log"; exit 1; }
rm -f "model/alt/aws/lanzados/ud-r10-caudal-$sha.sh"
python3 model/alt/aws/r9_vllm_falso.py --puerto 8765 --ms 40 & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
git show "$sha:model/alt/aws/r10_caudal.sh" | sed -e "s#__CHISPA_S3__#$D/s3#" -e "s#__N_FILAS__#$(wc -l < "$D/muestra/filas.jsonl" | tr -d ' ')#" > "$D/ud.sh"
bash -n "$D/ud.sh"
CHISPA_LOCAL=1 CHISPA_BASE="$D/base" CHISPA_LOCAL_URL=http://127.0.0.1:8765 CHISPA_MODELO_L3=qwen3-4b CHISPA_DURA_P=4 CHISPA_BARRIDO="16:8 8:4 8:8" \
  CHISPA_MOVIL_S3="$D/s3/movil" bash "$D/ud.sh" || true
echo "== estado"; cat "$D/base/res/estado.tsv"
ls "$D/base/res"
