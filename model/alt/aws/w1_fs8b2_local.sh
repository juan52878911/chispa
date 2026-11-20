#!/usr/bin/env bash
# Prueba en seco local de W1d (B2) sin AWS ni GPU: fixture de resultados.tgz y faseA.tar INVENTADOS (model.alt.w1_fixture_b2), empaqueta SHA (por defecto HEAD) con empaquetar_w1_fs8b2.sh, corre el ud con
# CHISPA_LOCAL=1 contra el vLLM falso y comprueba que: solo se re-miden las filas no válidas o ausentes (ninguna ya válida), las filas válidas de B quedan intactas, y w1_medir ve 600+600 (aquí 6 x N).
# Uso: bash model/alt/aws/w1_fs8b2_local.sh [SHA]      (CHISPA_PY=python con numpy; CHISPA_N filas por tarea, 25 por defecto)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}; PY=${CHISPA_PY:-python3}; N=${CHISPA_N:-25}
D=${CHISPA_DRY_DIR:-/tmp/chispa-w1b2-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/ud" "$D/sal" "$D/s3a" "$D/s3b" "$D/s3"
PYTHONPATH=$PWD "$PY" -m model.alt.w1_fixture_b2 "$D/fx" --n-por-tarea "$N" | tail -1 > "$D/fx.json"
get() { "$PY" -c "import json;print(json.load(open('$D/fx.json'))$1)"; }
RES=$(get "['res_sha256']"); FA=$(get "['fasea_sha256']")
cp "$D/fx/faseA.tar" "$D/fx/faseA.json" "$D/s3a/"; cp "$D/fx/resultados.tgz" "$D/s3b/"
crit=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | shasum -a 256 | cut -d' ' -f1)
CRIT_SHA256=$crit RESB_SHA256=$RES FASEA_SHA256=$FA CHISPA_PERMITIR_CRIT_SIN_CONGELAR=1 CHISPA_UD_DIR="$D/ud" bash model/alt/aws/empaquetar_w1_fs8b2.sh "$sha" "$D/sal" > "$D/empaquetar.log" 2>&1 || { cat "$D/empaquetar.log"; exit 1; }
tail -5 "$D/empaquetar.log"
cp "$D/sal/codigo.tgz" "$D/s3/"
ud=$(ls "$D"/ud/ud-w1-fs8b2-*.sh)
( cd model/alt/aws && "$PY" w1_vllm_falso.py 8767 ) & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
CHISPA_LOCAL=1 CHISPA_S3_LOCAL="$D/s3" CHISPA_S3A_LOCAL="$D/s3a" CHISPA_S3B_LOCAL="$D/s3b" CHISPA_BASE="$D/b2" CHISPA_PY="$PY" CHISPA_LOCAL_URL=http://127.0.0.1:8767 bash "$ud" > "$D/b2.out" 2>&1 || true
echo "== estado B2"; cat "$D/b2/res/estado.tsv"; cat "$D/b2/res/medir.txt" 2>/dev/null || { tail -20 "$D/b2.out"; exit 1; }
PYTHONPATH=$PWD "$PY" - "$D" "$N" <<'PYEOF'
import json, sys, tarfile
from pathlib import Path
d, n = Path(sys.argv[1]), int(sys.argv[2])
res, nuevo = d / "b2/res", d / "b2/res/py-nuevo"
falt = json.load(open(d / "fx.json"))["faltan"]
leer = lambda p: [json.loads(l) for l in open(p) if l.strip()]
with tarfile.open(d / "fx/resultados.tgz") as t:
    t.extractall(d / "b-orig")
bad = []
for t, k in falt.items():
    nu = leer(nuevo / (t + ".jsonl")) if (nuevo / (t + ".jsonl")).exists() else []
    viejas = {r["id"]: r for r in leer(d / "b-orig/res/py" / (t + ".jsonl"))}
    fin = {r["id"]: r for r in leer(res / "py" / (t + ".jsonl"))}
    if len(nu) != k: bad.append("%s: %d filas nuevas, esperadas %d" % (t, len(nu), k))
    if any(viejas.get(r["id"], {}).get("valida") for r in nu): bad.append("%s: se re-midió una fila ya válida" % t)
    for i, v in viejas.items():
        if v["valida"] and fin[i] != v: bad.append("%s %s: una fila válida de B cambió" % (t, i))
    if len(fin) != n or not all(r["valida"] for r in fin.values()): bad.append("%s: %d filas, no todas válidas" % (t, len(fin)))
c = json.load(open(res / "cifras.json"))
if c["n"] != 6 * n or c["tareas_medidas"] != 6 or c["cobertura_minima_por_tarea"] != {"go": 1.0, "python": 1.0}: bad.append("cifras.json no ve %d+%d: %s" % (6 * n, 6 * n, {k: c[k] for k in ("n", "tareas_medidas", "cobertura_minima_por_tarea")}))
print("PRUEBA EN SECO:", "FALLA " + "; ".join(bad) if bad else "OK (solo se re-midieron %d filas no válidas o ausentes; w1_medir ve %d de Go y %d de Python)" % (sum(falt.values()), 6 * n, 6 * n))
sys.exit(1 if bad else 0)
PYEOF
