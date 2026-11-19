#!/usr/bin/env bash
# Prueba en seco local de W1-fs8 (sin AWS, GPU ni Qwen3-Embedding): empaqueta SHA (por defecto HEAD) con empaquetar_w1_fs8.sh sobre un dev INVENTADO (model/alt/tests/dev_sintetico_qwen.py),
# arranca un vLLM falso (w1_vllm_falso.py: hash de toda la conversación) y corre w1_fs8.sh con CHISPA_LOCAL=1: compila servidor y herramienta de bancos desde gomod-proxy/, usa bancos fabricados
# (w1_banco_falso.py), pasa las filas por el servidor en modo gpu, corre el fs8 de Python contra el mismo vLLM falso y calcula cifras.json. Como el falso contesta por hash del
# prompt, un prompt de Go distinto del de Python (demostraciones, orden, formato) se vería como diferencia de macro. Resultados en $D/base/res.
# Uso: bash model/alt/aws/w1_fs8_local.sh [SHA]       (CHISPA_PY=python con numpy; CHISPA_N filas en total, por defecto 120)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}
PY=${CHISPA_PY:-python3}
D=${CHISPA_DRY_DIR:-/tmp/chispa-w1-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/s3" "$D/ud"
"$PY" model/alt/tests/dev_sintetico_qwen.py "$D/dev" --test 60 --por-clase 12
crit=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | shasum -a 256 | cut -d' ' -f1)
GO_SHA256=$(printf '0%.0s' $(seq 64)) CRIT_SHA256=$crit CHISPA_PERMITIR_CRIT_SIN_CONGELAR=1 CHISPA_UD_DIR="$D/ud" bash model/alt/aws/empaquetar_w1_fs8.sh "$sha" "$D/dev" "$D/s3" > "$D/empaquetar.log" 2>&1 || { cat "$D/empaquetar.log"; exit 1; }
ud=$(ls "$D"/ud/ud-w1-fs8-*.sh); grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'S3_LOCAL' && { echo "placeholders sin sustituir"; exit 1; }
# bancos fabricados con las mismas entradas que preparará el ud
mkdir -p "$D/ent" && PYTHONPATH=$PWD "$PY" -m model.alt.w1_filas --dev "$D/dev" --salida "$D/ent" --n-por-tarea 20 > /dev/null
PYTHONPATH=$PWD "$PY" -m model.alt.w1_banco_falso --soporte "$D/ent/soporte.jsonl" --salida "$D/bancos" > /dev/null
( cd model/alt/aws && "$PY" w1_vllm_falso.py 8766 ) & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
export GOPROXY="file://$D/base/codigo/gomod-proxy" GOSUMDB=off GOFLAGS=-mod=readonly GOTOOLCHAIN=local GOCACHE=${GOCACHE:-$D/gocache} GOPATH=${GOPATH:-$D/gopath}
CHISPA_LOCAL=1 CHISPA_S3_LOCAL="$D/s3" CHISPA_BASE="$D/base" CHISPA_PY="$PY" CHISPA_LOCAL_URL=http://127.0.0.1:8766 CHISPA_LOCAL_BANCOS="$D/bancos" CHISPA_N=${CHISPA_N:-120} \
  CHISPA_LOCAL_ORT="${CHISPA_ORT_LIB:-}" bash "$ud" || true
echo "== estado"; cat "$D/base/res/estado.tsv"
ls "$D/base/res"; cat "$D/base/res/medir.txt" 2>/dev/null || true
