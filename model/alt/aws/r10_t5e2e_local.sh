#!/usr/bin/env bash
# Prueba en seco local de R10-t5e2e (sin AWS ni vLLM real): empaqueta SHA (por defecto HEAD) con empaquetar_r10_t5e2e.sh, arranca un vLLM falso
# (r9_vllm_falso.py) y corre r10_t5e2e.sh con CHISPA_LOCAL=1 sobre una muestra de CHISPA_N consultas (por defecto 60): compila el servidor con
# -tags ort, carga los bancos reales de banco-paridad y el e5.onnx real, y pasa las consultas por HTTP con el codificador dentro del servidor.
# Necesita en local: el fixture (CHISPA_FIXTURE), las baterías (CHISPA_BATS), e5.onnx (CHISPA_ONNX), e5.chispa-tok (CHISPA_TOK) y una libonnxruntime
# del sistema (CHISPA_ORT_LIB; en macOS arm64 sirve la de test_data de onnxruntime_go). Resultados en $D/base/res.
# Uso: CHISPA_FIXTURE=… CHISPA_BATS=… CHISPA_ONNX=… CHISPA_TOK=… CHISPA_ORT_LIB=… bash model/alt/aws/r10_t5e2e_local.sh [SHA]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}
D=${CHISPA_DRY_DIR:-/tmp/chispa-r10t5-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/s3/movil"
: "${CHISPA_FIXTURE:?}" "${CHISPA_BATS:?}" "${CHISPA_ONNX:?}" "${CHISPA_TOK:?}" "${CHISPA_ORT_LIB:?}"
export CHISPA_FIXTURE CHISPA_BATS CHISPA_ONNX
cp server/testdata/intencion.chispa "$D/s3/movil/modelo-int8.chispa"; cp "$CHISPA_TOK" "$D/s3/movil/e5.chispa-tok"
GO_SHA256=$(printf '0%.0s' $(seq 64)) bash model/alt/aws/empaquetar_r10_t5e2e.sh "$sha" "$D/s3" > "$D/empaquetar.log" 2>&1 || { cat "$D/empaquetar.log"; exit 1; }
rm -f "model/alt/aws/lanzados/ud-r10-t5e2e-$sha.sh"
python3 model/alt/aws/r9_vllm_falso.py --puerto 8765 --ms 30 & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
git show "$sha:model/alt/aws/r10_t5e2e.sh" | sed -e "s#__CHISPA_S3__#$D/s3#" -e "s#__N_FILAS__#$(wc -l < "$D/s3/consultas.jsonl" | tr -d ' ')#" -e "s#__E5_SHA256__#local#" -e "s#__COD_SHA256__#local#" -e "s#__BAN_SHA256__#local#" -e "s#__CON_SHA256__#local#" > "$D/ud.sh"
bash -n "$D/ud.sh"
CHISPA_LOCAL=1 CHISPA_BASE="$D/base" CHISPA_LOCAL_URL=http://127.0.0.1:8765 CHISPA_MODELO_L3=qwen3-4b CHISPA_DURA_P=6 CHISPA_CARGA="4 8" CHISPA_N=${CHISPA_N:-60} \
  CHISPA_LOCAL_TAGS="-tags ort" CHISPA_LOCAL_ORT="$CHISPA_ORT_LIB" CHISPA_PAR_N=${CHISPA_PAR_N:-40} CHISPA_MOVIL_S3="$D/s3/movil" bash "$D/ud.sh" || true
echo "== estado"; cat "$D/base/res/estado.tsv"
ls "$D/base/res"
