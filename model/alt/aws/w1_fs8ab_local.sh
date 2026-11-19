#!/usr/bin/env bash
# Prueba en seco local de W1c (A -> B, sin AWS, GPU ni Qwen3-Embedding): empaqueta SHA (por defecto HEAD) con empaquetar_w1_fs8ab.sh sobre un dev INVENTADO (model/alt/tests/dev_sintetico_qwen.py), corre el ud de la
# fase A con CHISPA_LOCAL=1 (bancos fabricados con w1_banco_falso en vez del ONNX; escribe faseA.tar y faseA.json en un "S3" de directorio) y después el ud de la fase B, que lee esos dos ficheros, comprueba
# sus sha256 y corre el e2e del servidor Go en modo gpu contra el vLLM falso (w1_vllm_falso.py), el fs8 de Python y w1_medir. Comprueba además que B se niega a usar una faseA.tar manipulada.
# Uso: bash model/alt/aws/w1_fs8ab_local.sh [SHA]       (CHISPA_PY=python con numpy; CHISPA_N filas en total, por defecto 120). Resultados en $D/a/res y $D/b/res.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}
PY=${CHISPA_PY:-python3}
D=${CHISPA_DRY_DIR:-/tmp/chispa-w1ab-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/s3" "$D/ud"
"$PY" model/alt/tests/dev_sintetico_qwen.py "$D/dev" --test 60 --por-clase 12
crit=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | shasum -a 256 | cut -d' ' -f1)
GO_SHA256=$(printf '0%.0s' $(seq 64)) CRIT_SHA256=$crit CHISPA_PERMITIR_CRIT_SIN_CONGELAR=1 CHISPA_UD_DIR="$D/ud" bash model/alt/aws/empaquetar_w1_fs8ab.sh "$sha" "$D/dev" "$D/s3" > "$D/empaquetar.log" 2>&1 || { cat "$D/empaquetar.log"; exit 1; }
tail -4 "$D/empaquetar.log"
mkdir -p "$D/s3a" "$D/s3b"; cp "$D/s3/codigo.tgz" "$D/s3/dev.tgz" "$D/s3a/"; cp "$D/s3/codigo.tgz" "$D/s3/dev.tgz" "$D/s3b/"
uda=$(ls "$D"/ud/ud-w1-fs8a-*.sh); udb=$(ls "$D"/ud/ud-w1-fs8b-*.sh)
# bancos fabricados con las mismas entradas que preparará A
mkdir -p "$D/ent" && PYTHONPATH=$PWD "$PY" -m model.alt.w1_filas --dev "$D/dev" --salida "$D/ent" --n-por-tarea $(( ${CHISPA_N:-120} / 6 )) > /dev/null
PYTHONPATH=$PWD "$PY" -m model.alt.w1_banco_falso --soporte "$D/ent/soporte.jsonl" --salida "$D/bancos" > /dev/null
echo "== fase A"
export GOTOOLCHAIN=local GOSUMDB=off GOFLAGS=-mod=readonly GOCACHE=${GOCACHE:-$D/gocache} GOPATH=${GOPATH:-$D/gopath}
GOPROXY="file://$D/a/codigo/gomod-proxy" CHISPA_LOCAL=1 CHISPA_S3_LOCAL="$D/s3a" CHISPA_BASE="$D/a" CHISPA_PY="$PY" CHISPA_LOCAL_BANCOS="$D/bancos" CHISPA_N=${CHISPA_N:-120} bash "$uda" > "$D/a.out" 2>&1 || true
cat "$D/a/res/estado.tsv"; [ -f "$D/s3a/faseA.json" ] && [ -f "$D/s3a/faseA.tar" ] || { tail -20 "$D/a.out"; echo "A no dejó faseA"; exit 1; }
( cd "$D/s3a" && sha256sum faseA.tar | cut -c1-16; ls -l faseA.tar faseA.json | awk '{print $5, $9}' )
( cd "$D/s3a/" && cp faseA.tar faseA.tar.bueno )
( cd "$D/s3a" && python3 -I - <<'PYEOF'
import json; j = json.load(open("faseA.json")); print("faseA.json:", {k: (v if k != "ficheros" else "%d ficheros" % len(v)) for k, v in j.items()})
PYEOF
)
echo "== fase B con una faseA.tar manipulada: debe negarse"
python3 -I - "$D/s3a/faseA.tar" <<'PYEOF'
import sys
with open(sys.argv[1], "r+b") as f:
    f.seek(600); b = f.read(1); f.seek(600); f.write(bytes([b[0] ^ 1]))
PYEOF
( cd "$D"; export GOPROXY="file://$D/b0/codigo/gomod-proxy"; CHISPA_LOCAL=1 CHISPA_S3_LOCAL="$D/s3b0" CHISPA_S3A_LOCAL="$D/s3a" CHISPA_BASE="$D/b0" CHISPA_PY="$PY" CHISPA_LOCAL_URL=http://127.0.0.1:8766 CHISPA_N=${CHISPA_N:-120} bash "$udb" > "$D/b0.out" 2>&1 ) || true
grep -q 'FALLO sha' "$D/b0/res/estado.tsv" && echo "B rechazó la faseA manipulada: $(grep -h 'faseA:' "$D/b0.out" | head -1)" || { echo "B NO rechazó la faseA manipulada"; tail "$D/b0.out"; exit 1; }
mv "$D/s3a/faseA.tar.bueno" "$D/s3a/faseA.tar"
echo "== fase B"
( cd model/alt/aws && "$PY" w1_vllm_falso.py 8766 ) & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
GOPROXY="file://$D/b/codigo/gomod-proxy" CHISPA_LOCAL=1 CHISPA_S3_LOCAL="$D/s3b" CHISPA_S3A_LOCAL="$D/s3a" CHISPA_BASE="$D/b" CHISPA_PY="$PY" CHISPA_LOCAL_URL=http://127.0.0.1:8766 CHISPA_N=${CHISPA_N:-120} \
  CHISPA_LOCAL_ORT="${CHISPA_ORT_LIB:-}" bash "$udb" > "$D/b.out" 2>&1 || true
echo "== estado B"; cat "$D/b/res/estado.tsv"
ls "$D/b/res"; cat "$D/b/res/medir.txt" 2>/dev/null || { tail -20 "$D/b.out"; exit 1; }
