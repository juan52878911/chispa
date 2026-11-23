#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-cuenta-aws/llm-bateria-0e58743
# Paso 2: techo del LLM en la batería no vista (bf16 con vLLM y Q4_K_M con llama.cpp CUDA) y etiquetado de un fondo sin etiqueta,
# en una g5.xlarge BAJO DEMANDA (1,006 USD/h; tope duro CHISPA_MIN=85 min = 1,43 USD). user-data (< 16 KB; DLAMI GPU PyTorch AL2023,
# rol rol-ec2-pruebas sin ListBucket), con --instance-initiated-shutdown-behavior terminate. Todo por nombre exacto.
# Lee:     $CHISPA_S3/{llm.tgz,filas.jsonl,filas-q4.jsonl,fondo.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Fases, en orden de prioridad (cada una con su tope; si se acaba el tiempo se corta la última y se sube lo que haya):
#   A. bf16: Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f (Apache-2.0), vLLM 0.33.0, mismo método que llm-3fcfd14 (esquema con todos los ids,
#      contexto 16 k, un reintento) sobre filas.jsonl (muestra estratificada; hatecheck-es completa).
#   B. Q4_K_M: unsloth/Qwen3-4B-Instruct-2507-GGUF @ a06e946b (el GGUF de L3), llama.cpp b12138 CUDA 12.8 (binario oficial, sha256
#      comprobado), sobre filas-q4.jsonl (la submuestra anidada primero, luego el resto intercalado), hasta Q4_MIN minutos.
#   C. fondo: el mismo bf16 etiqueta fondo.jsonl (textos sin etiqueta intercalados por tarea) hasta que queden 6 + ESPERA_MIN minutos.
# Las etiquetas de la batería son de las fuentes (nada de Jev); el fondo va sin etiqueta.
MIN=${CHISPA_MIN:-85}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-8}; Q4_MIN=${CHISPA_Q4_MIN:-22}; HILOS=${CHISPA_HILOS:-32}
REV=cdbee75f17c01a7cc42f958dc650907174af0554; VLLM=${CHISPA_VLLM:-0.33.0}
GGUF_REPO=unsloth/Qwen3-4B-Instruct-2507-GGUF; GGUF_REV=a06e946bb6b655725eafa393f4a9745d460374c9; GGUF=Qwen3-4B-Instruct-2507-Q4_K_M.gguf
GGUF_SHA=3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597
LCPP=b12138; LCPP_URL=https://github.com/ggml-org/llama.cpp/releases/download/$LCPP
LCPP_SHA=8aa6fd55e99501a5865fda102f974f87464d1212b5b3bffa4efd3fb6affa9d9e; CUDART_SHA=1402566babfd307f8e9d1e2933b270de79b8a0c15169ac21d6e772f3ec8d26a4
N_FILAS=3116; N_Q4=3116; N_FONDO=8137
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-llmb-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-llmb}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque llm-bateria $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min q4=${Q4_MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in vllm vllm2 llama; do [ -f "$BASE/$l.log" ] && tail -n 200 "$BASE/$l.log" > "$OUT/$l-cola.log"; done
  ( cd "$BASE" && tar -czf resultados.tgz res ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    # sin ListBucket: GetObject de la marca exacta; el orquestador trae y verifica resultados.tgz y entonces sube `verificado`
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 30; do [ "$(quedan_min)" -le $((6 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_jevbench; pkill -TERM -f vllm.entrypoints; pkill -TERM -f llama-server; break; }; done ) &
# cliente con tope: corre hasta acabar o hasta que queden $1 minutos (o pasen $2 minutos); lo que haya escrito queda (reanudable)
cliente() {
  local queda=$1 dura=$2; shift 2; local t=$(date +%s)
  "$VPY" "$BASE/codigo/llm_jevbench.py" "$@" & local p=$!
  while kill -0 $p 2>/dev/null; do
    if [ "$(quedan_min)" -le "$queda" ] || [ $(( ($(date +%s) - t) / 60 )) -ge "$dura" ]; then echo "== corte por tiempo"; kill -TERM $p; sleep 5; kill -KILL $p 2>/dev/null; break; fi
    sleep 10
  done; wait $p 2>/dev/null
}
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

# --- 1. entorno: venv con vLLM fijado (el de llm-3fcfd14 en esta AMI)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub \
    || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto
echo "== 2. entradas $(date -u +%T)"
for f in llm.tgz filas.jsonl filas-q4.jsonl fondo.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$BASE/codigo" && tar -xzf "$BASE/llm.tgz" -C "$BASE/codigo" || { fallo "llm.tgz"; cierre; }
[ "$(lineas "$BASE/filas.jsonl")" = "$N_FILAS" ] && [ "$(lineas "$BASE/filas-q4.jsonl")" = "$N_Q4" ] && [ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] \
  || { fallo "recuentos de entrada distintos de $N_FILAS/$N_Q4/$N_FONDO"; estado entradas "FALLO recuentos"; cierre; }
# autoprueba: la primera fila de cada tarea (incluye massive-es, 60 opciones)
"$PY" -c "
import json,sys
v={}
for l in open(sys.argv[1]):
    r=json.loads(l); v.setdefault(r['conjunto'], l)
open(sys.argv[2],'w').write(''.join(v[k] for k in sorted(v)))" "$BASE/filas.jsonl" "$BASE/auto.jsonl"
estado entradas "OK $N_FILAS filas, $N_Q4 q4, $N_FONDO fondo, $(lineas "$BASE/auto.jsonl") de autoprueba"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-2}")

arrancar_vllm() {   # $1 = log
  # la DLAMI no trae nvcc: el muestreador de FlashInfer se compila con JIT y tumba el arranque (ronda d36f8ab); se usa el de PyTorch
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/modelo" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/$1.log" 2>&1 &
  local p=$!
  for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $p 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b
}

if [ "$LOCAL" = 1 ]; then
  URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL (un servidor OpenAI ya arrancado)}; MOD=${CHISPA_LOCAL_MODELO:-q4}
  estado modelo "OMITIDO (local: $URL)"
else
  URL=http://127.0.0.1:8000; MOD=qwen3-4b
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE/modelo', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" \
    || { fallo "descarga del modelo"; estado modelo "FALLO"; cierre; }
  estado modelo "OK Qwen/Qwen3-4B-Instruct-2507@$REV"
  arrancar_vllm vllm || { tail -5 "$BASE/vllm.log"; fallo "vLLM no arrancó"; estado servidor "FALLO"; cierre; }
  estado servidor "OK vllm $VLLM, ctx 16384"
fi

# --- 3. autoprueba bf16 (6 casos, uno por tarea): si el servidor rechaza el esquema se sabe en segundos
"$VPY" "$BASE/codigo/llm_jevbench.py" --filas "$BASE/auto.jsonl" $(urls $URL 6) --modelo $MOD --salida "$OUT/auto-bf16.jsonl" --formato vllm --timeout 600 "${MAX[@]}"
NV=$(validas "$OUT/auto-bf16.jsonl"); NA=$(lineas "$OUT/auto-bf16.jsonl")
[ "$NA" -gt 0 ] && [ $(( NV * 3 )) -ge $(( NA * 2 )) ] || { head -c 600 "$OUT/auto-bf16.jsonl"; fallo "autoprueba bf16: $NV de $NA válidas"; estado prueba "FALLO"; cierre; }
estado prueba "OK bf16 $NV/$NA válidas"

# --- 4. fase A: bf16 en toda la muestra (prioridad máxima; solo la corta el tope general)
T1=$(date +%s)
cliente $((6 + ESPERA_MIN + 1)) 999 --filas "$BASE/filas.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/l3_bateria.jsonl" --formato vllm --timeout 600 --reintentos 1 "${MAX[@]}"
estado bf16 "$(lineas "$OUT/l3_bateria.jsonl") de $N_FILAS casos ($(validas "$OUT/l3_bateria.jsonl") válidas) en $(( $(date +%s) - T1 )) s"
[ "$LOCAL" = 1 ] || parar vllm.entrypoints

# --- 5. fase B: Q4_K_M con llama.cpp CUDA (el mismo GGUF de L3), hasta Q4_MIN minutos
echo "== 5. q4 $(date -u +%T) quedan $(quedan_min) min"
Q4OK=1
if [ "$LOCAL" = 1 ]; then QURL=$URL; QMOD=$MOD
elif [ "$(quedan_min)" -le $((6 + ESPERA_MIN + 8)) ]; then estado q4 "OMITIDO (sin tiempo)"; Q4OK=0
else
  QURL=http://127.0.0.1:8001; QMOD=q4; L=$BASE/llamacpp; mkdir -p "$L"
  ( cd "$L" && curl -sSfLo l.tgz "$LCPP_URL/llama-$LCPP-bin-ubuntu-cuda-12.8-x64.tar.gz" && curl -sSfLo c.tgz "$LCPP_URL/cudart-llama-$LCPP-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    && printf '%s  l.tgz\n%s  c.tgz\n' "$LCPP_SHA" "$CUDART_SHA" | sha256sum -c --quiet && tar -xzf l.tgz && tar -xzf c.tgz && rm -f l.tgz c.tgz ) \
    || { fallo "llama.cpp $LCPP CUDA (descarga o sha256)"; Q4OK=0; }
  "$VPY" -c "from huggingface_hub import hf_hub_download as h; print(h('$GGUF_REPO', '$GGUF', revision='$GGUF_REV', local_dir='$BASE/gguf'))" > /dev/null \
    && echo "$GGUF_SHA  $BASE/gguf/$GGUF" | sha256sum -c --quiet || { fallo "GGUF (descarga o sha256)"; Q4OK=0; }
  if [ $Q4OK = 1 ]; then
    LS=$(find "$L" -name llama-server -type f | head -1); mkdir -p "$L/lib"
    ldconfig -p | grep -q 'libgomp.so.1 ' || { g=$(find /opt /usr/lib64 -name 'libgomp*.so*' 2>/dev/null | head -1); [ -n "$g" ] && ln -sf "$g" "$L/lib/libgomp.so.1"; }
    export LD_LIBRARY_PATH="$L/lib:$(find "$L" -name '*.so*' -printf '%h\n' | sort -u | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ldd "$LS" | grep 'not found' && echo "AVISO: bibliotecas sin resolver"
    nohup "$LS" -m "$BASE/gguf/$GGUF" --host 127.0.0.1 --port 8001 -ngl 99 -np 16 -c 98304 -fa on --jinja --alias q4 > "$BASE/llama.log" 2>&1 &
    p=$!; for _ in $(seq 60); do curl -s -m 3 localhost:8001/v1/models | grep -q q4 && break; kill -0 $p 2>/dev/null || break; sleep 5; done
    curl -s -m 3 localhost:8001/v1/models | grep -q q4 || { tail -5 "$BASE/llama.log"; fallo "llama-server no arrancó"; Q4OK=0; }
    grep -m3 -i -E "offloaded|CUDA0|ggml_cuda_init" "$BASE/llama.log"
  fi
fi
if [ $Q4OK = 1 ]; then
  T1=$(date +%s)
  cliente $((6 + ESPERA_MIN + 4)) "$Q4_MIN" --filas "$BASE/filas-q4.jsonl" $(urls $QURL 16) --modelo $QMOD --salida "$OUT/l3_q4.jsonl" --formato vllm --timeout 600 --reintentos 1 "${MAX[@]}"
  estado q4 "$(lineas "$OUT/l3_q4.jsonl") de $N_Q4 casos ($(validas "$OUT/l3_q4.jsonl") válidas) en $(( $(date +%s) - T1 )) s"
else grep -q '^q4' "$OUT/estado.tsv" 2>/dev/null || estado q4 "FALLO (ver log)"; fi
[ "$LOCAL" = 1 ] || parar llama-server

# --- 6. fase C: etiquetado del fondo con bf16 hasta el tope
echo "== 6. fondo $(date -u +%T) quedan $(quedan_min) min"
if [ "$(quedan_min)" -le $((6 + ESPERA_MIN + 4)) ]; then estado fondo "OMITIDO (sin tiempo)"
else
  if [ "$LOCAL" != 1 ]; then arrancar_vllm vllm2 || { fallo "vLLM no volvió a arrancar"; estado fondo "FALLO servidor"; cierre; }; fi
  T1=$(date +%s)
  cliente $((6 + ESPERA_MIN + 1)) 999 --filas "$BASE/fondo.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/l3_fondo.jsonl" --formato vllm --timeout 600 --reintentos 1 "${MAX[@]}"
  estado fondo "$(lineas "$OUT/l3_fondo.jsonl") de $N_FONDO textos ($(validas "$OUT/l3_fondo.jsonl") válidas) en $(( $(date +%s) - T1 )) s"
fi

# --- 7. resumen sin dependencias (la medición con IC, kNN y fusión se hace fuera, con `llm_bateria medir`)
"$PY" - "$BASE" > "$OUT/resumen.json" <<'EOF'
import json, sys
b = sys.argv[1]
def leer(p):
    try: return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    except FileNotFoundError: return []
filas = {r["id"]: r for r in leer(b + "/filas.jsonl")}
out = {}
for nom in ("l3_bateria", "l3_q4"):
    t = {}
    for r in leer(b + "/res/%s.jsonl" % nom):
        f = filas[r["id"]]; c = t.setdefault(f["conjunto"], [0, 0, 0])
        c[0] += 1; c[1] += r["valida"]
        c[2] += bool(r["valida"] and max(f["opciones"], key=lambda o: r["probs"][o]) == f["etiqueta"])
    out[nom] = {k: {"n": v[0], "validas": v[1], "acierto_bruto": round(v[2] / v[0], 4)} for k, v in sorted(t.items())}
fon = leer(b + "/res/l3_fondo.jsonl")
out["l3_fondo"] = {"n": len(fon), "validas": sum(r["valida"] for r in fon)}
print(json.dumps(out, ensure_ascii=False, indent=1))
EOF
estado resumen "$(tr -d '\n ' < "$OUT/resumen.json" | cut -c1-400)"
cierre
