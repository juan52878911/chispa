#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# Ronda R5: el LLM puntuado por logprobs en vez de pedir un JSON (llm_logprobs.py; llm_jevbench.py solo aporta el prompt). Qwen/Qwen3-1.7B @ 70d244cc,
# Qwen/Qwen3-0.6B @ c1899de2 y, de referencia del método, Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f (todos Apache-2.0), en bf16 con vLLM 0.33.0
# (--max-logprobs 20, contexto 16 k, 32 hilos) sobre la misma filas.jsonl del paso 2. Los dos Qwen3 híbridos piden el modo sin pensamiento de su
# plantilla (--sin-pensar); el 4B-Instruct-2507 no es híbrido y va sin esa bandera. g5.xlarge BAJO DEMANDA, tope duro CHISPA_MIN=52 min (≈ 0,87 USD)
# user-data (DLAMI GPU PyTorch AL2023, rol rol-ec2-pruebas sin ListBucket), --instance-initiated-shutdown-behavior terminate.
# Lee:     $CHISPA_S3/{llm.tgz,filas.jsonl}   (llm.tgz lleva llm_logprobs.py y llm_jevbench.py)
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Por modelo, en orden (1.7B, 0.6B, 4B): autoprueba (una fila por tarea), latencia con UN hilo en lat.jsonl (10 filas por tarea) y la muestra
# entera con 32 hilos, con tope de MOD_MIN minutos por modelo. Los pesos de los tres se bajan en segundo plano mientras corre el primero.
# La medición (IC, pares con el 4B y criterio) se hace fuera: `llm_bateria pequeno`.
MIN=${CHISPA_MIN:-52}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; MOD_MIN=${CHISPA_MOD_MIN:-12}; VLLM=${CHISPA_VLLM:-0.33.0}
MODELOS="qwen3-1.7b=Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e qwen3-0.6b=Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca qwen3-4b=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554"
N_FILAS=__N_FILAS__
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-logprobs-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-logprobs}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque llm-logprobs $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min por modelo=${MOD_MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log; do [ -f "$l" ] && tail -n 200 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
  ( cd "$BASE" && tar -czf resultados.tgz res ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 30; do [ "$(quedan_min)" -le $((5 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_logprobs; pkill -TERM -f vllm.entrypoints; break; }; done ) &
# cliente con tope: corre hasta acabar, hasta que queden $1 minutos o hasta que pasen $2 minutos; lo escrito queda
cliente() {
  local queda=$1 dura=$2; shift 2; local t=$(date +%s)
  "$VPY" "$BASE/codigo/llm_logprobs.py" "$@" & local p=$!
  while kill -0 $p 2>/dev/null; do
    if [ "$(quedan_min)" -le "$queda" ] || [ $(( ($(date +%s) - t) / 60 )) -ge "$dura" ]; then echo "== corte por tiempo"; kill -TERM $p; sleep 5; kill -KILL $p 2>/dev/null; break; fi
    sleep 10
  done; wait $p 2>/dev/null
}
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

# --- 1. entorno: venv con vLLM fijado (el del paso 2)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto; autoprueba (primera fila de cada tarea) y lat.jsonl (10 primeras de cada tarea)
echo "== 2. entradas $(date -u +%T)"
for f in llm.tgz filas.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$BASE/codigo" && tar -xzf "$BASE/llm.tgz" -C "$BASE/codigo" || { fallo "llm.tgz"; cierre; }
for c in llm_logprobs.py llm_jevbench.py; do [ -f "$BASE/codigo/$c" ] || { fallo "llm.tgz no trae $c"; estado entradas "FALLO $c"; cierre; }; done
grep -q -- "--sin-pensar" "$BASE/codigo/llm_logprobs.py" || { fallo "el cliente no tiene --sin-pensar"; estado entradas "FALLO cliente"; cierre; }
[ "$(lineas "$BASE/filas.jsonl")" = "$N_FILAS" ] || { fallo "filas.jsonl no tiene $N_FILAS filas"; estado entradas "FALLO recuento"; cierre; }
"$PY" -c "
import json,sys
v={}
for l in open(sys.argv[1]):
    r=json.loads(l); v.setdefault(r['conjunto'], []).append(l)
open(sys.argv[2],'w').write(''.join(v[k][0] for k in sorted(v)))
open(sys.argv[3],'w').write(''.join(l for k in sorted(v) for l in v[k][:10]))" "$BASE/filas.jsonl" "$BASE/auto.jsonl" "$BASE/lat.jsonl"
estado entradas "OK $N_FILAS filas, $(lineas "$BASE/auto.jsonl") de autoprueba, $(lineas "$BASE/lat.jsonl") de latencia"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-2}")
if [ "$LOCAL" != 1 ]; then   # los pesos de los tres, en segundo plano y en orden; cada uno deja m-NOM.ok (o m-NOM.mal)
  ( for par in $MODELOS; do
      NOM=${par%%=*}; REPO_REV=${par#*=}; HF=${REPO_REV%@*}; REV=${REPO_REV#*@}
      HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('$HF', revision='$REV', local_dir='$BASE/m-$NOM', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" \
        && touch "$BASE/m-$NOM.ok" || touch "$BASE/m-$NOM.mal"
    done ) &
fi

# --- 3. cada modelo
for par in $MODELOS; do
  NOM=${par%%=*}; REPO_REV=${par#*=}; HF=${REPO_REV%@*}; REV=${REPO_REV#*@}
  echo "== 3. $NOM $(date -u +%T) quedan $(quedan_min) min"
  if [ "$(quedan_min)" -le $((5 + ESPERA_MIN + 3)) ]; then estado "$NOM" "OMITIDO (sin tiempo)"; continue; fi
  T1=$(date +%s)
  if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}
  else
    URL=http://127.0.0.1:8000; MOD=$NOM
    for _ in $(seq 150); do [ -f "$BASE/m-$NOM.ok" ] || [ -f "$BASE/m-$NOM.mal" ] && break; sleep 5; done
    [ -f "$BASE/m-$NOM.ok" ] || { fallo "$NOM: descarga"; estado "$NOM" "FALLO descarga"; continue; }
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda d36f8ab)
    nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/m-$NOM" --served-model-name "$NOM" --dtype bfloat16 --max-model-len 16384 \
      --max-logprobs 20 --gpu-memory-utilization 0.85 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-$NOM.log" 2>&1 &
    VP=$!
    for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
    curl -s -m 3 localhost:8000/v1/models | grep -q "$NOM" || { tail -5 "$BASE/vllm-$NOM.log"; fallo "$NOM: vLLM no arrancó"; estado "$NOM" "FALLO servidor"; parar vllm.entrypoints; continue; }
  fi
  SP=(--sin-pensar); case $NOM in qwen3-4b*) SP=();; esac   # el 4B-Instruct-2507 no es híbrido
  estado "$NOM" "servidor OK $HF@$REV en $(( $(date +%s) - T1 )) s"
  "$VPY" "$BASE/codigo/llm_logprobs.py" --filas "$BASE/auto.jsonl" $(urls $URL 6) --modelo $MOD --salida "$OUT/auto-$NOM.jsonl" --timeout 600 ${SP[@]+"${SP[@]}"} ${MAX[@]+"${MAX[@]}"}
  NV=$(validas "$OUT/auto-$NOM.jsonl"); NA=$(lineas "$OUT/auto-$NOM.jsonl")
  if [ "$NA" -eq 0 ] || [ "$NV" -eq 0 ]; then head -c 600 "$OUT/auto-$NOM.jsonl"; fallo "$NOM: autoprueba $NV de $NA válidas"; estado "$NOM" "FALLO autoprueba"; [ "$LOCAL" = 1 ] || parar vllm.entrypoints; continue; fi
  estado "$NOM" "autoprueba $NV/$NA válidas"
  T2=$(date +%s)
  cliente $((5 + ESPERA_MIN + 1)) 4 --filas "$BASE/lat.jsonl" --url $URL --modelo $MOD --salida "$OUT/lat-$NOM.jsonl" --timeout 600 --reintentos 1 ${SP[@]+"${SP[@]}"} ${MAX[@]+"${MAX[@]}"}
  estado "$NOM" "latencia un hilo: $(lineas "$OUT/lat-$NOM.jsonl") casos en $(( $(date +%s) - T2 )) s"
  T2=$(date +%s); DURA=$(( MOD_MIN - (T2 - T1) / 60 )); [ "$DURA" -lt 3 ] && DURA=3
  cliente $((5 + ESPERA_MIN + 1)) "$DURA" --filas "$BASE/filas.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/l3-$NOM.jsonl" --timeout 600 --reintentos 1 ${SP[@]+"${SP[@]}"} ${MAX[@]+"${MAX[@]}"}
  estado "$NOM" "$(lineas "$OUT/l3-$NOM.jsonl") de $N_FILAS casos ($(validas "$OUT/l3-$NOM.jsonl") válidas) en $(( $(date +%s) - T2 )) s"
  [ "$LOCAL" = 1 ] || { nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/memoria-$NOM.txt" 2>/dev/null; parar vllm.entrypoints; }
done

# --- 4. resumen sin dependencias (la medición con IC y pares con el 4B se hace fuera)
"$PY" - "$BASE" > "$OUT/resumen.json" <<'EOF'
import glob, json, os, sys
b = sys.argv[1]
filas = {r["id"]: r for r in (json.loads(l) for l in open(b + "/filas.jsonl", encoding="utf-8") if l.strip())}
out = {}
for p in sorted(glob.glob(b + "/res/l3-*.jsonl")):
    t = {}
    for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()):
        f = filas[r["id"]]; c = t.setdefault(f["conjunto"], [0, 0, 0])
        c[0] += 1; c[1] += r["valida"]
        c[2] += bool(r["valida"] and max(f["opciones"], key=lambda o: r["probs"][o]) == f["etiqueta"])
    out[os.path.basename(p)[3:-6]] = {k: {"n": v[0], "validas": v[1], "acierto_bruto": round(v[2] / v[0], 4)} for k, v in sorted(t.items())}
print(json.dumps(out, ensure_ascii=False, indent=1))
EOF
estado resumen "$(tr -d '\n ' < "$OUT/resumen.json" | cut -c1-400)"
cierre
