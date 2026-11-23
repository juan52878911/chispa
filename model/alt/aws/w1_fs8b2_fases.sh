#!/bin/bash
# W1d fase B2: la carga con `.` el user-data w1_fs8b2.sh tras comprobar el sha256 de codigo.tgz, resultados.tgz de B y faseA.tar (comparte BASE, OUT, LOCAL, MIN, REV4, VLLM, PY, estado, fallo, hay_tiempo, cierre,
# sha_ok, juntar, medir, VP, ESPERA_MIN). Modo local (CHISPA_LOCAL=1, prueba en seco con vLLM falso): CHISPA_PY, CHISPA_LOCAL_URL.
fases_main() {
CH=$BASE/codigo; ENT=$BASE/ent; NUEVO=$BASE/faltan; HILOS=${CHISPA_HILOS:-4}; URL=${CHISPA_LOCAL_URL:-http://127.0.0.1:8000}; export PYTHONPATH=$CH

# --- 2. vLLM + Qwen3-4B (revisión fijada) en segundo plano, igual que B: venv con el torch de la DLAMI y `uv pip` (si vllm exige otro torch, uv instala el suyo DENTRO de /opt/vllm). ANTES de lanzar vLLM se
#        comprueba que el torch de ese venv carga y ve la GPU; si no, cierre con motivo.
if [ "$LOCAL" != 1 ]; then
  PYG=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PYG=$c; break; }; done
  [ -n "$PYG" ] && "$PYG" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado vllm "FALLO"; cierre; }
  VPY=/opt/vllm/bin/python
  arrancar_vllm() {
    "$PYG" -m venv --system-site-packages /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir uv huggingface_hub || return 1
    echo "torch antes: $($VPY -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
    ( HF_HUB_ENABLE_HF_TRANSFER=0 $VPY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV4', local_dir='$BASE/modelo4b', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" ) & HFP=$!
    /opt/vllm/bin/uv pip install -q --python $VPY "vllm==$VLLM" || { echo "uv falló: pip"; /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" || { kill $HFP 2>/dev/null; return 3; }; }
    echo "torch después: $($VPY -c 'import torch; print(torch.__version__, torch.__file__)' 2>&1 | tail -1)"
    $VPY -c "import torch, vllm; assert torch.cuda.is_available()" || { kill $HFP 2>/dev/null; return 6; }   # el torch del venv de vLLM carga y ve la GPU
    wait $HFP || return 2
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda d36f8ab)
    nohup $VPY -m vllm.entrypoints.openai.api_server --model "$BASE/modelo4b" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
      --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
    echo $! > "$BASE/vllm.pid"
    for _ in $(seq 180); do curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b && return 0; kill -0 "$(cat "$BASE/vllm.pid")" 2>/dev/null || return 4; sleep 5; done; return 5
  }
  T1=$(date +%s)
  ( arrancar_vllm > "$BASE/vllm-arranque.log" 2>&1; echo $? > "$BASE/vllm.rc" ) &
fi

# --- 3. entorno Python con numpy (como B)
echo "== 3. entorno $(date -u +%T)"
if [ "$LOCAL" != 1 ]; then
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 60 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 60)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip gcc tar gzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
fi
"$PY" -c "import model.alt.w1_juntar, model.alt.w1_medir, model.alt.llm_logprobs, numpy" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }

# --- 4. lo de B (solo res/) y lo de A (solo ent/, con el sha256 de cada fichero de faseA.json); qué filas faltan y con qué demostraciones
echo "== 4. filas que faltan $(date -u +%T)"
J=(env PYTHONPATH=$CH "$PY" -m model.alt.w1_juntar)
( cd "$CH" && "${J[@]}" extraer --res-tgz "$BASE/in/resultados.tgz" --fasea-tar "$BASE/in/faseA.tar" --fasea-json "$BASE/in/faseA.json" --salida "$BASE/B" ) > "$BASE/extraer.out" 2>&1 || { tail -3 "$BASE/extraer.out"; fallo "extraer: $(tail -1 "$BASE/extraer.out")"; estado entradas "FALLO extraer"; cierre; }
mv "$BASE/B/ent" "$ENT"; rm -rf "$BASE/B/faseA"
( cd "$CH" && "${J[@]}" faltan --res "$BASE/B/res" --ent "$ENT" --salida "$NUEVO" ) > "$BASE/faltan.out" 2>&1 || { tail -3 "$BASE/faltan.out"; fallo "faltan: $(tail -1 "$BASE/faltan.out")"; estado faltan "FALLO"; cierre; }
estado faltan "$(cat "$BASE/faltan.out")"
TAREAS=$("$PY" -I -c "import json;print(' '.join(json.load(open('$NUEVO/plan.json'))['orden']))")
cp "$NUEVO/plan.json" "$OUT/"

# --- 5. vLLM listo
if [ "$LOCAL" != 1 ]; then
  for _ in $(seq 600); do [ -f "$BASE/vllm.rc" ] && break; sleep 2; done
  RC=$(cat "$BASE/vllm.rc" 2>/dev/null || echo 9)
  [ "$RC" = 0 ] || { tail -5 "$BASE/vllm-arranque.log"; tail -5 "$BASE/vllm-4b.log"; [ "$RC" = 6 ] && fallo "el torch del venv de vLLM no carga o no ve la GPU" || fallo "vLLM no arrancó (rc=$RC)"; estado vllm "FALLO rc=$RC"; cierre; }
  VP=$(cat "$BASE/vllm.pid")
  estado vllm "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null) Qwen3-4B-Instruct-2507@${REV4:0:12} en $(( $(date +%s) - T1 )) s; $(grep '^torch' "$BASE/vllm-arranque.log" | tr '\n' ';'); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 6. el fs8-kNN de Python SOLO sobre las filas que faltan (mismas ids, mismas demostraciones que el servidor); misma orden que B. Tras cada tarea se juntan: si el tope corta, lo hecho queda en py/
echo "== 6. fs8-kNN de Python $(date -u +%T) quedan $(quedan_min) min"
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
T3=$(date +%s); NP=0
for t in $TAREAS; do
  hay_tiempo 2 || { fallo "sin tiempo para $t en Python"; continue; }
  T4=$(date +%s)
  ( cd "$CH" && "$PY" -m model.alt.llm_logprobs --filas "$NUEVO/test-demos/$t.jsonl" $(urls "$URL" "$HILOS") --modelo qwen3-4b --salida "$OUT/py-nuevo/$t.jsonl" --timeout 600 --reintentos 1 \
      --ejemplos "$ENT/soporte/$t.jsonl" --demos "$NUEVO/demos/$t.jsonl" --k 8 ) > "$BASE/py-$t.log" 2>&1 || { tail -3 "$BASE/py-$t.log"; fallo "llm_logprobs $t"; }
  N=$(wc -l < "$OUT/py-nuevo/$t.jsonl" 2>/dev/null || echo 0); NP=$(( NP + N ))
  estado "py-$t" "$N de $(wc -l < "$NUEVO/test-demos/$t.jsonl" | tr -d ' ') filas ($(grep -c '"valida": true' "$OUT/py-nuevo/$t.jsonl" 2>/dev/null || echo 0) válidas) en $(( $(date +%s) - T4 )) s"
  mkdir -p "$OUT/py"; juntar > "$BASE/junta.out" 2>&1 || fallo "juntar tras $t: $(tail -1 "$BASE/junta.out")"
done
estado fs8-knn-python "$NP filas nuevas en $(( $(date +%s) - T3 )) s"
cierre
}
