#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-879381269751/r10-t3-9ca7524
# R10-T3: líneas base de la batería v2, SOLO las 6 tareas de DESARROLLO (dev.tgz; las selladas no se abren aquí), en UNA g5.xlarge BAJO DEMANDA (A10G 24 GB),
# tope duro CHISPA_MIN=110 min (≈ 1,85 USD). user-data (DLAMI GPU PyTorch AL2023, rol chispa-pruebas-ec2 sin ListBucket), terminate. Sistemas en orden de
# prioridad, cada uno con su tope; lo hecho queda aunque se corte:
#   1 4b-zs  Qwen3-4B-Instruct-2507 por logprobs zero-shot (llm_logprobs.py)   2 4b-fs8  el mismo con 8 ejemplos en contexto
#   3 14b-awq-zs  Qwen3-14B-AWQ por logprobs zero-shot, sin pensar            4 mdeberta-nli  NLI zero-shot
#   5 l2-zs (checkpoint de Chispa, sin soporte) y knn-e5-k8
# Lectura congelada: model/alt/crit-r10-t3.json (su sha256 se comprueba ANTES de medir). Cifras: res/cifras.json (lineas_base_v2.py, solo numpy).
# Lee:     $CHISPA_S3/{codigo.tgz,dev.tgz} y s3://chispa-pruebas-879381269751/ajuste-80ab8ef/ckpt/mejor.pt (L2), todos por nombre exacto
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
MIN=${CHISPA_MIN:-110}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; VLLM=${CHISPA_VLLM:-0.33.0}
D1=${CHISPA_D1:-14}; D2=${CHISPA_D2:-18}; D3=${CHISPA_D3:-26}; D4=${CHISPA_D4:-10}; D5=${CHISPA_D5:-8}   # minutos máximos por sistema (el 3 incluye cargar el modelo)
REV4=cdbee75f17c01a7cc42f958dc650907174af0554; REV14=31c69efc29464b6bb0aee1398b5a7b50a99340c3; REVNLI=b5113eb38ab63efdd7f280f8c144ea8b13f978ce
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT_SHA=db6fdacba87fab217edee0e81c02f2729aaa0bcc1cee30c4cd29630d56c073bd; N_TEST=2930; N_SOP=728; CKPT=${CHISPA_CKPT:-s3://chispa-pruebas-879381269751/ajuste-80ab8ef/ckpt/mejor.pt}
ORDEN="horeca-resenas usajobs-nivel cfpb-quejas jql-valor-educativo congreso-eeuu-partido finanzas-documentos"
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-t3-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-t3}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-t3 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; PRED=$OUT/pred; REPO=$BASE/repo; FALLO=""; VP=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
medir() { [ -d "$BASE/dev" ] && ( cd "$REPO" && PYTHONPATH=$REPO timeout 120 "${PY:-python3}" -m model.alt.lineas_base_v2 medir --dev "$BASE/dev" --pred "$PRED" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; medir; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
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
FIN=$(( MIN - 5 - ESPERA_MIN ))   # minuto (desde T0) en que se corta el trabajo; los 5 + ESPERA_MIN finales son para medir, subir y esperar
( while sleep 30; do [ "$(quedan_min)" -le $((5 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_logprobs; pkill -TERM -f lineas_base_v2_pred; pkill -TERM -f vllm.entrypoints; break; }; done ) &
# fecha límite (epoch) de un sistema: DUR min desde ahora, sin pasar del minuto FIN
limite() { local a=$(( $(date +%s) + $1 * 60 )) b=$(( T0 + (FIN - 1) * 60 )); echo $(( a < b ? a : b )); }
# cliente con tope: corre hasta acabar o hasta la fecha límite $1 (epoch); lo escrito queda
cliente() {
  local dl=$1; shift
  ( cd "$REPO" && PYTHONPATH=$REPO "$VPY" -m model.alt.llm_logprobs "$@" ) & local p=$!
  while kill -0 $p 2>/dev/null; do
    if [ "$(date +%s)" -ge "$dl" ]; then echo "== corte por tiempo"; kill -TERM $p; sleep 5; kill -KILL $p 2>/dev/null; break; fi
    sleep 1
  done; wait $p 2>/dev/null
}
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
bajar() {   # bajar REPO REV DIR PATRONES...
  local r=$1 v=$2 d=$3; shift 3; local pat; pat=$(printf "'%s'," "$@")
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('$r', revision='$v', local_dir='$d', allow_patterns=[${pat%,}])"
}
# sistema por logprobs: lp SISTEMA MODELO_SERVIDO DURACION_MIN [opciones del cliente...]; una pasada por tarea, en orden de coste
lp() {
  local s=$1 mod=$2 dur=$3; shift 3; local dl t0=$(date +%s) t n=0; dl=$(limite "$dur"); mkdir -p "$PRED/$s"
  for t in $ORDEN; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "== $s: sin tiempo para $t"; continue; }
    local ej=(); case $s in *fs8*) ej=(--ejemplos "$BASE/t/soporte/$t.jsonl" --k 8);; esac
    cliente "$dl" --filas "$BASE/t/test/$t.jsonl" $(urls "$URL" "$HILOS") --modelo "$mod" --salida "$PRED/$s/$t.jsonl" --timeout 600 --reintentos 1 "${ej[@]}" "$@" "${MAX[@]}"
    n=$(( n + $(lineas "$PRED/$s/$t.jsonl") ))
  done
  estado "$s" "$n de $N_TEST filas ($(( $(date +%s) - t0 )) s; tope $dur min)"
}

# --- 1. entorno: venv con vLLM fijado y venv con transformers fijado (NLI, L2, e5) en segundo plano
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY; NPY=$PY; DISP=cpu
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python; NPY=/opt/nlp/bin/python; DISP=cuda
  ( "$PY" -m venv --system-site-packages /opt/nlp && /opt/nlp/bin/pip install -q --no-cache-dir "transformers==$TRANSFORMERS" sentencepiece protobuf tiktoken huggingface_hub numpy > "$BASE/pip-nlp.log" 2>&1 && echo ok > "$BASE/nlp.ok" ) &
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto; la lectura congelada se comprueba ANTES de medir nada
echo "== 2. entradas $(date -u +%T)"
for f in codigo.tgz dev.tgz; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$REPO" "$BASE/dev" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/dev/._*; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae 6 jsonl"; estado entradas "FALLO dev"; cierre; }
SC=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO/model/alt/crit-r10-t3.json")
[ "$SC" = "$CRIT_SHA" ] || { fallo "criterio distinto del congelado ($SC)"; estado entradas "FALLO criterio"; cierre; }
CN=$(cd "$REPO" && "$PY" -m model.alt.lineas_base_v2 filas --dev "$BASE/dev" --salida "$BASE/t")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
estado entradas "OK $CN; criterio ${SC:0:12}"
MAX=(); MAXF=(); [ "$LOCAL" = 1 ] && { MAX=(--max "${CHISPA_LOCAL_MAX:-10}"); MAXF=(--max-filas "${CHISPA_LOCAL_MAX:-10}"); }
"$PY" - "$BASE/t/test" "$BASE/auto.jsonl" <<'PY'
import glob, sys
open(sys.argv[2], "w").write("".join(open(f).readline() for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl"))))
PY

# --- 3. 4B: servidor, autoprueba, sistemas 1 y 2
echo "== 3. 4B $(date -u +%T) quedan $(quedan_min) min"
arrancar() {   # arrancar RUTA NOMBRE LARGO UTIL [args...]
  local ruta=$1 nom=$2 largo=$3 util=$4; shift 4
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$ruta" --served-model-name "$nom" --max-model-len "$largo" --gpu-memory-utilization "$util" \
    --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 "$@" > "$BASE/vllm-$nom.log" 2>&1 &
  VP=$!
  for _ in $(seq 180); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q "$nom"
}
probar() {   # probar SISTEMA: autoprueba de una fila por tarea
  local o="$OUT/auto-$1.jsonl" nv na
  ( cd "$REPO" && PYTHONPATH=$REPO "$VPY" -m model.alt.llm_logprobs --filas "$BASE/auto.jsonl" $(urls "$URL" 6) --modelo "$2" --salida "$o" --timeout 600 "${@:3}" )
  nv=$(validas "$o"); na=$(lineas "$o"); [ "$na" -gt 0 ] && [ $(( nv * 2 )) -ge "$na" ] || { head -c 600 "$o"; return 1; }
  estado "prueba-$1" "OK $nv/$na válidas"
}
URL=http://127.0.0.1:8000; MOD=qwen3-4b
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}; estado servidor "OMITIDO (local: $URL)"; OK4=1
else
  T1=$(date +%s); OK4=0
  if bajar Qwen/Qwen3-4B-Instruct-2507 $REV4 "$BASE/m4" '*.json' '*.safetensors' '*.txt' 'LICENSE*'; then
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
    arrancar "$BASE/m4" qwen3-4b 16384 0.85 --dtype bfloat16 && OK4=1
  fi
  if [ $OK4 = 1 ]; then estado servidor "OK Qwen3-4B-Instruct-2507@$REV4 en $(( $(date +%s) - T1 )) s"; else tail -5 "$BASE/vllm-qwen3-4b.log"; fallo "4B no arrancó"; estado servidor "FALLO 4B"; fi
  # en segundo plano mientras corre el 4B: el 14B AWQ, mDeBERTa y e5 (solo red y disco)
  ( bajar Qwen/Qwen3-14B-AWQ $REV14 "$BASE/m14" '*.json' '*.safetensors' '*.txt' 'LICENSE*' && echo ok > "$BASE/m14.ok"
    bajar MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 $REVNLI "$BASE/nli" '*.json' '*.safetensors' 'spm.model' 'tokenizer*' 'LICENSE*' && echo ok > "$BASE/nli.ok"
    bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' && echo ok > "$BASE/e5.ok" ) > "$BASE/bajadas.log" 2>&1 &
fi
if [ $OK4 = 1 ] && probar 4b "$MOD" "${MAX[@]}"; then
  lp 4b-zs "$MOD" "$D1"
  # latencia en serie (una petición a la vez) de las 15 primeras filas de cada tarea, solo del 4B zero-shot
  mkdir -p "$PRED/lat1-4b-zs"
  for t in $ORDEN; do head -n 15 "$BASE/t/test/$t.jsonl" > "$BASE/l15.jsonl"
    cliente "$(limite 2)" --filas "$BASE/l15.jsonl" --url "$URL" --modelo "$MOD" --salida "$PRED/lat1-4b-zs/$t.jsonl" --timeout 600 "${MAX[@]}"; done
  lp 4b-fs8 "$MOD" "$D2"
elif [ $OK4 = 1 ]; then fallo "autoprueba del 4B"; estado 4b "FALLO prueba"; fi
[ -n "$VP" ] && parar vllm.entrypoints

# --- 4. 14B AWQ (sin pensar, contexto 8192)
echo "== 4. 14B $(date -u +%T) quedan $(quedan_min) min"
OK14=0
if [ "$LOCAL" = 1 ]; then OK14=1
elif [ "$(quedan_min)" -gt $((5 + ESPERA_MIN + 12)) ]; then
  for _ in $(seq 60); do [ -f "$BASE/m14.ok" ] && break; sleep 10; done
  T1=$(date +%s)
  [ -f "$BASE/m14.ok" ] && arrancar "$BASE/m14" qwen3-14b 8192 0.90 --dtype half && OK14=1
  if [ $OK14 = 1 ]; then estado servidor14 "OK Qwen3-14B-AWQ@$REV14 en $(( $(date +%s) - T1 )) s"; else tail -5 "$BASE/vllm-qwen3-14b.log"; fallo "14B no arrancó"; estado servidor14 "FALLO"; fi
else estado 14b-awq-zs "OMITIDO: sin tiempo"; fi
MOD=qwen3-14b; [ "$LOCAL" = 1 ] && MOD=${CHISPA_LOCAL_MODELO:-q4}
if [ $OK14 = 1 ] && probar 14b "$MOD" --sin-pensar "${MAX[@]}"; then lp 14b-awq-zs "$MOD" "$D3" --sin-pensar
elif [ $OK14 = 1 ]; then fallo "autoprueba del 14B"; estado 14b "FALLO prueba"; fi
[ -n "$VP" ] && parar vllm.entrypoints

# --- 5. NLI, L2 y kNN (transformers fijado; GPU)
echo "== 5. NLI, L2, kNN $(date -u +%T) quedan $(quedan_min) min"
[ "$LOCAL" = 1 ] || for _ in $(seq 60); do [ -f "$BASE/nlp.ok" ] && break; sleep 10; done
if [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ]; then
  TV=$("$NPY" -c "import transformers; print(transformers.__version__)"); [ "$LOCAL" = 1 ] || [ "$TV" = "$TRANSFORMERS" ] || fallo "transformers $TV"
  export HF_HUB_OFFLINE=1; [ "$LOCAL" = 1 ] && { NLI=${CHISPA_LOCAL_NLI:-}; export CHISPA_E5=${CHISPA_LOCAL_E5:-}; CK=${CHISPA_LOCAL_M1:-}; } || { NLI=$BASE/nli; export CHISPA_E5=$BASE/e5; CK=$BASE/mejor.pt; }
  pred() { local n=$1 dur=$2; shift 2; local dl; dl=$(limite "$dur"); local t0=$(date +%s)   # pred SISTEMA MIN subcomando args...
    ( cd "$REPO" && timeout $(( dl - t0 )) "$NPY" -m model.alt.lineas_base_v2_pred "$@" ) > "$OUT/pred-$n.log" 2>&1 \
      && estado "$n" "OK en $(( $(date +%s) - t0 )) s" || { tail -3 "$OUT/pred-$n.log"; fallo "$n"; estado "$n" "FALLO o corte"; }; }
  if [ "$(quedan_min)" -gt $((5 + ESPERA_MIN + 4)) ]; then [ "$LOCAL" = 1 ] || [ -f "$BASE/nli.ok" ] && pred mdeberta-nli "$D4" nli --modelo "$NLI" --dev "$BASE/dev" --salida "$PRED/mdeberta-nli" --dispositivo $DISP "${MAXF[@]}"; fi
  if [ "$(quedan_min)" -gt $((5 + ESPERA_MIN + 4)) ]; then
    [ "$LOCAL" = 1 ] || { [ -f "$BASE/mejor.pt" ] || s3cp "$CKPT" "$BASE/mejor.pt"; }
    [ -f "$CK" ] && pred l2-zs "$D5" l2 --m1 "$CK" --dev "$BASE/dev" --salida "$PRED/l2-zs" --dispositivo $DISP "${MAXF[@]}" || estado l2-zs "FALLO: sin checkpoint"
  fi
  if [ "$(quedan_min)" -gt $((5 + ESPERA_MIN + 3)) ]; then
    pred knn-e5-k8 "$D5" vectores --dev "$BASE/dev" --vec "$BASE/vec" --transformers "$TV" --hilos 4 && pred knn-e5-k8 "$D5" knn --dev "$BASE/dev" --vec "$BASE/vec" --salida "$PRED/knn-e5-k8" "${MAXF[@]}"
  fi
else fallo "pip del venv de transformers"; estado nlp "FALLO pip"; fi
cierre
