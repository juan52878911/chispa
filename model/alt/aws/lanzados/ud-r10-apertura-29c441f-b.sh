#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-879381269751/r10-apertura-29c441f-b
# R10-apertura: la apertura ÚNICA de las 8 tareas SELLADAS de la batería v2 con el 4B (Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f) por logprobs, en DOS g5.xlarge BAJO DEMANDA
# en paralelo (PARTE a y b, cada una con su ud y su prefijo S3), tope duro CHISPA_MIN (a y b: 110 min ≈ 1,84 USD cada una). user-data (DLAMI GPU PyTorch AL2023,
# rol chispa-pruebas-ec2 sin ListBucket), terminate. Lectura congelada: model/alt/crit-r10-apertura.json (su sha256 y el de cada fichero de datos se comprueban ANTES de
# computar). Nadie lee filas selladas: filas y rúbrica MECÁNICA (apertura_v2.py preparar) salen del soporte dentro de la instancia. En modo `ensayo` los datos son dev.
#   PARTE a: rub-sop y rub (score: final), fs8-knn (choice: final); kNN-e5 k=8 y mayoritaria en CPU (la mayoritaria al medir)
#   PARTE b: zs (score y choice), luego fs8-azar (score y choice)   Orden: primero el final y el 4B zs; luego fs8 al azar; kNN y mayoritaria, baratos, al final.
# Lee:     $CHISPA_S3/{codigo.tgz,selladas.tgz}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar. Cifras de la parte: res/cifras.json (solo vale la
# unión de a y b: apertura_v2.py medir --pred A --pred B).
PARTE=b; MODO=real
MIN_DEF=110
MIN=${CHISPA_MIN:-$MIN_DEF}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; VLLM=${CHISPA_VLLM:-0.33.0}
CRS=${CHISPA_CRS:-3}; CR=${CHISPA_CR:-8}; CFK=${CHISPA_CFK:-60}; CZS=${CHISPA_CZS:-12}; CZC=${CHISPA_CZC:-20}; CAS=${CHISPA_CAS:-15}; CAC=${CHISPA_CAC:-25}   # minutos máximos de rub-sop, rub, fs8-knn, zs score, zs choice, fs8-azar score, fs8-azar choice
REV4=cdbee75f17c01a7cc42f958dc650907174af0554; E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TORCH=2.7.1; TRANSFORMERS=5.22.2
CRIT_SHA=a79d33a9712d4c5a02228509e951ecc3764235b6d3a56afe8503edbc0b77cc9d; N_TAREAS=8; N_TEST=2955; N_SOP=572
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-ap-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-ap}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-apertura parte=$PARTE modo=$MODO $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; PRED=$OUT/pred; REPO=$BASE/repo; FALLO=""; VP=""; TOPK=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
medir() { [ -d "$BASE/datos" ] && ( cd "$REPO" && PYTHONPATH=$REPO timeout 300 "${VPY:-python3}" -m model.alt.apertura_v2 medir --datos "$BASE/datos" --pred "$PRED" \
  --crit "$REPO/model/alt/crit-r10-apertura.json" --modo "$MODO" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null; }
subir_y_apagar() {
  trap - EXIT; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; medir; cp "$BASE/t/rubrica-mecanica.json" "$OUT/" 2>/dev/null; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
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
CIERRE=$(( 5 + ESPERA_MIN )); FINM=$(( MIN - CIERRE ))   # minuto en que se corta el trabajo
( while sleep 30; do [ "$(quedan_min)" -le $CIERRE ] && { echo "== tope: se cierra"; pkill -TERM -f llm_logprobs; pkill -TERM -f lineas_base_v2_pred; pkill -TERM -f vllm.entrypoints; break; }; done ) & WPID=$!
limite() { local a=$(( $(date +%s) + $1 * 60 )) b=$(( T0 + (FINM - 1) * 60 )); echo $(( a < b ? a : b )); }
cliente() {   # corre hasta acabar o hasta la fecha límite $1 (epoch); lo escrito queda
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
# sistema por logprobs: lp SISTEMA MIN SUBDIR "TAREAS" [SUBDIR_EJEMPLOS [DIR_DEMOS]]; filas de $BASE/t/SUBDIR; una pasada por tarea; las de >20 opciones (TOPK) usan las 10 candidatas
lp() {
  local s=$1 dur=$2 sub=$3 tareas=$4 esub=${5:-} ddir=${6:-} dl t0=$(date +%s) t n=0; dl=$(limite "$dur"); mkdir -p "$PRED/$s"
  for t in $tareas; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "== $s: sin tiempo para $t"; continue; }
    local fd=$BASE/t/$sub ed=$BASE/t/$esub ej=() mx=("${MAX[@]}"); [ "${s%-sop}" != "$s" ] && mx=()
    case " $TOPK " in *" $t "*) case $sub in test|soporte) fd=$BASE/t/topk/$sub;; esac; ed=$BASE/t/topk/$esub;; esac
    [ -n "$esub" ] && ej=(--ejemplos "$ed/$t.jsonl" --k 8); [ -n "$ddir" ] && ej+=(--demos "$ddir/$t.jsonl")
    cliente "$dl" --filas "$fd/$t.jsonl" $(urls "$URL" "$HILOS") --modelo "$MOD" --salida "$PRED/$s/$t.jsonl" --timeout 600 --reintentos 1 "${ej[@]}" "${mx[@]}"
    n=$(( n + $(lineas "$PRED/$s/$t.jsonl") ))
  done
  estado "$s" "$n filas ($(( $(date +%s) - t0 )) s; tope $dur min)"
}

# --- 1. entorno: vLLM fijado (venv) y un venv con Python >= 3.10, torch CPU y transformers fijados (e5 para el kNN y la preselección)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY; NPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  VPY=/opt/vllm/bin/python; NPY=/opt/nlp/bin/python
  ( for c in python3.12 python3.11 python3.10 "$PY" python3; do p=$(command -v "$c" 2>/dev/null) && "$p" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && "$p" -m venv /opt/nlp 2>/dev/null && break; done
    /opt/nlp/bin/pip install -q --no-cache-dir "torch==$TORCH" --index-url https://download.pytorch.org/whl/cpu \
      && /opt/nlp/bin/pip install -q --no-cache-dir "transformers==$TRANSFORMERS" sentencepiece protobuf tiktoken numpy huggingface_hub \
      && /opt/nlp/bin/python -c "import torch, transformers, numpy, huggingface_hub, sentencepiece; assert torch.__version__.startswith('$TORCH') and transformers.__version__ == '$TRANSFORMERS'" && echo ok > "$BASE/nlp.ok" || echo fallo > "$BASE/nlp.fail" ) > "$BASE/pip-nlp.log" 2>&1 &
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python
  "$VPY" -c "import vllm, numpy, huggingface_hub" || { fallo "el venv de vLLM no importa vllm, numpy y huggingface_hub"; estado entorno "FALLO venv vllm"; cierre; }
  for _ in $(seq 120); do [ -f "$BASE/nlp.ok" ] || [ -f "$BASE/nlp.fail" ] && break; sleep 5; done
  [ -f "$BASE/nlp.ok" ] || { tail -3 "$BASE/pip-nlp.log"; fallo "el venv de transformers no importa torch, transformers, numpy, huggingface_hub y sentencepiece"; estado entorno "FALLO venv nlp"; cierre; }
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto; el criterio y cada fichero de datos se comprueban por sha256 ANTES de computar nada
echo "== 2. entradas $(date -u +%T)"
for f in codigo.tgz selladas.tgz; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$REPO" "$BASE/datos" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/selladas.tgz" -C "$BASE/datos" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/datos/._*; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"; AP="$VPY -m model.alt.apertura_v2"; CRIT=$REPO/model/alt/crit-r10-apertura.json
( cd "$REPO" && $AP verificar --crit "$CRIT" --crit-sha "$CRIT_SHA" --datos "$BASE/datos" --modo "$MODO" ) || { fallo "criterio o datos distintos de los congelados"; estado entradas "FALLO criterio"; cierre; }
CN=$(cd "$REPO" && $AP preparar --datos "$BASE/datos" --salida "$BASE/t" --crit "$CRIT" --modo "$MODO") || { fallo "preparar: $CN"; estado entradas "FALLO preparar"; cierre; }
[ "$CN" = "tareas=$N_TAREAS test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
SCORE=$(cat "$BASE/t/score.txt"); CHOICE=$(cat "$BASE/t/choice.txt"); TOPK=$(cat "$BASE/t/topk.txt")
estado entradas "OK $CN; criterio ${CRIT_SHA:0:12}; score [$SCORE] choice [$CHOICE] topk [$TOPK]"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-10}")
"$VPY" - "$BASE/t/test" "$BASE/auto.jsonl" <<'PY'
import glob, sys
open(sys.argv[2], "w").write("".join(open(f).readline() for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl"))))
PY

# --- 3. CPU en segundo plano mientras el 4B se descarga y arranca: preselección top-k (si alguna choice pasa de 20 opciones) y, en la parte a, vectores de todo, kNN-e5 k=8 y demostraciones por fila
presel() {
  [ -z "$TOPK" ] && echo ok > "$BASE/topk.ok"
  [ "$PARTE" = a ] || [ -n "$TOPK" ] || return 0
  for _ in $(seq 120); do [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ] && break; sleep 5; done
  [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ] || { echo "sin venv de transformers"; return 1; }
  if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?}; else
    bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' || return 1; export CHISPA_E5=$BASE/e5; fi
  export HF_HUB_OFFLINE=1; local TV P t; TV=$("$NPY" -c "import transformers; print(transformers.__version__)"); P="$NPY -m model.alt.lineas_base_v2_pred"; cd "$REPO" || return 1
  if [ -n "$TOPK" ]; then
    $P vectores --dev "$BASE/datos" --vec "$BASE/vec-topk" --transformers "$TV" --hilos 4 --solo "${TOPK// /,}" || return 1
    for t in $TOPK; do $P topk --dev "$BASE/datos" --vec "$BASE/vec-topk" --tarea "$t" --salida "$BASE/t/topk" || return 1; done; echo ok > "$BASE/topk.ok"
  fi
  [ "$PARTE" = a ] && $P vectores --dev "$BASE/datos" --vec "$BASE/vec" --transformers "$TV" --hilos 4 && $P knn --dev "$BASE/datos" --vec "$BASE/vec" --salida "$PRED/knn-e5-k8" \
    && $P demos --dev "$BASE/datos" --vec "$BASE/vec" --salida "$BASE/t/demos" --k 8 && echo ok > "$BASE/demos.ok"
}
presel > "$BASE/presel.log" 2>&1 & PP=$!

# --- 4. el 4B: servidor, autoprueba y sistemas
echo "== 4. 4B $(date -u +%T) quedan $(quedan_min) min"
URL=http://127.0.0.1:8000; MOD=qwen3-4b; OK4=0
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}; OK4=1; estado servidor "OMITIDO (local: $URL)"
else
  T1=$(date +%s)
  if bajar Qwen/Qwen3-4B-Instruct-2507 $REV4 "$BASE/m4" '*.json' '*.safetensors' '*.txt' 'LICENSE*'; then
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
    nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/m4" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
      --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-qwen3-4b.log" 2>&1 &
    VP=$!
    for _ in $(seq 180); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
    curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b && OK4=1
  fi
  if [ $OK4 = 1 ]; then estado servidor "OK Qwen3-4B-Instruct-2507@$REV4 en $(( $(date +%s) - T1 )) s"; else tail -5 "$BASE/vllm-qwen3-4b.log"; fallo "4B no arrancó"; estado servidor "FALLO 4B"; fi
fi
if [ $OK4 = 1 ]; then
  ( cd "$REPO" && PYTHONPATH=$REPO "$VPY" -m model.alt.llm_logprobs --filas "$BASE/auto.jsonl" $(urls "$URL" 6) --modelo "$MOD" --salida "$OUT/auto-4b.jsonl" --timeout 600 "${MAX[@]}" )
  NV=$(validas "$OUT/auto-4b.jsonl"); NA=$(lineas "$OUT/auto-4b.jsonl")
  if [ "$NA" -gt 0 ] && [ $(( NV * 2 )) -ge "$NA" ]; then estado prueba "OK $NV/$NA válidas"; else head -c 600 "$OUT/auto-4b.jsonl"; fallo "autoprueba: $NV de $NA válidas"; estado prueba "FALLO"; OK4=0; fi
fi
if [ $OK4 = 1 ]; then
  for _ in $(seq 120); do [ -f "$BASE/topk.ok" ] && break; kill -0 $PP 2>/dev/null || break; sleep 5; done
  [ -f "$BASE/topk.ok" ] && estado preseleccion "OK topk [$TOPK]" || { tail -3 "$BASE/presel.log"; fallo "preselección top-k"; estado preseleccion "FALLO"; }
  if [ "$PARTE" = a ]; then
    lp rub-sop "$CRS" rub/soporte "$SCORE"   # sistema final, score: la rúbrica mecánica sobre el soporte (umbrales) y sobre el test
    lp rub "$CR" rub/test "$SCORE"
    for _ in $(seq 240); do [ -f "$BASE/demos.ok" ] && break; kill -0 $PP 2>/dev/null || break; sleep 5; done
    if [ -f "$BASE/demos.ok" ] && [ -f "$BASE/topk.ok" ]; then lp fs8-knn "$CFK" test "$CHOICE" soporte "$BASE/t/demos"   # sistema final, choice
    else tail -3 "$BASE/presel.log"; fallo "vectores, kNN, demostraciones o preselección"; estado demos "FALLO"; fi
  else
    lp zs "$CZS" test "$SCORE"   # 4B zero-shot (necesario para el veredicto): primero en la parte b
    [ -f "$BASE/topk.ok" ] && lp zs "$CZC" test "$CHOICE"
    lp fs8-azar "$CAS" test "$SCORE" soporte   # 8 al azar, semilla 20261115
    [ -f "$BASE/topk.ok" ] && lp fs8-azar "$CAC" test "$CHOICE" soporte
  fi
fi
[ -n "$VP" ] && parar vllm.entrypoints
cierre
