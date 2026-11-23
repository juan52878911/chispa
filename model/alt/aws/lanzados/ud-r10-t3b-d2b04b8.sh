#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-cuenta-aws/r10-t3b-d2b04b8
# R10-T3b: variantes de ESPECIFICACIÓN del 4B (Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f) por logprobs en las 6 tareas de DESARROLLO de la batería v2 (solo dev.tgz;
# las selladas no se abren), en UNA g5.xlarge BAJO DEMANDA (A10G 24 GB), tope duro CHISPA_MIN=45 min (≈ 0,76 USD). user-data (DLAMI GPU PyTorch AL2023, rol
# rol-ec2-pruebas sin ListBucket), terminate. Lectura congelada: model/alt/crit-r10-t3b.json (su sha256 y el de la rúbrica y el de las predicciones de T3
# se comprueban ANTES de medir). Corre, por orden de prioridad y cada una con su tope (lo hecho queda aunque se corte):
#   zs-sop / rub-sop  zero-shot sin y con rúbrica sobre el SOPORTE de oro de las 3 score (para ajustar los umbrales)    rub  zero-shot con la rúbrica en el test de las 3 score
#   zs-topk   4B zero-shot en finanzas con las 10 candidatas del kNN-e5      fs8-topk  8 ejemplos en finanzas con las 10 candidatas      fs8rub  8 ejemplos + rúbrica en las 3 score
# La esperanza ordinal, los umbrales (isotónica sobre el soporte), QWK/MAE y la composición con las predicciones de T3 (4b-zs, 4b-fs8) se hacen al medir
# (lineas_base_v2b.py, solo numpy, sin GPU). Cada fila guarda las probabilidades de TODOS los niveles (probs de llm_logprobs.py).
# Lee:     $CHISPA_S3/{codigo.tgz,dev.tgz,t3.tgz}   (t3.tgz = 4b-zs/ y 4b-fs8/ de la ronda T3)
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar. Cifras: res/cifras.json.
MIN=${CHISPA_MIN:-45}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; VLLM=${CHISPA_VLLM:-0.33.0}
DS=${CHISPA_DS:-2}; D2=${CHISPA_D2:-6}; D1=${CHISPA_D1:-3}; D4=${CHISPA_D4:-3}; D3=${CHISPA_D3:-6}   # minutos máximos de cada -sop, rub, zs-topk, fs8-topk, fs8rub
REV4=cdbee75f17c01a7cc42f958dc650907174af0554; E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TORCH=2.7.1; TRANSFORMERS=5.22.2
CRIT_SHA=823cf37adf04d635ab201cee40b44d8abc699f7082c4f07c79c01bb360083cfb; N_TEST=2930; N_SOP=728
SCORE="horeca-resenas jql-valor-educativo usajobs-nivel"; FIN=finanzas-documentos
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-t3b-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-t3b}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-t3b $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; PRED=$OUT/pred; REPO=$BASE/repo; FALLO=""; VP=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
medir() { [ -d "$BASE/dev" ] && [ -d "$BASE/t3" ] && ( cd "$REPO" && PYTHONPATH=$REPO timeout 120 "${NPY:-python3}" -m model.alt.lineas_base_v2b medir --dev "$BASE/dev" --pred "$PRED" --t3 "$BASE/t3" \
  --presel "$BASE/t/topk/preseleccion.json" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null; }
subir_y_apagar() {
  trap - EXIT; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; medir; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
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
  HF_HUB_ENABLE_HF_TRANSFER=0 "$NPY" -c "from huggingface_hub import snapshot_download as s; s('$r', revision='$v', local_dir='$d', allow_patterns=[${pat%,}])"
}
# sistema por logprobs: lp SISTEMA MIN DIR_FILAS "TAREAS" [DIR_EJEMPLOS]; una pasada por tarea
lp() {
  local s=$1 dur=$2 fdir=$3 tareas=$4 edir=${5:-} dl t0=$(date +%s) t n=0; dl=$(limite "$dur"); mkdir -p "$PRED/$s"
  for t in $tareas; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "== $s: sin tiempo para $t"; continue; }
    local ej=(); [ -n "$edir" ] && ej=(--ejemplos "$edir/$t.jsonl" --k 8)
    cliente "$dl" --filas "$fdir/$t.jsonl" $(urls "$URL" "$HILOS") --modelo "$MOD" --salida "$PRED/$s/$t.jsonl" --timeout 600 --reintentos 1 "${ej[@]}" "${MAX[@]}"
    n=$(( n + $(lineas "$PRED/$s/$t.jsonl") ))
  done
  estado "$s" "$n filas ($(( $(date +%s) - t0 )) s; tope $dur min)"
}

# --- 1. entorno: vLLM fijado (venv) y un venv con Python >= 3.10, torch CPU y transformers fijados (e5 para el kNN de la preselección)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY; NPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python; NPY=/opt/nlp/bin/python
  ( for c in python3.12 python3.11 python3.10 "$PY" python3; do p=$(command -v "$c" 2>/dev/null) && "$p" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && "$p" -m venv /opt/nlp 2>/dev/null && break; done
    /opt/nlp/bin/pip install -q --no-cache-dir "torch==$TORCH" --index-url https://download.pytorch.org/whl/cpu \
      && /opt/nlp/bin/pip install -q --no-cache-dir "transformers==$TRANSFORMERS" sentencepiece protobuf tiktoken numpy huggingface_hub \
      && /opt/nlp/bin/python -c "import torch, transformers; assert torch.__version__.startswith('$TORCH') and transformers.__version__ == '$TRANSFORMERS'" && echo ok > "$BASE/nlp.ok" ) > "$BASE/pip-nlp.log" 2>&1 &
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto; criterio, rúbrica y predicciones de T3 se comprueban ANTES de medir nada
echo "== 2. entradas $(date -u +%T)"
for f in codigo.tgz dev.tgz t3.tgz; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$REPO" "$BASE/dev" "$BASE/t3" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev" && tar -xzf "$BASE/t3.tgz" -C "$BASE/t3" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/dev/._* "$BASE"/t3/._*; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae 6 jsonl"; estado entradas "FALLO dev"; cierre; }
"$PY" - "$REPO/model/alt/crit-r10-t3b.json" "$REPO/model/alt/rubricas-r10-t3b.json" "$BASE/t3" "$CRIT_SHA" <<'PY' || { fallo "criterio, rúbrica o predicciones de T3 distintos de los congelados"; estado entradas "FALLO criterio"; cierre; }
import hashlib, json, sys
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
crit = json.load(open(sys.argv[1]))
mal = [] if h(sys.argv[1]) == sys.argv[4] else ["criterio"]
mal += [] if h(sys.argv[2]) == crit["rubricas_sha256"] else ["rubricas"]
mal += [f for f, s in crit["entradas_t3_sha256"].items() if h(sys.argv[3] + "/" + f) != s]
print("MAL:", mal) if mal else print("criterio, rúbrica y %d ficheros de T3 OK" % len(crit["entradas_t3_sha256"]))
sys.exit(1 if mal else 0)
PY
CN=$(cd "$REPO" && "$PY" -m model.alt.lineas_base_v2 filas --dev "$BASE/dev" --salida "$BASE/t")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
( cd "$REPO" && "$PY" -m model.alt.lineas_base_v2b rubrica --dev "$BASE/dev" --rubricas "$REPO/model/alt/rubricas-r10-t3b.json" --salida "$BASE/t" ) || { fallo "rúbrica"; estado entradas "FALLO rubrica"; cierre; }
estado entradas "OK $CN; criterio ${CRIT_SHA:0:12}"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-10}")
"$PY" - "$BASE/t/test" "$BASE/auto.jsonl" <<'PY'
import glob, sys
open(sys.argv[2], "w").write("".join(open(f).readline() for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl"))))
PY

# --- 3. preselección top-k de finanzas (e5 en CPU, en segundo plano mientras el 4B se descarga y arranca)
presel() {
  for _ in $(seq 120); do [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ] && break; sleep 5; done
  [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ] || { echo "sin venv de transformers"; return 1; }
  if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?}; else
    bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' || return 1; export CHISPA_E5=$BASE/e5; fi
  export HF_HUB_OFFLINE=1; local TV; TV=$("$NPY" -c "import transformers; print(transformers.__version__)")
  cd "$REPO" && "$NPY" -m model.alt.lineas_base_v2_pred vectores --dev "$BASE/dev" --vec "$BASE/vec" --transformers "$TV" --hilos 4 --solo $FIN \
    && "$NPY" -m model.alt.lineas_base_v2_pred topk --dev "$BASE/dev" --vec "$BASE/vec" --tarea $FIN --salida "$BASE/t/topk" && echo ok > "$BASE/topk.ok"
}
presel > "$BASE/presel.log" 2>&1 & PP=$!

# --- 4. el 4B: servidor, autoprueba y sistemas
echo "== 4. 4B $(date -u +%T) quedan $(quedan_min) min"
URL=http://127.0.0.1:8000; MOD=qwen3-4b; OK4=0
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}; OK4=1; estado servidor "OMITIDO (local: $URL)"
else
  T1=$(date +%s)
  if bajar Qwen/Qwen3-4B-Instruct-2507 $REV4 "$BASE/m4" '*.json' '*.safetensors' '*.txt' 'LICENSE*'; then
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda d36f8ab)
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
  if [ -f "$BASE/topk.ok" ]; then estado preseleccion "OK $(cat "$BASE/t/topk/preseleccion.json" | tr -d '\n ')"; else tail -3 "$BASE/presel.log"; fallo "preselección top-k"; estado preseleccion "FALLO"; fi
  lp zs-sop "$DS" "$BASE/t/soporte" "$SCORE"
  lp rub-sop "$DS" "$BASE/t/rub/soporte" "$SCORE"
  lp rub "$D2" "$BASE/t/rub/test" "$SCORE"
  [ -f "$BASE/topk.ok" ] && lp zs-topk "$D1" "$BASE/t/topk/test" $FIN
  [ -f "$BASE/topk.ok" ] && lp fs8-topk "$D4" "$BASE/t/topk/test" $FIN "$BASE/t/topk/soporte"
  lp fs8rub "$D3" "$BASE/t/rub/test" "$SCORE" "$BASE/t/soporte"
fi
[ -n "$VP" ] && parar vllm.entrypoints
cierre
