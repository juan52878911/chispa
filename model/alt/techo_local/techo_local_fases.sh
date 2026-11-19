#!/bin/bash
# Techo local: fases de la medida (las carga con `.` techo_local.sh tras bajar y comprobar el sha256 de codigo.tgz y dev.tgz; comparte sus variables y funciones: BASE, OUT, PRED, LOCAL,
# MIN, ESPERA_MIN, HILOS, VLLM, D4/D30/DOSS, MODELOS, REV*, E5_REV, TRANSFORMERS, T0, estado, fallo, quedan_min, s3cp, parar, cierre, VP).
# Criterio congelado: model/alt/crit-techo-local.json. Cada modelo deja sus predicciones en res/pred/<modelo>/<sistema>/<tarea>.jsonl y sus bench-*.json; las cifras y el veredicto
# los calcula `python -m model.alt.techo_local medir` al cerrar.
# Modo local (CHISPA_LOCAL=1, con un vLLM falso): CHISPA_PY, CHISPA_LOCAL_URL (un servidor para los tres modelos), CHISPA_LOCAL_E5 (e5 diminuto), CHISPA_LOCAL_RUBRICAS (por defecto «mecanica»),
# CHISPA_LOCAL_N (filas del bench), CHISPA_LOCAL_LIC_<MODELO en mayúsculas con _> (licencia que declara la tarjeta falsa de ese modelo).
fases_main() {
CH=$BASE/codigo; REPO=$CH; T=$BASE/t; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"; mkdir -p "$OUT/licencias" "$PRED"
FIN=$(( MIN - 5 - ESPERA_MIN ))   # minuto (desde T0) en que se corta el trabajo; los 5 + ESPERA_MIN finales son para medir, subir y esperar
limite() { local a=$(( $(date +%s) + $1 * 60 )) b=$(( T0 + (FIN - 1) * 60 )); echo $(( a < b ? a : b )); }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
# cliente con tope: corre hasta acabar o hasta la fecha límite $1 (epoch); lo escrito queda
cliente() {
  local dl=$1; shift
  ( cd "$REPO" && "$VPY" -m model.alt.techo_local_cliente "$@" ) & local p=$!
  while kill -0 $p 2>/dev/null; do
    if [ "$(date +%s)" -ge "$dl" ]; then echo "== corte por tiempo"; kill -TERM $p; sleep 5; kill -KILL $p 2>/dev/null; break; fi
    sleep 1
  done; wait $p 2>/dev/null
}
bajar() {   # bajar REPO REV DIR PATRONES... (los pesos de la revisión fijada; sin las carpetas original/ y metal/ de gpt-oss)
  local r=$1 v=$2 d=$3; shift 3; local pat; pat=$(printf "'%s'," "$@")
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('$r', revision='$v', local_dir='$d', allow_patterns=[${pat%,}], ignore_patterns=['original/*','metal/*'])"
}

# --- 2. entorno: se comprueba que importa todo ANTES de bajar nada
echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY; NPY=$PY; MAX=(); NB=${CHISPA_LOCAL_N:-20}; NBS=4
else
  NB=240; NBS=20; MAX=()
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 120 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 120)"; estado entorno "FALLO disco"; cierre; }
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch, numpy" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python; NPY=/opt/nlp/bin/python
  ( "$PY" -m venv --system-site-packages /opt/nlp && /opt/nlp/bin/pip install -q --no-cache-dir "transformers==$TRANSFORMERS" sentencepiece protobuf tiktoken huggingface_hub numpy > "$BASE/pip-nlp.log" 2>&1 && echo ok > "$BASE/nlp.ok" ) &
  GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $GPU; $LIBRES GB libres"
fi
"$PY" -c "import numpy, model.alt.techo_local, model.alt.lineas_base_v2_pred, model.alt.lineas_base_v2b, model.alt.auditoria" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
"$VPY" -c "import model.alt.techo_local_cliente, model.alt.llm_logprobs" || { fallo "imports del cliente"; estado entorno "FALLO imports cliente"; cierre; }

# --- 3. entradas: recuentos, rúbrica de T3b (sha256 del criterio), filas, partición A/B
echo "== 3. filas $(date -u +%T)"
RUB=$REPO/model/alt/rubricas-r10-t3b.json; RUBSHA=$("$PY" -c "import json; print(json.load(open('$REPO/model/alt/crit-techo-local.json'))['sistemas']['rubricas_sha256'])")
if [ "$LOCAL" = 1 ]; then RUB=${CHISPA_LOCAL_RUBRICAS:-mecanica}; else sha_ok "$RUBSHA" "$RUB" || { fallo "rúbrica distinta de la congelada"; estado entradas "FALLO rúbrica"; cierre; }; fi
( cd "$REPO" && $PY -m model.alt.techo_local preparar --dev "$BASE/dev" --salida "$T" --rubricas "$RUB" --umbral-opciones "${CHISPA_TOPK_UMBRAL:-20}" ) > "$BASE/preparar.txt" 2>&1; CN=$(tail -1 "$BASE/preparar.txt")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { cat "$BASE/preparar.txt"; fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
CHO=$(cat "$T/choice.txt"); SCO=$(cat "$T/score.txt"); TK=$(cat "$T/topk.txt")
for p in test rub/test; do [ -d "$T/$p" ] && ( cd "$REPO" && $PY -m model.alt.techo_local dividir --origen "$T/$p" --destino-b "$T/${p}B" --destino-a "$T/${p}A" ) | sed "s#^#$p #"; done
estado entradas "OK $CN; choice: $CHO | score: $SCO | topk: ${TK:-ninguna}"
"$PY" - "$T/test" "$BASE/auto.jsonl" <<'PY'
import glob, sys
open(sys.argv[2], "w").write("".join(open(f).readline() for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl"))))
PY

# --- 4. licencias: la tarjeta de cada modelo en su revisión fijada, ANTES de bajar los pesos; si no es Apache-2.0 o MIT, el candidato no se mide
echo "== 4. licencias $(date -u +%T)"
meta() {
  case $1 in
    qwen3-4b) M_REPO=Qwen/Qwen3-4B-Instruct-2507; M_REV=$REV4; M_LARGO=16384; M_UTIL=0.85; M_ARGS="--dtype bfloat16"; M_CAP=$D4; M_FMT=chat; M_PORT=;;
    qwen3-30b-a3b) M_REPO=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8; M_REV=$REV30; M_LARGO=16384; M_UTIL=0.92; M_ARGS=""; M_CAP=$D30; M_FMT=chat; M_PORT=;;
    gpt-oss-20b) M_REPO=openai/gpt-oss-20b; M_REV=$REVOSS; M_LARGO=16384; M_UTIL=0.90; M_ARGS=""; M_CAP=$DOSS; M_FMT=harmony; M_PORT=;;
    *) return 1;;
  esac
}
declare -A PERMITIDO
for M in qwen3-4b qwen3-30b-a3b gpt-oss-20b; do
  meta $M; L=$BASE/lic/$M; mkdir -p "$L"
  if [ "$LOCAL" = 1 ]; then v=CHISPA_LOCAL_LIC_$(echo "$M" | tr 'a-z-' 'A-Z_'); printf -- '---\nlicense: %s\n---\n# %s (tarjeta falsa)\n' "${!v:-apache-2.0}" "$M" > "$L/README.md"; echo "texto de licencia falso" > "$L/LICENSE"
  else
    "$VPY" -c "
from huggingface_hub import hf_hub_download as h
h('$M_REPO', 'README.md', revision='$M_REV', local_dir='$L')
try: h('$M_REPO', 'LICENSE', revision='$M_REV', local_dir='$L')
except Exception as e: print('sin LICENSE aparte:', str(e)[:80])
" || { fallo "tarjeta de $M"; estado "licencia-$M" "FALLO al bajar la tarjeta"; PERMITIDO[$M]=0; continue; }
  fi
  cp "$L/README.md" "$OUT/licencias/$M-README.md"; [ -f "$L/LICENSE" ] && cp "$L/LICENSE" "$OUT/licencias/$M-LICENSE"
  ( cd "$REPO" && $PY -m model.alt.techo_local licencia --readme "$L/README.md" --licencia-fichero "$L/LICENSE" --modelo "$M" --salida "$OUT/licencias/$M.json" ) > "$BASE/lic-$M.txt" 2>&1; rc=$?
  PERMITIDO[$M]=$(( rc == 0 )); estado "licencia-$M" "$(cat "$BASE/lic-$M.txt") (rev ${M_REV:0:12})"
done

# --- 5. CPU en segundo plano mientras se baja y arranca el primer modelo: preselección top-k, vectores de e5, demostraciones por fila y partición A/B de lo que depende de ellas
presel() {
  [ "$LOCAL" = 1 ] || for _ in $(seq 120); do [ -f "$BASE/nlp.ok" ] && break; sleep 5; done
  [ "$LOCAL" = 1 ] || [ -f "$BASE/nlp.ok" ] || { echo "sin venv de transformers"; return 1; }
  if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?}; else bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' || return 1; export CHISPA_E5=$BASE/e5; fi
  export HF_HUB_OFFLINE=1; local TV P t; TV=$("$NPY" -c "import transformers; print(transformers.__version__)"); [ "$LOCAL" = 1 ] || [ "$TV" = "$TRANSFORMERS" ] || { echo "transformers $TV"; return 1; }
  P="$NPY -m model.alt.lineas_base_v2_pred"; cd "$REPO" || return 1
  for t in $TK; do $P vectores --dev "$BASE/dev" --vec "$BASE/vec" --transformers "$TV" --hilos 4 --solo "$t" && $P topk --dev "$BASE/dev" --vec "$BASE/vec" --tarea "$t" --salida "$T/topk" || return 1; done
  if [ -n "$TK" ]; then $PY -m model.alt.techo_local dividir --origen "$T/topk/test" --destino-b "$T/topk/testB" --destino-a "$T/topk/testA" || return 1; fi
  echo ok > "$BASE/topk.ok"
  $P vectores --dev "$BASE/dev" --vec "$BASE/vec" --transformers "$TV" --hilos 4 && $P demos --dev "$BASE/dev" --vec "$BASE/vec" --salida "$T/demos" --k 8 && echo ok > "$BASE/demos.ok"
}
presel > "$BASE/presel.log" 2>&1 & PP=$!
esperar() {   # esperar MARCADOR SEGUNDOS: hasta que exista $BASE/MARCADOR, el proceso de CPU muera o pasen los segundos
  local i; for i in $(seq $(( $2 / 5 ))); do [ -f "$BASE/$1" ] && return 0; kill -0 $PP 2>/dev/null || { [ -f "$BASE/$1" ]; return; }; sleep 5; done; return 1
}

# --- 6. los modelos, uno tras otro, en la misma instancia
URL=http://127.0.0.1:8000
bajar_modelo() {   # bajar_modelo M: pesos en la revisión fijada + sha256 de cada fichero (en segundo plano)
  meta "$1"; local d=$BASE/m-$1
  bajar "$M_REPO" "$M_REV" "$d" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' 'chat_template*' && echo ok > "$BASE/m-$1.ok" \
    && ( cd "$d" && find . -type f -not -path './.cache/*' | sort | xargs sha256sum > "$OUT/pesos-sha256-$1.txt" )
}
if [ "$LOCAL" != 1 ]; then ( for M in $MODELOS; do [ "${PERMITIDO[$M]:-0}" = 1 ] || [ "$M" = qwen3-4b ] || continue; bajar_modelo "$M"; done ) > "$BASE/bajadas.log" 2>&1 & fi
lp() {   # lp SISTEMA ETAPA(B|A|"") DEADLINE tareas...
  local s=$1 et=$2 dl=$3; shift 3; local t f d ej
  for t in "$@"; do
    [ "$(date +%s)" -ge "$dl" ] && { echo "== $M $s $et: sin tiempo para $t"; continue; }
    d=$T; case " $TK " in *" $t "*) d=$T/topk;; esac; ej=()
    case $s in
      zs) f=$d/test$et/$t.jsonl;;
      fs8-knn) f=$d/test$et/$t.jsonl; ej=(--ejemplos "$d/soporte/$t.jsonl" --k 8 --demos "$T/demos/$t.jsonl");;
      rub) f=$T/rub/test$et/$t.jsonl;;
      rub-sop) f=$T/rub/soporte/$t.jsonl;;
    esac
    [ -f "$f" ] || { echo "== $M $s $et: no hay $f"; continue; }
    mkdir -p "$PRED/$M/$s"
    cliente "$dl" --formato "$M_FMT" --filas "$f" $(urls "$URL" "$HILOS") --modelo "$M" --salida "$PRED/$M/$s/$t.jsonl" --timeout 600 --reintentos 1 "${ej[@]}" "${MAX[@]}"
  done
}
bench() {   # bench CONDICION CONCURRENCIA N
  [ "$(quedan_min)" -gt $((5 + ESPERA_MIN + 2)) ] || return 0
  ( cd "$REPO" && timeout 600 $PY -m model.alt.techo_local bench --t "$T" --url "$URL" --modelo "$M" --condicion "$1" --concurrencia "$2" --n "$3" --formato "$M_FMT" --salida "$PRED/$M/bench-$1-c$2.json" ) 2>&1 | tail -2
}
arrancar() {
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/m-$M" --served-model-name "$M" --max-model-len "$M_LARGO" --gpu-memory-utilization "$M_UTIL" \
    --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 $M_ARGS > "$BASE/vllm-$M.log" 2>&1 &
  VP=$!
  for _ in $(seq 240); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q "$M"
}
probar() {   # autoprueba: una fila por tarea, al menos la mitad válidas
  local o="$OUT/auto-$M.jsonl" nv na
  ( cd "$REPO" && "$VPY" -m model.alt.techo_local_cliente --formato "$M_FMT" --filas "$BASE/auto.jsonl" $(urls "$URL" 6) --modelo "$M" --salida "$o" --timeout 600 )
  nv=$(validas "$o"); na=$(lineas "$o"); [ "$na" -gt 0 ] && [ $(( nv * 2 )) -ge "$na" ] || { head -c 600 "$o"; return 1; }
  estado "prueba-$M" "OK $nv/$na válidas"
}
medir_modelo() {
  M=$1; meta "$M" || return
  if [ "${PERMITIDO[$M]:-0}" != 1 ] && [ "$M" != qwen3-4b ]; then estado "$M" "NO SE MIDE: la licencia de la tarjeta no es Apache-2.0 ni MIT (o no se pudo leer)"; return; fi
  if [ "$(quedan_min)" -le $((5 + ESPERA_MIN + 14)) ]; then estado "$M" "OMITIDO: sin tiempo (quedan $(quedan_min) min)"; return; fi
  if [ "$LOCAL" != 1 ] && [ "$M" = qwen3-30b-a3b ]; then
    MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); [ "${MIB:-0}" -ge 40000 ] || { estado "$M" "OMITIDO: GPU de ${MIB} MiB, el FP8 de 31 GB no cabe"; return; }
  fi
  echo "== modelo $M $(date -u +%T) quedan $(quedan_min) min"
  local t1=$(date +%s) dl; dl=$(limite "$M_CAP")
  if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; estado "servidor-$M" "OMITIDO (local: $URL)"
  else
    for _ in $(seq 120); do [ -f "$BASE/m-$M.ok" ] && break; sleep 10; done
    [ -f "$BASE/m-$M.ok" ] || { tail -3 "$BASE/bajadas.log"; fallo "descarga de $M"; estado "servidor-$M" "FALLO descarga"; return; }
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
    [ "$M" = gpt-oss-20b ] && export VLLM_MXFP4_USE_MARLIN=1   # MXFP4 en una GPU Ada (L40S) sin kernels de Hopper; sin verificar con este vLLM
    URL=http://127.0.0.1:8000
    if arrancar; then estado "servidor-$M" "OK $M_REPO@${M_REV:0:12} en $(( $(date +%s) - t1 )) s ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
    else tail -5 "$BASE/vllm-$M.log"; fallo "$M no arrancó"; estado "servidor-$M" "FALLO"; [ -n "$VP" ] && parar vllm.entrypoints; return; fi
    [ "$M_FMT" = harmony ] && "$NPY" -m model.alt.techo_local_cliente --verificar-harmony "$BASE/m-$M" "$OUT/harmony-check.json" 2>&1 | tail -2
  fi
  esperar topk.ok 1800 || { tail -3 "$BASE/presel.log"; fallo "preselección top-k"; estado preseleccion "FALLO"; [ -n "$VP" ] && parar vllm.entrypoints; return; }
  if probar; then
    # 1. latencia en serie y caudal de zs (servidor recién arrancado, filas de A sin ver); 2. zs en B; 3. fs8 (espera las demos): latencia, caudal, y choice en B; 4. rúbrica en B; 5. todo en A
    bench zs 1 $NBS; bench zs 32 $NB
    lp zs B "$dl" $CHO $SCO
    esperar demos.ok 1800 || { tail -3 "$BASE/presel.log"; fallo "vectores o demostraciones"; estado demos "FALLO"; }
    if [ -f "$BASE/demos.ok" ]; then bench fs8-knn 1 $NBS; bench fs8-knn 32 $NB; lp fs8-knn B "$dl" $CHO; fi
    lp rub-sop "" "$dl" $SCO; lp rub B "$dl" $SCO
    lp zs A "$dl" $CHO $SCO; [ -f "$BASE/demos.ok" ] && lp fs8-knn A "$dl" $CHO; lp rub A "$dl" $SCO
    estado "$M" "hecho en $(( $(date +%s) - t1 )) s (tope $M_CAP min); filas: $(cat "$PRED/$M"/*/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  else fallo "autoprueba de $M"; estado "$M" "FALLO prueba"; fi
  [ -n "$VP" ] && parar vllm.entrypoints; VP=""
}
for M in $MODELOS; do medir_modelo "$M"; done
estado fases "fin de los modelos; quedan $(quedan_min) min"
}
