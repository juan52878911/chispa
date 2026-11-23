#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-cuenta-aws/r11-alternativas-b0e2ea3
# R11: alternativas nuevas en las 6 tareas de DESARROLLO de la batería v2 (las selladas no se abren). Criterio congelado: model/alt/crit-r11.json (sha abajo).
#   A  Qwen/Qwen3-Embedding-0.6B (Apache-2.0) en lugar de e5-small: kNN con 32 y 8 vecinos y zero-shot por descripción; referencia e5 en esta máquina (código de T3a)
#   B  Qwen/Qwen3-Reranker-0.6B (Apache-2.0) como zero-shot: sí/no por etiqueta, logit de «yes»; referencias 4B zs y L2-zs (ref.tgz, de T3/T3a/T3b) y latencia
#   C  Horizon-Labs/multilingual-zeroshot-small: NO se mide (licencia: datos de entrenamiento con -SA); el ud no lo baja
#   +  informativo: la cascada de E con el kNN de Qwen3-Emb como L2 (solo numpy)
# Una g5.xlarge BAJO DEMANDA, tope duro CHISPA_MIN=90 min (≈ 1,5 USD). user-data (DLAMI GPU PyTorch AL2023, rol rol-ec2-pruebas sin ListBucket), terminate.
# Lee:     $CHISPA_S3/{codigo.tgz,dev.tgz,ref.tgz} por nombre exacto. Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado.
MIN=${CHISPA_MIN:-90}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-5}; HILOS=${CHISPA_HILOS:-4}
TRANSFORMERS=5.22.2
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; EMB_REV=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; RR_REV=e61197ed45024b0ed8a2d74b80b4d909f1255473
CRIT_SHA=6d48047802a67b6ebf6ba54964d6f15ad4e3a337cf9acab684503ce9bdcec8c7; N_TEST=2930; N_SOP=728; N_REF=9110
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r11-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r11}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r11-alternativas $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; PRED=$OUT/pred; REPO=$BASE/repo; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
medir() { [ -d "$BASE/ref" ] && ( cd "$REPO" && PYTHONPATH=$REPO timeout 300 "${PY:-python3}" -m model.alt.alternativas_r11 medir --dev "$BASE/dev" --pred "$PRED" --ref "$BASE/ref" --crit "$REPO/model/alt/crit-r11.json" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null; }
subir_y_apagar() {
  trap - EXIT; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; medir; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
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
FIN=$(( MIN - 6 - ESPERA_MIN ))   # minuto en que se corta el trabajo; los 6 + ESPERA_MIN finales son para medir, subir y esperar
( while sleep 20; do [ "$(quedan_min)" -le $((6 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt.alternativas_r11 (vec|op|knn|zs|rerank)"; pkill -TERM -f lineas_base_v2_pred; break; }; done ) & WPID=$!
limite() { local a=$(( $(date +%s) + $1 * 60 )) b=$(( T0 + (FIN - 1) * 60 )); echo $(( a < b ? a : b )); }
bajar() {   # bajar REPO REV DIR PATRONES...
  local r=$1 v=$2 d=$3; shift 3; local pat; pat=$(printf "'%s'," "$@")
  HF_HUB_ENABLE_HF_TRANSFER=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('$r', revision='$v', local_dir='$d', allow_patterns=[${pat%,}])"
}

# --- 1. entorno: el python de la DLAMI (torch con CUDA) + transformers fijado; se comprueba que importa todo ANTES de bajar nada
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; DISP=cpu
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -m pip install -q --no-cache-dir "transformers==$TRANSFORMERS" huggingface_hub numpy sentencepiece protobuf || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  DISP=cuda
fi
"$PY" -c "import torch, transformers, numpy, huggingface_hub, sentencepiece; assert transformers.__version__ == '$TRANSFORMERS'; assert '$DISP' == 'cpu' or torch.cuda.is_available()" \
  || { fallo "imports o versiones"; estado entorno "FALLO imports"; cierre; }
estado entorno "OK $("$PY" -c "import sys, torch, transformers; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)" 2>&1 | tail -1); $(nproc) vCPU; $DISP"

# --- 2. entradas por nombre exacto; criterio, recuentos y referencias se comprueban ANTES de medir
echo "== 2. entradas $(date -u +%T)"
for f in codigo.tgz dev.tgz ref.tgz; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$REPO" "$BASE/dev" "$BASE/ref" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev" && tar -xzf "$BASE/ref.tgz" -C "$BASE/ref" \
  || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/dev" "$BASE/ref" -name '._*' -delete; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae 6 jsonl"; estado entradas "FALLO dev"; cierre; }
SC=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO/model/alt/crit-r11.json")
[ "$SC" = "$CRIT_SHA" ] || { fallo "criterio distinto del congelado ($SC)"; estado entradas "FALLO criterio"; cierre; }
CN=$(cd "$REPO" && "$PY" -m model.alt.lineas_base_v2 filas --dev "$BASE/dev" --salida "$BASE/t")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
NR=$(cat "$BASE"/ref/*/*.jsonl | wc -l | tr -d ' ')
[ "$NR" = "$N_REF" ] || { fallo "ref.tgz trae $NR filas y no $N_REF"; estado entradas "FALLO ref"; cierre; }
( cd "$REPO" && "$PY" -c "import model.alt.alternativas_r11, model.alt.lineas_base_v2_pred, model.alt.cascada_sin_fondo, model.alt.fusion_knn_l2, model.l2.modelo" ) \
  || { fallo "imports del código"; estado entradas "FALLO imports código"; cierre; }
estado entradas "OK $CN; ref $NR filas; criterio ${SC:0:12}"
MAXF=(); [ "$LOCAL" = 1 ] && MAXF=(--max-filas "${CHISPA_LOCAL_MAX:-10}")

# --- 3. modelos por revisión fijada (en paralelo); e5 primero, que la referencia de A corre en CPU mientras la GPU trabaja
echo "== 3. modelos $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?}; EMB=${CHISPA_LOCAL_EMB:?}; RR=${CHISPA_LOCAL_RR:?}; touch "$BASE/e5.ok" "$BASE/emb.ok" "$BASE/rr.ok"
else
  export CHISPA_E5=$BASE/e5; EMB=$BASE/emb; RR=$BASE/rr
  ( bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' && echo ok > "$BASE/e5.ok" ) > "$BASE/bajada-e5.log" 2>&1 &
  ( bajar Qwen/Qwen3-Embedding-0.6B $EMB_REV "$BASE/emb" '*.json' '*.safetensors' '*.txt' 'merges.txt' 'vocab.json' 'tokenizer*' && echo ok > "$BASE/emb.ok" ) > "$BASE/bajada-emb.log" 2>&1 &
  ( bajar Qwen/Qwen3-Reranker-0.6B $RR_REV "$BASE/rr" '*.json' '*.safetensors' '*.txt' '*.jinja' 'merges.txt' 'vocab.json' 'tokenizer*' && echo ok > "$BASE/rr.ok" ) > "$BASE/bajada-rr.log" 2>&1 &
  for _ in $(seq 120); do [ -f "$BASE/e5.ok" ] && [ -f "$BASE/emb.ok" ] && [ -f "$BASE/rr.ok" ] && break; sleep 5; done
  cat "$BASE"/bajada-*.log | tail -5
fi
estado modelos "e5 $([ -f "$BASE/e5.ok" ] && echo ok || echo FALTA) emb $([ -f "$BASE/emb.ok" ] && echo ok || echo FALTA) rr $([ -f "$BASE/rr.ok" ] && echo ok || echo FALTA)"
export HF_HUB_OFFLINE=1   # desde aquí todo es local
alt() {   # alt NOMBRE MINUTOS modulo subcomando args...
  local n=$1 dur=$2 mod=$3; shift 3; local dl t0=$(date +%s); dl=$(limite "$dur")
  [ "$(quedan_min)" -gt $((6 + ESPERA_MIN + 1)) ] || { estado "$n" "OMITIDO: sin tiempo"; return 1; }
  ( cd "$REPO" && timeout $(( dl - t0 )) "$PY" -m "$mod" "$@" ) > "$OUT/pred-$n.log" 2>&1 \
    && estado "$n" "OK en $(( $(date +%s) - t0 )) s" || { tail -3 "$OUT/pred-$n.log"; fallo "$n"; estado "$n" "FALLO o corte"; return 1; }
}
TV=$("$PY" -c "import transformers; print(transformers.__version__)")

# referencia de A en CPU (el código de T3a) en segundo plano: vectores de e5, descripciones con e5, kNN y zero-shot
( if [ -f "$BASE/e5.ok" ]; then
    alt e5-vectores 25 model.alt.lineas_base_v2_pred vectores --dev "$BASE/dev" --vec "$BASE/vec-e5" --transformers "$TV" --hilos "$HILOS" \
      && alt e5-op 5 model.alt.alternativas_r11 op-e5 --dev "$BASE/dev" --vec "$BASE/vec-e5" \
      && for v in 32 8; do alt knn-e5-v$v 3 model.alt.alternativas_r11 knn --dev "$BASE/dev" --vec "$BASE/vec-e5" --vecinos $v --salida "$PRED/knn-e5-v$v" "${MAXF[@]}"; done \
      && alt zs-e5 3 model.alt.alternativas_r11 zs-emb --dev "$BASE/dev" --vec "$BASE/vec-e5" --salida "$PRED/zs-e5" "${MAXF[@]}"
  else estado e5 "FALLO: no se bajó"; fi ) &
E5PID=$!

# --- 4. A: Qwen3-Embedding en la GPU
echo "== 4. A $(date -u +%T) quedan $(quedan_min) min"
if [ -f "$BASE/emb.ok" ]; then
  alt vec-qwen 20 model.alt.alternativas_r11 vec-qwen --modelo "$EMB" --dev "$BASE/dev" --salida "$BASE/vec-qwen" --dispositivo "$DISP" "${MAXF[@]}" \
    && { mkdir -p "$PRED/vec-qwen"; cp "$BASE/vec-qwen/lat-emb.json" "$BASE/vec-qwen/tiempo.json" "$PRED/vec-qwen/"
         for v in 32 8; do alt knn-qwen-v$v 3 model.alt.alternativas_r11 knn --dev "$BASE/dev" --vec "$BASE/vec-qwen" --vecinos $v --salida "$PRED/knn-qwen-v$v" "${MAXF[@]}"; done
         alt zs-qwen 3 model.alt.alternativas_r11 zs-emb --dev "$BASE/dev" --vec "$BASE/vec-qwen" --salida "$PRED/zs-qwen" "${MAXF[@]}"; }
else fallo "emb no se bajó"; estado vec-qwen "FALLO: no se bajó"; fi

# --- 5. B: Qwen3-Reranker en la GPU, de la tarea más barata a la más cara, con fecha límite
echo "== 5. B $(date -u +%T) quedan $(quedan_min) min"
if [ -f "$BASE/rr.ok" ]; then
  alt rerank 60 model.alt.alternativas_r11 rerank --modelo "$RR" --dev "$BASE/dev" --salida "$PRED/rerank" --dispositivo "$DISP" --hasta "$(limite 58)" "${MAXF[@]}"
else fallo "rr no se bajó"; estado rerank "FALLO: no se bajó"; fi

# --- 6. la referencia de A y, si hay tiempo, la cascada de E con Qwen3-Emb como L2 (informativo; solo numpy)
echo "== 6. referencia e5 y cascada $(date -u +%T) quedan $(quedan_min) min"
wait $E5PID
[ "$LOCAL" = 1 ] && CASC=() || CASC=(--l3-extra "finanzas-documentos=$BASE/ref/zs-topk/finanzas-documentos.jsonl")
[ -d "$BASE/vec-qwen" ] && [ "$LOCAL" != 1 ] && alt cascada-qwen 5 model.alt.cascada_sin_fondo medir --dev "$BASE/dev" --vec "$BASE/vec-qwen" --l3 "$BASE/ref/4b-zs" "${CASC[@]}" --salida "$OUT/cascada-qwen.json"
cierre
