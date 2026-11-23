#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R10-T3a: lo que T3 no pudo medir (el venv no tenía torch): kNN-e5 k=8, L2 zero-shot (mejor.pt) y mDeBERTa-NLI zero-shot, sobre las 6 tareas de DESARROLLO
# de la batería v2, en CPU: c7i.2xlarge (8 vCPU) BAJO DEMANDA, tope duro CHISPA_MIN=40 min (≈ 0,28 USD). user-data (DLAMI o AL2023, rol rol-ec2-pruebas
# sin ListBucket), terminate. Solo dev.tgz: las selladas no se abren. Cada sistema es independiente de los demás (si uno falla, los otros siguen):
#   1 knn-e5-k8 (vectores de e5 con transformers fijado + kNN)   2 l2-zs (checkpoint de Chispa, sin soporte)   3 mdeberta-nli (el más caro: va el último,
#   de lo barato a lo caro por tarea, con tope de filas por tarea CHISPA_NLI_FILAS y tope de tiempo; lo no puntuado cuenta como fallo y se informa la cobertura)
# Entorno: venv propio con Python >= 3.10, torch==2.7.1 (rueda CPU de download.pytorch.org/whl/cpu; transformers 5 exige torch >= 2.6) y transformers==5.22.2.
# Lee:     $CHISPA_S3/{codigo.tgz,dev.tgz} y s3://chispa-pruebas-cuenta-aws/ajuste-4a2e745/ckpt/mejor.pt, por nombre exacto
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar. Cifras: res/cifras.json (lineas_base_v2.py).
MIN=${CHISPA_MIN:-40}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-5}; HILOS=${CHISPA_HILOS:-8}; NLI_FILAS=${CHISPA_NLI_FILAS:-__NLI_FILAS__}
TORCH=2.7.1; TRANSFORMERS=5.22.2
REVNLI=b5113eb38ab63efdd7f280f8c144ea8b13f978ce; E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3
CRIT_SHA=__CRIT_SHA__; N_TEST=__N_TEST__; N_SOP=__N_SOP__; CKPT=${CHISPA_CKPT:-s3://chispa-pruebas-cuenta-aws/ajuste-4a2e745/ckpt/mejor.pt}
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-t3a-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-t3a}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-t3a $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS nli_filas=$NLI_FILAS"
OUT=$BASE/res; PRED=$OUT/pred; REPO=$BASE/repo; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
medir() { [ -d "$BASE/dev" ] && ( cd "$REPO" && PYTHONPATH=$REPO timeout 120 "${NPY:-python3}" -m model.alt.lineas_base_v2 medir --dev "$BASE/dev" --pred "$PRED" --crit "$REPO/model/alt/crit-r10-t3a.json" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null; }
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
FIN=$(( MIN - 4 - ESPERA_MIN ))   # minuto en que se corta el trabajo; los 4 + ESPERA_MIN finales son para medir, subir y esperar
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f lineas_base_v2_pred; break; }; done ) & WPID=$!
limite() { local a=$(( $(date +%s) + $1 * 60 )) b=$(( T0 + (FIN - 1) * 60 )); echo $(( a < b ? a : b )); }
bajar() {   # bajar REPO REV DIR PATRONES...
  local r=$1 v=$2 d=$3; shift 3; local pat; pat=$(printf "'%s'," "$@")
  HF_HUB_ENABLE_HF_TRANSFER=0 "$NPY" -c "from huggingface_hub import snapshot_download as s; s('$r', revision='$v', local_dir='$d', allow_patterns=[${pat%,}])"
}

# --- 1. entorno: venv con Python >= 3.10, torch CPU fijado y transformers fijado
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then NPY=${CHISPA_PY:-python3}; BF16=()
else
  PY=""; for c in python3.12 python3.11 python3.10 /opt/pytorch/bin/python python3; do p=$(command -v "$c" 2>/dev/null) && "$p" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null && "$p" -m venv /opt/nlp 2>/dev/null && { PY=$p; break; }; done
  [ -n "$PY" ] || { fallo "sin Python >= 3.10 con venv"; estado entorno "FALLO python"; cierre; }
  NPY=/opt/nlp/bin/python
  { /opt/nlp/bin/pip install -q --no-cache-dir "torch==$TORCH" --index-url https://download.pytorch.org/whl/cpu \
    && /opt/nlp/bin/pip install -q --no-cache-dir "transformers==$TRANSFORMERS" sentencepiece protobuf tiktoken numpy huggingface_hub; } || { fallo "pip torch/transformers"; estado entorno "FALLO pip"; cierre; }
  "$NPY" -c "import torch, transformers, numpy, huggingface_hub, sentencepiece; assert torch.__version__.startswith('$TORCH') and transformers.__version__ == '$TRANSFORMERS'; print(torch.__version__, transformers.__version__)" || { fallo "versiones"; estado entorno "FALLO versiones"; cierre; }
  BF16=(); grep -q amx_bf16 /proc/cpuinfo && BF16=(--bf16)
  estado entorno "OK $("$NPY" -c 'import sys, torch, transformers; print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)'); $(nproc) vCPU; nli ${BF16[*]:-fp32}"
fi

# --- 2. entradas por nombre exacto; la lectura congelada (el criterio de T3) se comprueba ANTES de medir
echo "== 2. entradas $(date -u +%T)"
for f in codigo.tgz dev.tgz; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$REPO" "$BASE/dev" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/dev/._*; export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae 6 jsonl"; estado entradas "FALLO dev"; cierre; }
SC=$("$NPY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO/model/alt/crit-r10-t3a.json")
[ "$SC" = "$CRIT_SHA" ] || { fallo "criterio distinto del congelado ($SC)"; estado entradas "FALLO criterio"; cierre; }
CN=$(cd "$REPO" && "$NPY" -m model.alt.lineas_base_v2 filas --dev "$BASE/dev" --salida "$BASE/t")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos distintos: $CN"; estado entradas "FALLO recuentos"; cierre; }
estado entradas "OK $CN; criterio ${SC:0:12}"
MAXF=(); [ "$LOCAL" = 1 ] && MAXF=(--max-filas "${CHISPA_LOCAL_MAX:-10}")
if [ "$LOCAL" = 1 ]; then NLI=${CHISPA_LOCAL_NLI:-}; export CHISPA_E5=${CHISPA_LOCAL_E5:-}; CK=${CHISPA_LOCAL_M1:-}
else
  NLI=$BASE/nli; export CHISPA_E5=$BASE/e5; CK=$BASE/mejor.pt
  ( bajar intfloat/multilingual-e5-small $E5_REV "$BASE/e5" '*.json' '*.safetensors' '*.txt' '*.model' 'tokenizer*' 'LICENSE*' && echo ok > "$BASE/e5.ok"
    bajar MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 $REVNLI "$BASE/nli" '*.json' '*.safetensors' 'spm.model' 'tokenizer*' 'LICENSE*' && echo ok > "$BASE/nli.ok" ) > "$BASE/bajadas.log" 2>&1 &
  s3cp "$CKPT" "$BASE/mejor.pt" || estado l2-zs "sin checkpoint"
  for _ in $(seq 60); do [ -f "$BASE/e5.ok" ] && break; sleep 5; done
fi
export HF_HUB_OFFLINE=1   # desde aquí todo es local
pred() {   # pred SISTEMA MINUTOS subcomando args...
  local n=$1 dur=$2; shift 2; local dl t0=$(date +%s); dl=$(limite "$dur")
  [ "$(quedan_min)" -gt $((4 + ESPERA_MIN + 2)) ] || { estado "$n" "OMITIDO: sin tiempo"; return 1; }
  ( cd "$REPO" && timeout $(( dl - t0 )) "$NPY" -m model.alt.lineas_base_v2_pred "$@" ) > "$OUT/pred-$n.log" 2>&1 \
    && estado "$n" "OK en $(( $(date +%s) - t0 )) s" || { tail -3 "$OUT/pred-$n.log"; fallo "$n"; estado "$n" "FALLO o corte"; return 1; }
}

# --- 3. kNN-e5 k=8, independiente de todo lo demás
echo "== 3. kNN $(date -u +%T) quedan $(quedan_min) min"
TV=$("$NPY" -c "import transformers; print(transformers.__version__)")
pred knn-vectores 9 vectores --dev "$BASE/dev" --vec "$BASE/vec" --transformers "$TV" --hilos "$HILOS" \
  && pred knn-e5-k8 3 knn --dev "$BASE/dev" --vec "$BASE/vec" --salida "$PRED/knn-e5-k8" "${MAXF[@]}"

# --- 4. L2 zero-shot
echo "== 4. L2 $(date -u +%T) quedan $(quedan_min) min"
if [ -f "$CK" ]; then pred l2-zs 8 l2 --m1 "$CK" --dev "$BASE/dev" --salida "$PRED/l2-zs" --dispositivo cpu --hilos "$HILOS" "${MAXF[@]}"; else estado l2-zs "FALLO: sin checkpoint"; fi

# --- 5. mDeBERTa-NLI: lo barato primero, tope de filas por tarea y de tiempo; lo que quede sin puntuar cuenta como fallo (cobertura en cifras.json)
echo "== 5. NLI $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ] || [ -f "$BASE/nli.ok" ]; then
  NF=(--max-filas "$NLI_FILAS"); [ "$LOCAL" = 1 ] && NF=("${MAXF[@]}")
  pred mdeberta-nli 99 nli --modelo "$NLI" --dev "$BASE/dev" --salida "$PRED/mdeberta-nli" --dispositivo cpu --hilos "$HILOS" --hasta "$(limite 99)" "${BF16[@]}" "${NF[@]}"
else estado mdeberta-nli "FALLO: no se bajó"; fi
cierre
