#!/usr/bin/env bash
# Latencia de L0-L2 de Chispa (int8) con 1 vCPU y 1800 MB: `docker run --cpus=1 --memory=1800m` como sustituto de AWS Lambda.
# Pensado para una EC2 (c7i.2xlarge, DLAMI AL2023 con docker) que hace otras cosas a la vez: mira la carga antes y después (README, «Ruido»).
# No toca AWS salvo, si se pide, `aws s3 cp` de claves EXACTAS (nunca --recursive: el rol no tiene ListBucket).
#
#   bash correr.sh [-d ARTEFACTOS] [-o SALIDA] [-t textos.jsonl] [-n 400] [-r 5] [-m 128] [-k 6] [-s]
#     -d  directorio con los artefactos (por defecto ./artefactos; ver README: modelo-int8.chispa, tokenizer-e5.json, tokenizer-potion.json)
#     -o  directorio de salida (por defecto ./salida-AAAAMMDD-HHMMSS); deja latencia-1vcpu.json y los JSON por corrida
#     -t  jsonl con el campo «texto» (opcional; si no, textos sintéticos de longitud variada)
#     -n  peticiones en caliente (≥ 300; por defecto 400)     -r  arranques en frío (por defecto 5)
#     -m  tope de tokens de L2 (por defecto 128)              -k  opciones de la pregunta y de las torres al vuelo (por defecto 6)
#     -s  no vaciar la caché de páginas antes de cada arranque en frío (por defecto se vacía si hay permiso)
# Variables: CHISPA_IMAGEN (python:3.12-slim; mejor con @sha256:… fijado), CHISPA_ORT_VERSION (1.31.1), CHISPA_NUMPY_VERSION (2.5.4),
#   CHISPA_TOKENIZERS_VERSION (0.23.3), CHISPA_CPUSET (p. ej. 3: fija el contenedor a ese núcleo además de --cpus=1),
#   CHISPA_S3 (prefijo s3://… del que bajar modelo-int8.chispa si falta), CHISPA_E5_REV y CHISPA_POTION_REV (revisión de 40 hex en
#   Hugging Face de intfloat/multilingual-e5-small y minishlab/potion-multilingual-128M para bajar sus tokenizer.json si faltan).
set -uo pipefail

AQUI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=./artefactos; OUT=; TEXTOS=; N=400; R=5; MAXTOK=128; K=6; VACIAR=1
while getopts "d:o:t:n:r:m:k:sh" o; do
  case $o in
    d) DIR=$OPTARG;; o) OUT=$OPTARG;; t) TEXTOS=$OPTARG;; n) N=$OPTARG;; r) R=$OPTARG;; m) MAXTOK=$OPTARG;; k) K=$OPTARG;; s) VACIAR=0;;
    *) sed -n '2,17p' "$0"; exit 2;;
  esac
done
[ "$N" -ge 300 ] 2>/dev/null || { echo "-n debe ser un entero ≥ 300 (hay $N)" >&2; exit 2; }
[ "$R" -ge 1 ] 2>/dev/null || { echo "-r debe ser un entero ≥ 1" >&2; exit 2; }
[ -z "$OUT" ] && OUT=./salida-$(date +%Y%m%d-%H%M%S)
IMAGEN_BASE=${CHISPA_IMAGEN:-python:3.12-slim}
ORT=${CHISPA_ORT_VERSION:-1.31.1}; NUMPY=${CHISPA_NUMPY_VERSION:-2.5.4}; TOK=${CHISPA_TOKENIZERS_VERSION:-0.23.3}
mkdir -p "$DIR" "$OUT" || exit 1
DIR=$(cd "$DIR" && pwd); OUT=$(cd "$OUT" && pwd)
[ -n "$TEXTOS" ] && TEXTOS=$(cd "$(dirname "$TEXTOS")" && pwd)/$(basename "$TEXTOS")
command -v docker >/dev/null || { echo "no hay docker" >&2; exit 1; }
LOG=$OUT/log.txt; exec > >(tee -a "$LOG") 2>&1
echo "== latencia 1 vCPU $(date -u +%FT%TZ) · imagen base $IMAGEN_BASE · ORT $ORT · n=$N · frío x$R"

# ---- 1. artefactos que falten (solo si se dan variables; nunca se inventa una ruta)
bajar() {  # bajar URL DESTINO
  curl -fsSL --retry 3 -o "$2.parte" "$1" && mv "$2.parte" "$2" || { rm -f "$2.parte"; echo "no se pudo bajar $1"; return 1; }
}
if [ ! -s "$DIR/modelo-int8.chispa" ] && [ -n "${CHISPA_S3:-}" ]; then
  echo "bajando $CHISPA_S3/modelo-int8.chispa"; aws s3 cp --only-show-errors "$CHISPA_S3/modelo-int8.chispa" "$DIR/modelo-int8.chispa" || true
fi
if [ ! -s "$DIR/tokenizer-e5.json" ] && [[ "${CHISPA_E5_REV:-}" =~ ^[0-9a-f]{40}$ ]]; then
  bajar "https://huggingface.co/intfloat/multilingual-e5-small/resolve/$CHISPA_E5_REV/tokenizer.json" "$DIR/tokenizer-e5.json" || true
fi
if [ ! -s "$DIR/tokenizer-potion.json" ] && [[ "${CHISPA_POTION_REV:-}" =~ ^[0-9a-f]{40}$ ]]; then
  bajar "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/$CHISPA_POTION_REV/tokenizer.json" "$DIR/tokenizer-potion.json" || true
fi
echo "artefactos en $DIR:"; ls -l "$DIR" | sed 's/^/  /'
( cd "$DIR" && sha256sum -- * 2>/dev/null ) > "$OUT/artefactos.sha256" || true
[ -n "$TEXTOS" ] && [ -s "$TEXTOS" ] && sha256sum "$TEXTOS" >> "$OUT/artefactos.sha256"

# ---- 2. imagen: base fijada + paquetes con versión exacta + bench.py (se construye una vez; no cuenta en ningún tiempo medido)
docker pull -q "$IMAGEN_BASE" >/dev/null || echo "AVISO: no se pudo hacer pull de $IMAGEN_BASE (se usa la local si existe)"
case $IMAGEN_BASE in *@sha256:*) ;; *) echo "AVISO: la imagen base no va fijada por digest; se anota el digest resuelto en el JSON";; esac
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
cp "$AQUI/bench.py" "$CTX/bench.py"
cat > "$CTX/Dockerfile" <<DOCKER
FROM $IMAGEN_BASE
RUN pip install --no-cache-dir onnxruntime==$ORT numpy==$NUMPY tokenizers==$TOK
COPY bench.py /app/bench.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
ENTRYPOINT ["python", "/app/bench.py"]
DOCKER
ETIQ=chispa-latencia-1vcpu:$(sha256sum "$CTX/Dockerfile" "$CTX/bench.py" | sha256sum | cut -c1-12)
docker build -q -t "$ETIQ" "$CTX" >/dev/null || { echo "falló docker build" >&2; exit 1; }
DIGEST=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGEN_BASE" 2>/dev/null || echo "sin digest (imagen local)")
echo "imagen $ETIQ sobre $DIGEST"

# ---- 3. corridas
# shellcheck disable=SC2054  # la coma es parte del valor de --tmpfs
RUN_FLAGS=(--rm --cpus=1 --memory=1800m --memory-swap=1800m --network none --read-only --tmpfs /tmp:rw,size=512m
           -v "$DIR":/artefactos:ro -v "$OUT":/salida)
[ -n "${CHISPA_CPUSET:-}" ] && RUN_FLAGS+=(--cpuset-cpus "$CHISPA_CPUSET")
BENCH_ARGS=(--dir /artefactos --max-tok "$MAXTOK" --opciones "$K" --hilos 1)
if [ -n "$TEXTOS" ] && [ -s "$TEXTOS" ]; then RUN_FLAGS+=(-v "$TEXTOS":/entrada/textos.jsonl:ro); BENCH_ARGS+=(--textos /entrada/textos.jsonl); fi

carga() { cat /proc/loadavg 2>/dev/null || echo "0 0 0 sin-/proc"; }
PUEDE_VACIAR=0
if [ "$VACIAR" = 1 ]; then
  if sync && { echo 3 > /proc/sys/vm/drop_caches; } 2>/dev/null; then PUEDE_VACIAR=1
  elif sudo -n true 2>/dev/null && sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then PUEDE_VACIAR=1
  else echo "AVISO: no se puede vaciar la caché de páginas; el arranque en frío será optimista (ficheros ya en memoria)"; fi
fi
vaciar() { [ "$PUEDE_VACIAR" = 1 ] || return 0; sync; if [ -w /proc/sys/vm/drop_caches ]; then echo 3 > /proc/sys/vm/drop_caches; else echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; fi; }

echo "carga antes: $(carga)" | tee "$OUT/carga.txt"
echo "docker: $(docker version --format '{{.Server.Version}}' 2>/dev/null)" | tee -a "$OUT/carga.txt"
ahora_ms() { echo $(( $(date +%s%N) / 1000000 )); }
for i in $(seq 1 "$R"); do
  vaciar
  t0=$(ahora_ms)
  docker run "${RUN_FLAGS[@]}" "$ETIQ" "${BENCH_ARGS[@]}" --fase frio --salida /salida/frio-$i.json > "$OUT/frio-$i.txt" 2>&1; rc=$?
  t1=$(ahora_ms)
  echo "$i $rc $((t1 - t0)) $(carga | cut -d' ' -f1)" >> "$OUT/frio-docker.tsv"      # corrida, código, ms de docker run entero, load1
  echo "frío $i: rc=$rc docker run entero $((t1 - t0)) ms"
done
docker run "${RUN_FLAGS[@]}" "$ETIQ" "${BENCH_ARGS[@]}" --fase caliente --n "$N" --salida /salida/caliente.json > "$OUT/caliente.txt" 2>&1; rc=$?
echo "caliente: rc=$rc (137 = matado por memoria)"; echo "carga después: $(carga)" | tee -a "$OUT/carga.txt"
echo "$rc" > "$OUT/caliente.rc"

# ---- 4. un solo JSON de cifras
python3 - "$OUT" "$ETIQ" "$DIGEST" "$IMAGEN_BASE" "$ORT" "$PUEDE_VACIAR" "${CHISPA_CPUSET:-}" <<'PY'
import json, os, sys
out, etiq, digest, base, ort, vaciado, cpuset = sys.argv[1:8]
def cargar(n):
    try:
        with open(os.path.join(out, n), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0; a = int(k); b = min(a + 1, len(xs) - 1)
    return xs[a] + (xs[b] - xs[a]) * (k - a)
frios, docker = [], {}
try:
    for l in open(os.path.join(out, "frio-docker.tsv")):
        i, rc, ms, load = l.split(); docker[int(i)] = {"rc": int(rc), "docker_run_entero_ms": int(ms), "load1": float(load)}
except OSError:
    pass
for i in sorted(docker):
    j = cargar("frio-%d.json" % i)
    frios.append(dict(docker[i], **({"ok": True, "desde_arranque_proceso_ms": j.get("tiempo_hasta_primera_respuesta_desde_arranque_proceso_ms"),
                                    "carga_ms": j.get("carga_ms"), "import_ms": j.get("import_ms"), "primera_peticion_ms": j.get("primera_peticion_ms"),
                                    "segunda_peticion_ms": j.get("segunda_peticion_ms"), "rss_pico_mb": j.get("rss_pico_mb"),
                                    "cgroup": j.get("cgroup_al_final"), "omitidos": j.get("omitidos")} if j else {"ok": False})))
res = {"que_es": "latencia de L0-L2 de Chispa en contenedor con 1 vCPU y 1800 MB (sustituto de Lambda; no es Lambda)",
       "contenedor": {"flags": "--cpus=1 --memory=1800m --memory-swap=1800m --network none --read-only", "cpuset": cpuset or None,
                      "imagen": etiq, "imagen_base": base, "digest_base": digest, "onnxruntime_pin": ort,
                      "cache_de_paginas_vaciada_antes_de_cada_frio": vaciado == "1"},
       "frio": {"corridas": frios}, "caliente": cargar("caliente.json"),
       "caliente_rc": int(open(os.path.join(out, "caliente.rc")).read().strip() or 0)}
ok = [f for f in frios if f.get("ok") and f.get("desde_arranque_proceso_ms")]
if ok:
    for k in ("desde_arranque_proceso_ms", "docker_run_entero_ms"):
        xs = [f[k] for f in ok]
        res["frio"]["resumen_" + k] = {"n": len(xs), "min": min(xs), "mediana": pct(xs, 50), "max": max(xs)}
omit = {}
for f in ok:
    omit.update(f.get("omitidos") or {})
if res["caliente"]:
    omit.update(res["caliente"].get("omitidos") or {})
res["omitidos"] = omit
res["advertencias"] = []
if res["caliente"] is None:
    res["advertencias"].append("la corrida en caliente no dejó JSON (código %d; 137 = OOM con 1800 MB)" % res["caliente_rc"])
if not ok:
    res["advertencias"].append("ninguna corrida en frío dejó JSON")
res["carga_instancia"] = open(os.path.join(out, "carga.txt")).read().strip().splitlines()
with open(os.path.join(out, "latencia-1vcpu.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False); f.write("\n")
print(json.dumps({k: res[k] for k in ("omitidos", "advertencias")}, ensure_ascii=False))
PY
[ $? = 0 ] || echo "AVISO: no se pudo armar latencia-1vcpu.json; quedan los JSON por corrida en $OUT" >&2
echo "listo: $OUT/latencia-1vcpu.json"
