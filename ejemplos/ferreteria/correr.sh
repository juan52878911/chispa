#!/usr/bin/env bash
# Demo de 5 minutos del modo clásico: entrena un .chispa con 50 ejemplos propios (10 por opción),
# lo sirve con chispa-server y clasifica los mensajes de pruebas.txt. Solo CPU, sin descargar modelos.
# Requisitos: Go >= 1.27 y Python >= 3.9. Uso, desde la raíz del repositorio: ejemplos/ferreteria/correr.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(cd "$DIR/../.." && pwd)"
PUERTO="${PUERTO:-8090}"
TMP="$(mktemp -d)"
trap 'kill "${PID:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT

echo "1/3 Entrenando ferreteria.chispa (L0: TF-IDF de caracteres + logística)…"
PYTHONPATH="$RAIZ/sdk/python" python3 -m chispa train \
  --esquema "$DIR/esquema.json" --ejemplos "$DIR/ejemplos.csv" \
  --salida "$TMP/ferreteria.chispa" --sin-torre 2>&1 | grep -E '^/|L0 validación' | sed "s|$TMP/||" || true

echo "2/3 Compilando y arrancando chispa-server en 127.0.0.1:$PUERTO…"
(cd "$RAIZ/server" && go build -o "$TMP/chispa-server" ./cmd/chispa-server)
"$TMP/chispa-server" -addr "127.0.0.1:$PUERTO" -modelo "$TMP/ferreteria.chispa" >"$TMP/server.log" 2>&1 &
PID=$!
for _ in $(seq 1 50); do
  curl -sf "127.0.0.1:$PUERTO/healthz" >/dev/null 2>&1 && break
  sleep 0.1
done

echo "3/3 Clasificando pruebas.txt (mensajes que no están en los ejemplos):"
python3 - "$DIR" "$PUERTO" <<'PY'
import json, sys, urllib.request
d, puerto = sys.argv[1], sys.argv[2]
esq = json.load(open(f"{d}/esquema.json", encoding="utf-8"))
opciones = [{"id": k, "description": v} for k, v in esq["opciones"].items()]
for texto in open(f"{d}/pruebas.txt", encoding="utf-8").read().splitlines():
    if not texto.strip():
        continue
    cuerpo = {"state": texto, "questions": [{"id": "q1", "question": esq["pregunta"], "kind": "choice", "options": opciones}]}
    req = urllib.request.Request(f"http://127.0.0.1:{puerto}/v1/decide", json.dumps(cuerpo).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))["results"][0]
    p = r["probs"][r["answer"]]
    print(f"  {r['answer']:<11} p={p:.2f} {r['layer']} escalate={str(r['escalate']).lower():<5} ms={r['ms']}  «{texto}»")
PY
