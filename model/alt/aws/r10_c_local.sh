#!/usr/bin/env bash
# Prueba en seco local de los dos ud de R10-c (sin AWS, sin red, sin torch): fixtures sintéticos con los nombres de las 6 tareas dev, el ud de la
# g5 (llm_r7b.sh) con el vLLM falso de R9 y el ud de la c7i (r10_c_medir.sh) con vectores falsos, encadenados (la c7i lee lo que etiquetó la «g5»).
# Uso: bash model/alt/aws/r10_c_local.sh [SHA]       Resultados en /tmp/chispa-r10c-dryrun/{g5,c7i}/res
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:-$(git rev-parse HEAD)}; r5=0bcbd78
D=${CHISPA_DRY_DIR:-/tmp/chispa-r10c-dryrun}; chmod -R u+w "$D" 2>/dev/null || true; rm -rf "$D"; mkdir -p "$D/dev" "$D/fx"
python3 - "$D/fx" "$D/dev" <<'PY'
import json, random, sys
sys.path.insert(0, ".")
from model.alt.llm_bateria import a_fila, intercalar
fx, dev = sys.argv[1:]
m = json.load(open("model/alt/bateria-v2-manifiesto.json"))
nombres = sorted(t["nombre"] for t in m["tareas"] if t["particion"] == "dev")
r = random.Random(3)
tipos = {"horeca-resenas": "score", "jql-valor-educativo": "score", "usajobs-nivel": "score"}
nops = {"finanzas-documentos": 59, "usajobs-nivel": 11, "cfpb-quejas": 9, "horeca-resenas": 5, "jql-valor-educativo": 5, "congreso-eeuu-partido": 2}
oro, fondo, fuente = [], [], []
por = {}
for c in nombres:
    tipo = tipos.get(c, "choice")
    ops = [str(i + 1) for i in range(nops[c])] if tipo == "score" else ["c%02d" % i for i in range(nops[c])]
    base = dict(tarea="t_" + c, tipo=tipo, pregunta="¿Qué clase es?", opciones=ops, descripciones={o: "clase " + o for o in ops}, idioma="es", licencia="mit", uso_comercial=True)
    def fila(split, i, o):
        return dict(base, id="%s:%s:%d" % (c, split, i), texto="texto %s palabra%s ñ <b>x</b> %d" % (c, o, r.randrange(10 ** 6)), etiqueta=o, split=split, fuente="fx")
    tr = [fila("train", n, o) for n, o in enumerate(o for o in ops for _ in range(8))]
    te = [fila("test", n, ops[n % len(ops)]) for n in range(24)]
    with open("%s/%s.jsonl" % (dev, c), "w") as fh:
        fh.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tr + te))
    oro += [dict(x, oro8=True) for x in tr] + [dict(base, id="%s:oro32:%d" % (c, n), texto="texto %s palabra%s extra %d" % (c, o, n), etiqueta=o, split="train", fuente="fx", oro8=False)
                                              for n, o in enumerate(o for o in ops for _ in range(6))]
    fl = []
    for n in range(40):
        o = ops[n % len(ops)]
        f = a_fila(dict(base, id="fondo-v2:%s:%d" % (c, n), texto="fondo %s palabra%s %d" % (c, o, n)), c, etiqueta=False)
        fl.append(f); fuente.append(dict(id=f["id"], conjunto=c, etiqueta=o))
    por[c] = fl
fondo = intercalar([f for v in por.values() for f in v])
w = lambda p, rs: open(p, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs))  # noqa: E731
w(fx + "/oro32.jsonl", oro); w(fx + "/fondo.jsonl", fondo); w(fx + "/fondo-fuente.jsonl", fuente)
print(len(fondo), "filas de fondo,", len(oro), "de oro32,", len(nombres), "tareas")
PY
echo "== empaquetar g5"
bash model/alt/aws/empaquetar_r10_c_lp.sh "$sha" "$r5" "$D/fx/fondo.jsonl" "$D/s3lp"
U=model/alt/aws/lanzados/ud-r10-c-lp-$sha.sh; bash -n "$U"; rm -f "$U"
python3 model/alt/aws/r9_vllm_falso.py --puerto 8766 --ms 5 & FALSO=$!
trap 'kill $FALSO 2>/dev/null' EXIT
sleep 1
git show "$sha:model/alt/aws/llm_r7b.sh" | sed -e "s#__CHISPA_S3__#$D/s3lp#" -e "s#__N_REET__#$(wc -l < "$D/s3lp/reetiqueta.jsonl" | tr -d ' ')#" > "$D/ud-g5.sh"
bash -n "$D/ud-g5.sh"
echo "== ud g5 (local, vLLM falso)"
CHISPA_LOCAL=1 CHISPA_BASE="$D/g5" CHISPA_LOCAL_URL=http://127.0.0.1:8766 CHISPA_LOCAL_MODELO=qwen3-4b CHISPA_LOCAL_MAX=${CHISPA_LOCAL_MAX:-120} CHISPA_HILOS=4 bash "$D/ud-g5.sh" || true
cat "$D/g5/res/estado.tsv"
LP="$D/g5/res/lp_reetiqueta.jsonl"; [ -s "$LP" ] || { echo "la g5 local no escribió lp_reetiqueta.jsonl"; exit 1; }
echo "== empaquetar c7i"
bash model/alt/aws/empaquetar_r10_c_medir.sh "$sha" "$D/dev" "$D/fx/oro32.jsonl" "$D/s3lp/reetiqueta.jsonl" "$D/fx/fondo-fuente.jsonl" "$LP" "$D/s3m"
U=model/alt/aws/lanzados/ud-r10-c-medir-$sha.sh
sed -e "s#s3://chispa-pruebas-cuenta-aws/r10-c-medir-$sha#$D/s3m#" "$U" > "$D/ud-c7i.sh"; rm -f "$U"; bash -n "$D/ud-c7i.sh"
echo "== ud c7i (local, vectores falsos)"
CHISPA_LOCAL=1 CHISPA_BASE="$D/c7i" bash "$D/ud-c7i.sh" || true
cat "$D/c7i/res/estado.tsv"
ls "$D/c7i/res"
