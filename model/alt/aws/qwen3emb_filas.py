"""L2-qwen3emb: prepara las entradas del servidor Go desde los 6 jsonl de DESARROLLO de la batería v2 (solo numpy; sin torch).

    python3 -m model.alt.aws.qwen3emb_filas filas  --dev DIR_DEV --salida DIR      soporte.jsonl, consultas.jsonl, tareas.json
    python3 -m model.alt.aws.qwen3emb_filas config --tareas tareas.json --bancos DIR --onnx RUTA --tok RUTA --vecinos 32 --salida cfg.json [--lib]

soporte.jsonl: el soporte de oro de cada tarea, 8 por clase como `muestrear(ys, 8, 0)` de R11 (mismo índice y mismo orden: ascendente), con tarea, etiqueta,
pregunta y texto: es la entrada de server/cmd/chispa-banco-qwen. consultas.jsonl: las filas de test (tarea, id, tipo, pregunta, texto, opciones, etiqueta,
descripciones) para model/alt/aws/r10_t5_cliente.py. tareas.json: {tarea: {preguntas, n_soporte, n_test}}.
config: la del servidor para la medida: kNN con Qwen, umbral 0 y sin tareas «siempre» (nada escala: contesta solo L2), sin L3.
"""
import argparse
import json
from pathlib import Path

from model.alt.fusion_knn_l2 import muestrear
from model.alt.lineas_base_v2 import cargar_dev

ORO_POR_CLASE = 8


def filas(dev, salida):
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    tareas, ns, nt = {}, 0, 0
    with open(sal / "soporte.jsonl", "w", encoding="utf-8") as fs, open(sal / "consultas.jsonl", "w", encoding="utf-8") as fc:
        for t, (sop, ev) in cargar_dev(dev).items():
            ys = [r["etiqueta"] for r in sop]
            idx = muestrear(ys, ORO_POR_CLASE, 0)
            preg = sorted({r["pregunta"] for r in sop + ev})
            for i in idx:
                r = sop[i]
                fs.write(json.dumps({"tarea": t, "id": r["id"], "etiqueta": r["etiqueta"], "pregunta": r["pregunta"], "texto": r["texto"]}, ensure_ascii=False) + "\n")
            for r in ev:
                fc.write(json.dumps({"tarea": t, "id": r["id"], "tipo": r.get("tipo", "choice"), "pregunta": r["pregunta"], "texto": r["texto"], "opciones": r["opciones"],
                                     "etiqueta": r["etiqueta"], "descripciones": r.get("descripciones") or {}}, ensure_ascii=False) + "\n")
            tareas[t] = {"preguntas": preg, "n_soporte": len(idx), "n_soporte_total": len(sop), "n_test": len(ev)}
            ns, nt = ns + len(idx), nt + len(ev)
    (sal / "tareas.json").write_text(json.dumps(tareas, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return "tareas=%d soporte=%d test=%d" % (len(tareas), ns, nt)


def config(tareas, bancos, onnx, tok, vecinos, salida, sesiones=1):
    ts = json.load(open(tareas, encoding="utf-8"))
    knn = {"umbral": 0.0, "siempre": [], "codificador": "qwen3-emb", "qwen_onnx": str(onnx), "qwen_tok": str(tok), "qwen_sesiones": sesiones, "vecinos": vecinos,
           "tareas": {t: {"banco": str(Path(bancos) / ("banco-" + t)), "preguntas": v["preguntas"]} for t, v in sorted(ts.items())}}
    Path(salida).write_text(json.dumps({"knn": knn}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("filas")
    a.add_argument("--dev", required=True)
    a.add_argument("--salida", required=True)
    b = sp.add_parser("config")
    for n in ("tareas", "bancos", "onnx", "tok", "salida"):
        b.add_argument("--" + n, required=True)
    b.add_argument("--vecinos", type=int, default=32)
    b.add_argument("--sesiones", type=int, default=1)
    x = ap.parse_args(argv)
    if x.cmd == "filas":
        print(filas(x.dev, x.salida))
    else:
        config(x.tareas, x.bancos, x.onnx, x.tok, x.vecinos, x.salida, x.sesiones)


if __name__ == "__main__":
    main()
