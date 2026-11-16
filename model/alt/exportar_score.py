"""Fixture de paridad Python → Go del score calibrado (R10-T3b rub+cal) y de la preselección top-k. Reutiliza las funciones de `lineas_base_v2` y `lineas_base_v2b` (no reimplementa nada).

    python -m model.alt.exportar_score --dev DIR_DEV --pred DIR_PRED --salida DIR [--rubricas rubricas-r10-t3b.json] [--topk zs-topk]

DIR_PRED trae, como la salida de `lineas_base_v2b rubrica` + `llm_logprobs.py`, `rub/<tarea>.jsonl` (rúbrica sobre el test), `rub-sop/<tarea>.jsonl` (rúbrica
sobre el soporte de oro) y, para la preselección, `zs-topk/<tarea>.jsonl`. Escribe en DIR:
  score-<tarea>.json   niveles, umbrales (`umbrales` con los ejemplos de la rúbrica excluidos), las predicciones del soporte y de cada fila del test con su esperanza y su
                       nivel predicho (`top1` con umbrales), y la exactitud del test (filas sin predicción válida = fallo);
  topk-<tarea>.json    las candidatas que Python mandó a L3 en cada fila (las claves de `probs` de zs-topk). Los vectores e5 de la preselección no se exportan: el
                       ranking no se puede rehacer en Go desde aquí, así que el top-k se prueba en Go con datos sintéticos.
No lleva textos, solo ids y números.
"""
import argparse
import json
import sys
from pathlib import Path

from model.alt import lineas_base_v2 as lb
from model.alt import lineas_base_v2b as vb

TAREAS_SCORE = ("horeca-resenas", "jql-valor-educativo", "usajobs-nivel")
TAREAS_TOPK = ("finanzas-documentos",)


def _filas(ruta):
    return {r["id"]: r for r in lb.leer(ruta)}


def exportar_score(tarea, sop, ev, pred_test, pred_sop, excluidos):
    ops = lb.orden(ev[0])
    ids_sop = {r["id"] for r in sop}
    umb = vb.umbrales(sop, pred_sop, excluidos)
    ok, preds, _, _ = lb.evaluar_tarea(ev, pred_test, esp=umb)
    filas = []
    for r, y in zip(ev, preds):
        p = pred_test.get(r["id"])
        validas = bool(p and p.get("valida", True) and p.get("probs"))
        probs = {o: float(p["probs"].get(o, 0.0)) for o in ops} if validas else None
        e = lb.esperanza(ops, [probs[o] for o in ops]) if validas and sum(probs.values()) > 0 else None
        filas.append({"id": r["id"], "etiqueta": r["etiqueta"], "probs": probs, "esperanza": e, "nivel": y})
    soporte = []
    for r in sop:
        p = pred_sop.get(r["id"])
        validas = bool(p and p.get("valida", True) and p.get("probs"))
        soporte.append({"id": r["id"], "etiqueta": r["etiqueta"], "valida": validas, "probs": {o: float(p["probs"].get(o, 0.0)) for o in ops} if validas else None})
    return {"tarea": tarea, "niveles": ops, "umbrales": umb, "excluidos": sorted(ids_sop & set(excluidos)), "soporte": soporte, "filas": filas,
            "n_test": len(ev), "exactitud": round(float(ok.mean()), 4)}


def exportar_topk(tarea, ev, pred):
    filas = []
    for r in ev:
        p = pred.get(r["id"])
        filas.append({"id": r["id"], "etiqueta": r["etiqueta"], "candidatas": sorted(p["probs"]) if p and p.get("probs") else None})
    return {"tarea": tarea, "k": max((len(f["candidatas"]) for f in filas if f["candidatas"]), default=0), "filas": filas}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--rubricas", default=str(Path(__file__).with_name("rubricas-r10-t3b.json")))
    ap.add_argument("--topk", default="zs-topk")
    a = ap.parse_args(argv)
    dev = lb.cargar_dev(a.dev)
    rub = json.loads(Path(a.rubricas).read_text(encoding="utf-8"))
    excl = vb.ids_ejemplos(rub)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    pred = Path(a.pred)
    for t in TAREAS_SCORE:
        sop, ev = dev[t]
        out = exportar_score(t, sop, ev, _filas(pred / "rub" / (t + ".jsonl")), _filas(pred / "rub-sop" / (t + ".jsonl")), excl)
        (sal / ("score-%s.json" % t)).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
        print("score", t, "umbrales", out["umbrales"], "exactitud", out["exactitud"], "n", out["n_test"])
    for t in TAREAS_TOPK:
        f = pred / a.topk / (t + ".jsonl")
        if not f.exists():
            print("topk", t, "sin", f)
            continue
        out = exportar_topk(t, dev[t][1], _filas(f))
        (sal / ("topk-%s.json" % t)).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
        print("topk", t, "k", out["k"], "filas", len(out["filas"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
